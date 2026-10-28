"""
Обучение добавок LoRA для режимов адаптации (mode: lora в конфиге).

LoRA — параметрически эффективная адаптация: к выбранным линейным слоям
добавляется произведение двух узких матриц, обучаются только они, остальные
веса заморожены. Обучение идёт на размеченных кадрах домена-источника вне
теста (scripts/make_adapt_split.py), целевые домены при обучении не видны —
так же, как при подборе порога. Это и проверяется: помогает ли дообучение
на ясном дне ночью, в дождь и снег, и переносится ли порог после него.

Гиперпараметры задаются в конфиге до обучения. После каждой эпохи считается
loss на части val, сохраняются веса лучшей эпохи и журнал обучения.

Запуск:
    python -m src.train_lora --config configs/experiment_bdd.yaml --model grounding-dino-base-lora
    python -m src.train_lora ... --limit 4 --epochs 1 --out /tmp/x   # проверка, что цикл работает
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.acdc import ACDCDetection, load_adaptation  # noqa: E402
from src.models.adapters import (  # noqa: E402
    QwenVLAdapter,
    build_grounding_text,
    qwen_answer,
    qwen_input_scale,
)
from src.run_inference import build_prompts, environment_fingerprint  # noqa: E402


def lora_model(model, train_cfg: dict):
    """Обернуть модель добавками LoRA по регулярному выражению имён слоёв."""
    from peft import LoraConfig, get_peft_model

    model = get_peft_model(model, LoraConfig(
        r=train_cfg["rank"],
        lora_alpha=train_cfg["alpha"],
        lora_dropout=train_cfg["dropout"],
        target_modules=train_cfg["target_modules"],
    ))
    # Добавки обучаются в float32 даже при bfloat16 модели: в bfloat16 шаги
    # AdamW порядка 1e-4 теряются в округлении
    for p in model.parameters():
        if p.requires_grad:
            p.data = p.data.float()
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    total = sum(p.numel() for p in model.parameters())
    print(f"Обучаемых параметров: {trainable / 1e6:.2f} млн из {total / 1e6:.0f} млн "
          f"({100 * trainable / total:.2f}%)", flush=True)
    return model, trainable, total


def batches(samples: list, size: int, rng: np.random.Generator | None):
    order = rng.permutation(len(samples)) if rng is not None else np.arange(len(samples))
    for i in range(0, len(order), size):
        yield [samples[j] for j in order[i: i + size]]


# --------------------------------------------------------------------------- #
#                              Grounding DINO                                 #
# --------------------------------------------------------------------------- #

class GroundingDINOTask:
    """Функцию потерь (венгерское сопоставление, focal loss, L1 и GIoU по
    боксам) считает сама модель transformers, если передать ей labels.

    Номер класса в labels — номер фразы в тексте запроса: модель делит текст
    на группы токенов по точкам, и группа i соответствует классу i. Текст
    тот же, что при инференсе (build_grounding_text), поэтому нумерация
    совпадает.
    """

    def __init__(self, model_cfg: dict, prompts: list[str], device: str) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch, self.device = torch, device
        self.processor = AutoProcessor.from_pretrained(model_cfg["checkpoint"])
        self.model = AutoModelForZeroShotObjectDetection.from_pretrained(model_cfg["checkpoint"])
        self.text = build_grounding_text(prompts)[0]

    def prepare(self, train_cfg: dict) -> tuple[int, int]:
        self.model, trainable, total = lora_model(self.model, train_cfg)
        self.model.to(self.device)
        return trainable, total

    def train_mode(self) -> None:
        """Режим обучения только для частей с добавками LoRA. Swin и BERT
        заморожены и остаются в режиме инференса: иначе в Swin включается
        случайное отключение блоков (drop path), и добавки учатся на
        зашумлённых признаках, которых при инференсе не будет."""
        self.model.train()
        for name, module in self.model.named_modules():
            if name.endswith(("model.backbone", "model.text_backbone")):
                module.eval()

    def loss(self, samples) -> "torch.Tensor":
        torch = self.torch
        images = [Image.open(s.image_path).convert("RGB") for s in samples]
        inputs = self.processor(images=images, text=[self.text] * len(images),
                                return_tensors="pt").to(self.device)
        labels = []
        for s in samples:
            b = s.boxes / np.array([s.width, s.height, s.width, s.height], dtype=np.float32)
            cxcywh = np.stack([(b[:, 0] + b[:, 2]) / 2, (b[:, 1] + b[:, 3]) / 2,
                               b[:, 2] - b[:, 0], b[:, 3] - b[:, 1]], 1)
            labels.append({
                "class_labels": torch.as_tensor(s.labels, dtype=torch.long, device=self.device),
                "boxes": torch.as_tensor(cxcywh, dtype=torch.float32, device=self.device),
            })
        return self.model(**inputs, labels=labels).loss


# --------------------------------------------------------------------------- #
#                                Qwen2.5-VL                                   #
# --------------------------------------------------------------------------- #

class QwenTask:
    """Обучение на правильный ответ: тот же диалог, что при инференсе, и
    ответ из разметки в формате инструкции. Loss считается только по токенам
    ответа — инструкцию и токены изображения модель не учится повторять."""

    def __init__(self, model_cfg: dict, prompts: list[str], device: str) -> None:
        import torch
        from transformers import AutoProcessor, Qwen2_5_VLForConditionalGeneration

        self.torch, self.device, self.prompts = torch, device, prompts
        self.processor = AutoProcessor.from_pretrained(model_cfg["checkpoint"])
        # bfloat16 на GPU; на CPU (проверка цикла) обратный проход в bfloat16
        # поддерживается не везде
        self.model = Qwen2_5_VLForConditionalGeneration.from_pretrained(
            model_cfg["checkpoint"],
            torch_dtype=torch.bfloat16 if device.startswith("cuda") else torch.float32)

    def prepare(self, train_cfg: dict) -> tuple[int, int]:
        # Пересчёт активаций при обратном проходе экономит память: иначе 7B
        # модель с кадром ~1200 токенов не помещается вместе с градиентами
        self.model.config.use_cache = False
        self.model.gradient_checkpointing_enable(gradient_checkpointing_kwargs={"use_reentrant": False})
        self.model.enable_input_require_grads()
        self.model, trainable, total = lora_model(self.model, train_cfg)
        self.model.to(self.device)
        return trainable, total

    def train_mode(self) -> None:
        self.model.train()

    def encode(self, sample):
        torch = self.torch
        image = Image.open(sample.image_path).convert("RGB")
        prefix = self.processor.apply_chat_template(
            [QwenVLAdapter.user_turn(image, self.prompts)],
            tokenize=False, add_generation_prompt=True)
        answer = qwen_answer(sample.boxes, sample.labels, self.prompts,
                             qwen_input_scale(self.processor, image))
        full = self.processor(text=[prefix + answer + "<|im_end|>"], images=[image],
                              return_tensors="pt")
        n_prefix = self.processor(text=[prefix], images=[image],
                                  return_tensors="pt")["input_ids"].shape[1]
        labels = full["input_ids"].clone()
        labels[:, :n_prefix] = -100
        return {**full, "labels": labels}

    def loss(self, samples) -> "torch.Tensor":
        # Кадры одного размера, но ответы разной длины — по одному кадру за
        # проход, батч набирается накоплением градиента
        losses = []
        for s in samples:
            batch = {k: v.to(self.device) for k, v in self.encode(s).items()}
            with self.torch.autocast("cuda", dtype=self.torch.bfloat16,
                                     enabled=self.device.startswith("cuda")):
                losses.append(self.model(**batch).loss)
        return self.torch.stack(losses).mean()


TASKS = {"grounding-dino": GroundingDINOTask, "qwen2.5-vl": QwenTask}


# --------------------------------------------------------------------------- #

def evaluate_loss(task, samples: list, batch_size: int) -> float:
    torch = task.torch
    task.model.eval()
    total, n = 0.0, 0
    with torch.no_grad():
        for b in batches(samples, batch_size, None):
            total += float(task.loss(b)) * len(b)
            n += len(b)
    task.train_mode()
    return total / max(n, 1)


def main() -> int:
    ap = argparse.ArgumentParser(description="Обучение LoRA для режима адаптации")
    ap.add_argument("--config", default="configs/experiment_bdd.yaml")
    ap.add_argument("--model", required=True, help="ключ модели с mode: lora")
    ap.add_argument("--out", default=None, help="каталог весов; по умолчанию lora из конфига")
    ap.add_argument("--limit", type=int, default=None, help="число кадров train и val (проверка)")
    ap.add_argument("--epochs", type=int, default=None, help="переопределить число эпох (проверка)")
    ap.add_argument("--device", default=None)
    ap.add_argument("--overwrite", action="store_true")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    mcfg = cfg["models"][args.model]
    if mcfg.get("mode") != "lora":
        sys.exit(f"У модели {args.model} нет mode: lora")
    tcfg = dict(mcfg["train"])
    if args.epochs:
        tcfg["epochs"] = args.epochs
    out = Path(args.out or mcfg["lora"])
    if (out / "adapter_config.json").exists() and not args.overwrite:
        print(f"[skip] {out} уже обучена — для переобучения добавьте --overwrite")
        return 0
    device = args.device or cfg["device"]

    import torch

    torch.manual_seed(cfg.get("seed", 0))
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True

    train: ACDCDetection = load_adaptation(cfg, "train")
    val: ACDCDetection = load_adaptation(cfg, "val")
    train_s, val_s = list(train)[: args.limit], list(val)[: args.limit]
    prompt_cfg = yaml.safe_load(open(cfg["prompts_file"], encoding="utf-8"))
    prompts = build_prompts(prompt_cfg, cfg["default_prompt_variant"], train.classes)
    print(f"{args.model}: train {len(train_s)} кадров, val {len(val_s)}, "
          f"домен {train.domain}, запрос {prompts}", flush=True)

    task = TASKS[mcfg["adapter"]](mcfg, prompts, device)
    trainable, total = task.prepare(tcfg)
    params = [p for p in task.model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(params, lr=tcfg["lr"], weight_decay=1e-4)

    bs, accum = tcfg["batch_size"], tcfg.get("grad_accum", 1)
    steps = tcfg["epochs"] * math.ceil(len(train_s) / (bs * accum))
    warmup = min(50, max(steps // 10, 1))
    # Линейный разогрев и косинусное затухание шага обучения
    sched = torch.optim.lr_scheduler.LambdaLR(opt, lambda i: min((i + 1) / warmup, 1.0)
                                              * 0.5 * (1 + math.cos(math.pi * min(i / steps, 1.0))))

    out.mkdir(parents=True, exist_ok=True)
    log, best = [], float("inf")
    val0 = evaluate_loss(task, val_s, bs)
    print(f"эпоха 0: val loss {val0:.4f} (до обучения)", flush=True)
    log.append({"epoch": 0, "val_loss": val0})
    rng = np.random.default_rng(cfg.get("seed", 0))
    task.train_mode()
    step = 0
    for epoch in range(1, tcfg["epochs"] + 1):
        t0, run, n = time.time(), 0.0, 0
        for i, b in enumerate(batches(train_s, bs, rng)):
            loss = task.loss(b)
            (loss / accum).backward()
            run, n = run + float(loss) * len(b), n + len(b)
            if (i + 1) % accum == 0:
                torch.nn.utils.clip_grad_norm_(params, 0.1 if mcfg["adapter"] == "grounding-dino" else 1.0)
                opt.step()
                sched.step()
                opt.zero_grad(set_to_none=True)
                step += 1
                if step % 20 == 0:
                    print(f"  шаг {step}/{steps}: loss {run / n:.4f}", flush=True)
        opt.zero_grad(set_to_none=True)
        val_loss = evaluate_loss(task, val_s, bs)
        log.append({"epoch": epoch, "train_loss": run / max(n, 1), "val_loss": val_loss,
                    "seconds": round(time.time() - t0, 1)})
        mark = ""
        if val_loss < best:
            best = val_loss
            task.model.save_pretrained(out)
            mark = " — сохранено"
        print(f"эпоха {epoch}: train {run / max(n, 1):.4f}, val {val_loss:.4f}, "
              f"{time.time() - t0:.0f} с{mark}", flush=True)

    (out / "train_log.json").write_text(json.dumps({
        "model": args.model,
        "base_checkpoint": mcfg["checkpoint"],
        "train": tcfg,
        "n_train": len(train_s),
        "n_val": len(val_s),
        "domain": train.domain,
        "prompts": prompts,
        "trainable_params": trainable,
        "total_params": total,
        "best_val_loss": best,
        "epochs": log,
        "env": environment_fingerprint(),
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"Готово: {out} (лучший val loss {best:.4f})")
    return 0


if __name__ == "__main__":
    sys.exit(main())
