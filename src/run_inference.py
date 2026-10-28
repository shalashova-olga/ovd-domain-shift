"""
Прогон инференса по единому протоколу: одна модель, один домен, один вариант промпта.

Результат пишется в JSONL — одна строка на изображение. Формат выбран ради
устойчивости: прогон на 8 моделях × 5 доменов × 3 промпта занимает часы, и
падение на последнем домене не должно обнулять уже посчитанное. Готовые
комбинации пропускаются при повторном запуске.

Запуск:
    python -m src.run_inference --model owlv2 --domain fog --prompt-variant plain
    python -m src.run_inference --all          # всё по конфигу
"""

from __future__ import annotations

import argparse
import gc
import json
import platform
import subprocess
import sys
import time
import traceback
from dataclasses import asdict
from pathlib import Path

import numpy as np
import yaml
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.acdc import ACDCDetection, load_domains, predictions_root  # noqa: E402
from src.models.adapters import build_detector  # noqa: E402


def environment_fingerprint() -> dict:
    """Отпечаток окружения в каждом файле результатов.

    Без него воспроизвести числа через полгода невозможно: версии torch и
    transformers меняют постобработку детекции, а модель весов на HuggingFace
    может быть обновлена без смены имени.
    """
    fp = {
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "timestamp": time.strftime("%Y-%m-%dT%H:%M:%S"),
    }
    for mod in ("torch", "transformers", "ultralytics", "numpy"):
        try:
            fp[mod] = __import__(mod).__version__
        except Exception:
            fp[mod] = "not installed"
    try:
        import torch

        if torch.cuda.is_available():
            fp["gpu"] = torch.cuda.get_device_name(0)
            fp["cuda"] = torch.version.cuda
    except Exception:
        pass
    try:
        fp["git_commit"] = subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], stderr=subprocess.DEVNULL, text=True
        ).strip()
    except Exception:
        fp["git_commit"] = "unknown"
    return fp


def build_prompts(prompt_cfg: dict, variant: str, classes: list[str]) -> list[str]:
    """Текстовые запросы для варианта: шаблон, либо явное переопределение класса.

    Переопределения (class_overrides) нужны классам, которые плохо выражаются
    одним словом, — прежде всего rider. Без них разница в AP rider между
    моделями объяснялась бы случайной формулировкой, а не моделью.
    """
    overrides = prompt_cfg.get("class_overrides") or {}
    template = prompt_cfg["variants"][variant]
    return [overrides.get(c, {}).get(variant, template.format(cls=c)) for c in classes]


def sensitivity_subset(dataset: ACDCDetection, n: int | None, seed: int) -> ACDCDetection:
    """Фиксированная подвыборка кадров для альтернативных формулировок запроса.

    Выбор зависит только от списка кадров и seed, поэтому одинаков для всех
    моделей и вариантов — разброс mAP сравнивается на одних и тех же кадрах.
    """
    if not n or len(dataset) <= n:
        return dataset
    ids = sorted(s.image_id for s in dataset)
    keep = np.random.default_rng(seed).choice(ids, size=n, replace=False)
    return dataset.filtered(keep)


def run_one(
    model_key: str,
    model_cfg: dict,
    detector,
    dataset: ACDCDetection,
    prompts: list[str],
    prompt_variant: str,
    out_path: Path,
    warmup_n: int = 3,
    limit: int | None = None,
) -> None:
    """Прогнать одну модель по одному домену с одним вариантом промпта.

    Файл пишется под временным именем и переименовывается в конце: прерванный
    прогон не должен оставить обрезанный журнал, который при повторном запуске
    сочли бы готовым.
    """
    samples = list(dataset)[:limit] if limit else list(dataset)
    if not samples:
        print(f"[warn] пустой набор: {dataset.domain}")
        return

    # Прогрев обязателен до первого замера, иначе p50 latency включает
    # компиляцию ядер и ленивую инициализацию весов.
    warm_img = Image.open(samples[0].image_path).convert("RGB")
    for _ in range(warmup_n):
        detector.predict(warm_img, prompts, measure_vram=False)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp_path = out_path.with_suffix(".partial")
    with open(tmp_path, "w", encoding="utf-8") as f:
        f.write(json.dumps({
            "_meta": {
                "model": model_key,
                "adapter": model_cfg["adapter"],
                "checkpoint": model_cfg["checkpoint"],
                "license": detector.license,
                "domain": dataset.domain,
                "split": dataset.split,
                "prompt_variant": prompt_variant,
                "prompts": prompts,
                "classes": dataset.classes,
                "box_threshold": detector.box_threshold,
                "n_images": len(samples),
                "env": environment_fingerprint(),
            }
        }, ensure_ascii=False) + "\n")

        for i, s in enumerate(samples):
            image = Image.open(s.image_path).convert("RGB")
            dets, validity, latency, vram = detector.predict(image, prompts)

            row = {
                "image_id": s.image_id,
                "boxes": dets.boxes.round(2).tolist(),
                "scores": dets.scores.round(5).tolist(),
                "labels": dets.labels.tolist(),
                "validity": asdict(validity),
                "latency_ms": round(latency, 2),
                "peak_vram_mb": round(vram, 1),
            }
            # Текст ответа VLM сохраняется целиком: по нему разбираются
            # невалидные ответы и правится промпт, а повторный прогон стоит часы.
            if isinstance(dets.raw, dict) and "answer" in dets.raw:
                row["raw_answer"] = dets.raw["answer"]
            f.write(json.dumps(row, ensure_ascii=False) + "\n")

            if (i + 1) % 50 == 0:
                print(f"  {model_key}/{dataset.domain}/{prompt_variant}: {i+1}/{len(samples)}",
                      flush=True)

    tmp_path.replace(out_path)
    print(f"[ok] {out_path}", flush=True)


def main() -> None:
    ap = argparse.ArgumentParser(description="Инференс по единому протоколу")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--model", help="ключ модели из конфига")
    ap.add_argument("--domain", help="домен ACDC")
    ap.add_argument("--prompt-variant", default=None)
    ap.add_argument("--all", action="store_true", help="прогнать все комбинации")
    ap.add_argument("--limit", type=int, default=None, help="ограничить число изображений (отладка)")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--subset", choices=["dev", "test", "all"], default=None,
                    help="часть разбиения; по умолчанию eval_subset из конфига (test)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    prompt_cfg = yaml.safe_load(open(cfg["prompts_file"], encoding="utf-8"))

    if not args.all and not args.model:
        ap.error("укажите --model или --all")
    models = cfg["models"] if args.all else {args.model: cfg["models"][args.model]}
    domains = [args.domain] if args.domain else cfg["domains"]
    subset = args.subset or cfg.get("eval_subset", "test")
    if args.prompt_variant:
        variants = [args.prompt_variant]
    elif (args.all or not args.domain) and subset != "dev":
        variants = list(prompt_cfg["variants"])
    else:
        variants = [cfg["default_prompt_variant"]]

    datasets = {d: ds for d, ds in load_domains(cfg, subset).items() if d in domains}
    if not datasets:
        sys.exit("Нет ни одного домена с разметкой — проверьте data_root.")
    print(f"Подвыборка: {subset}; кадров: "
          + ", ".join(f"{d} {len(ds)}" for d, ds in datasets.items()), flush=True)

    out_root = predictions_root(cfg, subset)
    if args.limit:
        # Отладочный прогон на части кадров пишется отдельно: иначе полный
        # прогон счёл бы обрезанные журналы готовыми и пропустил их.
        out_root = out_root / "_debug"
    failed = []
    for model_key, model_cfg in models.items():
        # Список работ модели: чувствительность к промпту замеряется только на
        # части доменов — полная сетка 8×5×3 избыточна, а разброс между
        # вариантами достаточно оценить на источнике и худшем домене.
        jobs = []
        for domain, dataset in datasets.items():
            for variant in variants:
                if (variant != cfg["default_prompt_variant"]
                        and domain not in cfg["prompt_sensitivity_domains"]):
                    continue
                out_path = out_root / model_key / f"{domain}__{variant}.jsonl"
                if out_path.exists() and not args.overwrite:
                    print(f"[skip] {out_path} уже существует")
                    continue
                if variant != cfg["default_prompt_variant"]:
                    dataset = sensitivity_subset(
                        dataset, cfg.get("prompt_sensitivity_max_images"), cfg.get("seed", 0))
                jobs.append((dataset, variant, out_path))
        if not jobs:
            continue

        # Модель загружается один раз на все домены: у Qwen2.5-VL загрузка
        # весов занимает минуты. Падение одной модели не останавливает
        # ночной прогон остальных.
        detector = None
        try:
            classes = next(iter(datasets.values())).classes
            detector = build_detector(
                model_cfg["adapter"],
                class_names=classes,
                device=cfg["device"],
                checkpoint=model_cfg["checkpoint"],
                box_threshold=model_cfg.get("box_threshold", 0.05),
            )
            for dataset, variant, out_path in jobs:
                prompts = build_prompts(prompt_cfg, variant, dataset.classes)
                run_one(
                    model_key, model_cfg, detector, dataset, prompts, variant,
                    out_path, cfg.get("warmup", 3), args.limit,
                )
        except Exception:
            traceback.print_exc()
            print(f"[fail] {model_key} — переход к следующей модели", flush=True)
            failed.append(model_key)
        finally:
            del detector
            _free_gpu()

    if failed:
        sys.exit(f"Модели с ошибками: {', '.join(failed)}")


def _free_gpu() -> None:
    gc.collect()
    try:
        import torch

        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    except ImportError:
        pass


if __name__ == "__main__":
    main()
