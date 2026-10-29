"""
Адаптеры конкретных моделей к единому интерфейсу BaseDetector.

Каждый адаптер решает три задачи:
    1. привести текстовые запросы к формату, которого ждёт модель;
    2. развернуть внутреннее масштабирование обратно в пиксели исходного кадра;
    3. сопоставить выход модели с индексами классов эксперимента.

Третий пункт — главный источник скрытых ошибок при сравнении open-vocabulary
моделей. Grounding DINO возвращает сходство с токенами промпта, OWLv2 — индекс
запроса, YOLO-World — индекс класса, VLM — строку. Несогласованный маппинг
даёт расхождение mAP в единицы процентов, которое легко принять за разницу
между моделями.
"""

from __future__ import annotations

from typing import Any

import numpy as np
from PIL import Image

from .base import BaseDetector


def merge_lora(model, path: str):
    """Подгрузить обученные добавки LoRA и влить их в веса модели.

    После слияния модель устроена так же, как исходная, поэтому задержка и
    память сравнимы с zero-shot без поправки на обвязку PEFT.
    """
    from peft import PeftModel

    return PeftModel.from_pretrained(model, path).merge_and_unload()


# --------------------------------------------------------------------------- #
#                              Grounding DINO                                 #
# --------------------------------------------------------------------------- #

def build_grounding_text(prompts: list[str]) -> tuple[str, list[tuple[int, int]]]:
    """Склеить запросы в строку 'person. rider. car.' и запомнить, где какая фраза.

    Нижний регистр и точка после каждой фразы — требование авторов Grounding
    DINO, иначе падает recall. Символьные границы фраз нужны, чтобы потом
    отнести токены текста к классам.
    """
    text, spans = "", []
    for p in prompts:
        phrase = p.lower().strip(" .")
        if text:
            text += ". "
        spans.append((len(text), len(text) + len(phrase)))
        text += phrase
    return text + ".", spans


def phrase_token_indices(
    offsets: list[tuple[int, int]], char_spans: list[tuple[int, int]]
) -> list[np.ndarray]:
    """Индексы токенов, попадающих в каждую фразу. Служебные токены ([CLS],
    [SEP]) имеют пустой offset (0, 0) и не попадают ни в одну фразу."""
    out = []
    for s, e in char_spans:
        idx = [i for i, (a, b) in enumerate(offsets) if b > a and a >= s and b <= e]
        out.append(np.array(idx, dtype=np.int64))
    return out


def phrase_scores(token_probs: np.ndarray, token_idx: list[np.ndarray]) -> np.ndarray:
    """(Q, T) вероятности по токенам -> (Q, C) скор каждого класса как
    максимум по токенам его фразы."""
    q = token_probs.shape[0]
    cols = [
        token_probs[:, idx].max(axis=1) if len(idx) else np.zeros(q, dtype=token_probs.dtype)
        for idx in token_idx
    ]
    return np.stack(cols, axis=1)


class GroundingDINOAdapter(BaseDetector):
    """IDEA-Research Grounding DINO через HuggingFace transformers.

    Особенность: промпт подаётся одной строкой с разделителем '. ', а модель
    выдаёт для каждого бокса сходство с каждым токеном этой строки.

    Класс бокса определяется по токенам, а не по тексту: скор класса — максимум
    по токенам его фразы, метка — класс с наибольшим скором. Так устроена
    оценка на COCO у авторов (positive map). Штатный постпроцессор transformers
    возвращает склеенную строку из активных токенов, и сопоставлять её с
    классами пришлось бы по подстроке. Это ломается на вариантах промпта с
    общими словами: у 'a photo of a car on a road' и 'a photo of a bus on a
    road' общая почти вся фраза, и обрывок 'a photo' приписывался первому
    классу — ложные срабатывания person вместо честного n_unknown_class.
    """

    name = "grounding-dino"
    license = "Apache-2.0"
    open_vocabulary = True

    def _load(
        self,
        checkpoint: str = "IDEA-Research/grounding-dino-base",
        lora: str | None = None,
        **kw: Any,
    ) -> None:
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.checkpoint = checkpoint
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        model = AutoModelForZeroShotObjectDetection.from_pretrained(checkpoint)
        if lora:
            model = merge_lora(model, lora)
            self.adaptation_meta = {"lora": lora}
        self.model = model.to(self.device).eval()
        self._torch = torch

    def _predict(self, image: Image.Image, prompts: list[str]):
        torch = self._torch
        text, char_spans = build_grounding_text(prompts)

        inputs = self.processor(images=image, text=text, return_tensors="pt").to(self.device)
        offsets = self.processor.tokenizer(text, return_offsets_mapping=True)["offset_mapping"]
        assert len(offsets) == inputs.input_ids.shape[1], "токенизация разошлась с процессором"
        token_idx = phrase_token_indices(offsets, char_spans)

        with torch.no_grad():
            outputs = self.model(**inputs)

        # logits: (Q, 256), позиции за концом текста заполнены -inf
        probs = outputs.logits[0].sigmoid().float().cpu().numpy()[:, : len(offsets)]
        cls_scores = phrase_scores(probs, token_idx)
        labels = cls_scores.argmax(axis=1).astype(np.int32)
        scores = cls_scores.max(axis=1)

        # pred_boxes: (Q, 4) cxcywh в долях исходного кадра
        cx, cy, w, h = outputs.pred_boxes[0].float().cpu().numpy().T
        scale = np.array([image.width, image.height, image.width, image.height], dtype=np.float32)
        boxes = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], axis=1) * scale

        keep = scores >= self.box_threshold
        return boxes[keep], scores[keep], labels[keep], None


# --------------------------------------------------------------------------- #
#                                   OWLv2                                     #
# --------------------------------------------------------------------------- #

class OWLv2Adapter(BaseDetector):
    """Google OWLv2 (Scaling Open-Vocabulary Object Detection).

    В отличие от Grounding DINO принимает список отдельных запросов и
    возвращает индекс запроса напрямую — маппинг тривиален и безошибочен.
    Это делает OWLv2 удобным эталоном при отладке маппинга других моделей.
    """

    name = "owlv2"
    license = "Apache-2.0"
    open_vocabulary = True

    def _load(
        self,
        checkpoint: str = "google/owlv2-base-patch16-ensemble",
        shots: dict[int, list[dict]] | None = None,
        shot_mix: float = 0.5,
        **kw: Any,
    ) -> None:
        import torch
        from transformers import Owlv2ForObjectDetection, Owlv2Processor

        self.checkpoint = checkpoint
        self.processor = Owlv2Processor.from_pretrained(checkpoint)
        self.model = Owlv2ForObjectDetection.from_pretrained(checkpoint).to(self.device).eval()
        self._torch = torch
        self._init_shots(shots, shot_mix)

    def _predict(self, image: Image.Image, prompts: list[str]):
        if self.shots:
            return self._predict_with_queries(image, self._query_embeds(prompts))

        torch = self._torch
        inputs = self.processor(text=[prompts], images=image, return_tensors="pt").to(self.device)
        with torch.no_grad():
            outputs = self.model(**inputs)
        return self._postprocess(outputs, image)

    def _postprocess(self, outputs, image: Image.Image):
        # OWLv2 дополняет кадр до квадрата, и боксы нормированы на длинную
        # сторону. Начиная с transformers 4.49 постпроцессор учитывает это сам,
        # поэтому передаётся исходный (h, w). В более ранних версиях здесь
        # нужен был (max(h, w), max(h, w)), иначе боксы сжимались по вертикали.
        torch = self._torch
        target_sizes = torch.tensor([(image.height, image.width)], device=self.device)
        results = self.processor.post_process_grounded_object_detection(
            outputs=outputs, target_sizes=target_sizes, threshold=self.box_threshold
        )[0]

        boxes = results["boxes"].cpu().numpy()
        scores = results["scores"].cpu().numpy()
        labels = results["labels"].cpu().numpy().astype(np.int32)
        return boxes, scores, labels, None

    # ------------------------------ few-shot ------------------------------- #
    #
    # Запрос класса у OWL — вектор, с которым сравнивается каждый бокс кадра.
    # Обычно это вектор текста; вместо него можно взять вектор, который модель
    # строит для бокса на кадре-образце (image-conditioned detection из статьи
    # OWL-ViT). Штатный image_guided_detection берёт один образец без указания,
    # где объект, поэтому образцы здесь кодируются вручную: из боксов модели на
    # кадре-образце выбирается совпадающий с размеченным, а векторы образцов
    # одного класса усредняются. Остальная обработка — штатная, через тот же
    # постпроцессор, что и в zero-shot.

    def _init_shots(self, shots: dict[int, list[dict]] | None, shot_mix: float) -> None:
        self.shots = {c: v for c, v in (shots or {}).items() if v}
        self.shot_mix = shot_mix
        self._query_cache: tuple[tuple[str, ...], Any] | None = None
        if self.shots:
            self.adaptation_meta = {
                "shot_mix": shot_mix,
                "shots": {self.class_names[c]: [s["image_id"] for s in v]
                          for c, v in self.shots.items()},
            }

    def _image_features(self, image: Image.Image):
        inputs = self.processor(images=image, return_tensors="pt").to(self.device)
        feature_map = self.model.image_embedder(pixel_values=inputs["pixel_values"])[0]
        b, h, w, d = feature_map.shape
        return feature_map, feature_map.reshape(b, h * w, d)

    def _box_scale(self, image: Image.Image) -> np.ndarray:
        """Множители «доли → пиксели» постпроцессора для кадра этого размера.

        Определяются через сам постпроцессор, чтобы размеченный бокс переводился
        в координаты модели ровно обратным ему преобразованием: у OWLv2 кадр
        дополняется до квадрата, у OWL-ViT растягивается, и это разное в разных
        версиях transformers.
        """
        from types import SimpleNamespace

        torch = self._torch
        probe = SimpleNamespace(
            logits=torch.full((1, 1, 1), 10.0),
            pred_boxes=torch.tensor([[[0.5, 0.5, 1.0, 1.0]]]),
        )
        res = self.processor.post_process_grounded_object_detection(
            outputs=probe, target_sizes=torch.tensor([(image.height, image.width)]), threshold=0.0
        )[0]
        x1, y1, x2, y2 = res["boxes"][0].tolist()
        return np.array([x2 - x1, y2 - y1, x2 - x1, y2 - y1], dtype=np.float32)

    def _shot_embedding(self, shot: dict):
        """Вектор образца: вектор бокса модели, лучше всего совпадающего с размеченным."""
        from src.eval.detection_metrics import box_iou

        torch = self._torch
        image = shot["image"]
        feature_map, feats = self._image_features(image)
        _, class_embeds = self.model.class_predictor(feats)
        pred = self.model.box_predictor(feats, feature_map)[0].float().cpu().numpy()
        pred_xyxy = np.concatenate([pred[:, :2] - pred[:, 2:] / 2, pred[:, :2] + pred[:, 2:] / 2], 1)
        gt = np.asarray(shot["box"], dtype=np.float32)[None] / self._box_scale(image)
        iou = box_iou(gt, pred_xyxy)[0]
        best = int(iou.argmax())
        emb = class_embeds[0, best]
        return emb / (emb.norm() + 1e-6), float(iou[best])

    def _query_embeds(self, prompts: list[str]):
        """(C, D) векторы запросов: текст, образцы или их смесь — один раз на словарь."""
        torch = self._torch
        key = tuple(prompts)
        if self._query_cache and self._query_cache[0] == key:
            return self._query_cache[1]

        with torch.no_grad():
            # Векторы текста — из штатного прохода, кадр на них не влияет
            blank = Image.new("RGB", (64, 64))
            inputs = self.processor(text=[prompts], images=blank, return_tensors="pt").to(self.device)
            text = self.model(**inputs).text_embeds[0]
            text = text / (text.norm(dim=-1, keepdim=True) + 1e-6)

            queries, ious = [], {}
            for c in range(len(prompts)):
                if c not in self.shots or self.shot_mix == 0:
                    queries.append(text[c])
                    continue
                embs = [self._shot_embedding(s) for s in self.shots[c]]
                ious[self.class_names[c]] = [round(i, 3) for _, i in embs]
                img = torch.stack([e for e, _ in embs]).mean(0)
                img = img / (img.norm() + 1e-6)
                q = (1 - self.shot_mix) * text[c] + self.shot_mix * img
                queries.append(q / (q.norm() + 1e-6))
        if self.adaptation_meta is not None:
            self.adaptation_meta["shot_iou"] = ious
        out = torch.stack(queries)
        self._query_cache = (key, out)
        return out

    def _predict_with_queries(self, image: Image.Image, queries):
        from types import SimpleNamespace

        torch = self._torch
        with torch.no_grad():
            feature_map, feats = self._image_features(image)
            mask = torch.ones(1, len(queries), dtype=torch.bool, device=self.device)
            logits, _ = self.model.class_predictor(feats, queries[None], mask)
            pred_boxes = self.model.box_predictor(feats, feature_map)
        return self._postprocess(SimpleNamespace(logits=logits, pred_boxes=pred_boxes), image)


class OWLViTAdapter(OWLv2Adapter):
    """OWL-ViT v1 — предыдущее поколение, нижняя точка семейства.

    Интерфейс постобработки совпадает с OWLv2, но архитектура и препроцессинг
    другие: у v1 нет головы objectness и кадр не дополняется до квадрата.
    Загрузка чекпоинта v1 классами OWLv2 проходит с предупреждением, часть
    весов инициализируется случайно, и метрики получаются бессмысленными.
    """

    name = "owlvit"

    def _load(
        self,
        checkpoint: str = "google/owlvit-base-patch32",
        shots: dict[int, list[dict]] | None = None,
        shot_mix: float = 0.5,
        **kw: Any,
    ) -> None:
        import torch
        from transformers import OwlViTForObjectDetection, OwlViTProcessor

        self.checkpoint = checkpoint
        self.processor = OwlViTProcessor.from_pretrained(checkpoint)
        self.model = OwlViTForObjectDetection.from_pretrained(checkpoint).to(self.device).eval()
        self._torch = torch
        self._init_shots(shots, shot_mix)


# --------------------------------------------------------------------------- #
#                                YOLO-World                                   #
# --------------------------------------------------------------------------- #

class YOLOWorldAdapter(BaseDetector):
    """Tencent YOLO-World через ultralytics.

    Внимание на лицензию: ultralytics распространяется под AGPL-3.0, что
    накладывает обязательства при встраивании в закрытый продукт. Для ВКР
    это несущественно, для продакшен-вывода — определяющий фактор, поэтому
    лицензия вынесена в таблицу сравнения наравне с метриками.
    """

    name = "yolo-world"
    license = "AGPL-3.0 (ultralytics)"
    open_vocabulary = True

    def _load(self, checkpoint: str = "yolov8l-worldv2.pt", **kw: Any) -> None:
        from ultralytics import YOLO

        self.checkpoint = checkpoint
        self.model = YOLO(checkpoint)
        self._prompt_cache: list[str] | None = None

    def _predict(self, image: Image.Image, prompts: list[str]):
        # set_classes пересчитывает текстовые эмбеддинги — дорогая операция,
        # поэтому вызывается только при смене словаря. Иначе она попала бы
        # в замер latency и исказила сравнение со статическими детекторами.
        if self._prompt_cache != prompts:
            # Закэшированный CLIP помнит устройство первого вызова (CPU), а predict
            # уже перенёс его веса на GPU — повторный set_classes падает на смешении
            # устройств. Сброс заставляет собрать CLIP на текущем устройстве.
            if getattr(self.model.model, "clip_model", None) is not None:
                self.model.model.clip_model = None
            self.model.set_classes(prompts)
            self._prompt_cache = list(prompts)

        res = self.model.predict(
            image, conf=self.box_threshold, verbose=False, device=self.device
        )[0]

        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int32), None

        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        labels = res.boxes.cls.cpu().numpy().astype(np.int32)
        return boxes, scores, labels, None


class YOLOEAdapter(YOLOWorldAdapter):
    """YOLOE (ultralytics, 2025) — следующее поколение после YOLO-World.

    Текстовые запросы кодируются MobileCLIP; маски сегментации не используются,
    берутся только боксы.
    """

    name = "yoloe"

    def _load(self, checkpoint: str = "yoloe-26l-seg.pt", **kw: Any) -> None:
        from ultralytics import YOLOE

        self.checkpoint = checkpoint
        self.model = YOLOE(checkpoint)
        self._prompt_cache: list[str] | None = None


# --------------------------------------------------------------------------- #
#                         VLM с grounding (Qwen2.5-VL)                        #
# --------------------------------------------------------------------------- #

def grid_scale(processor, grid_row, image: Image.Image) -> tuple[float, float]:
    """Множители «координаты Qwen → пиксели кадра» по строке image_grid_thw.

    Сетка задана в патчах до слияния, поэтому размер входа — число патчей,
    умноженное на размер патча.
    """
    patch = getattr(processor.image_processor, "patch_size", 14)
    _, gh, gw = (int(v) for v in grid_row)
    return image.width / (gw * patch), image.height / (gh * patch)


def qwen_input_scale(processor, image: Image.Image) -> tuple[float, float]:
    """Те же множители для отдельного кадра — примера few-shot или обучающего."""
    grid = processor.image_processor(images=[image], return_tensors="pt")["image_grid_thw"]
    return grid_scale(processor, grid[0], image)


def qwen_answer(boxes: np.ndarray, labels: np.ndarray, prompts: list[str],
                scale: tuple[float, float]) -> str:
    """Правильный ответ в формате инструкции — для примеров few-shot и обучения LoRA.

    Координаты переводятся в сетку входа Qwen, как в его собственных ответах.
    Объекты перечисляются сверху вниз и слева направо: устойчивый порядок
    легче воспроизвести, чем случайный. Уверенность размеченного объекта — 1.0:
    другого честного значения у разметки нет.
    """
    import json

    sx, sy = scale
    order = sorted(range(len(boxes)), key=lambda i: (round(float(boxes[i][1])), float(boxes[i][0])))
    items = []
    for i in order:
        x1, y1, x2, y2 = boxes[i]
        items.append({
            "bbox_2d": [round(x1 / sx), round(y1 / sy), round(x2 / sx), round(y2 / sy)],
            "label": prompts[int(labels[i])],
            "confidence": 1.0,
        })
    return json.dumps(items, ensure_ascii=False)


class QwenVLAdapter(BaseDetector):
    """Qwen2.5-VL как детектор через структурированный текстовый ответ.

    Единственная модель в наборе, у которой «невалидный ответ» буквален:
    она генерирует текст, который может не распарситься в JSON. Поэтому
    именно на ней измеряется та же величина, что в LLM-задачах — доля
    невалидных структурированных ответов.

    Координаты Qwen2.5-VL выдаёт в системе координат ресайзнутого входа,
    поэтому требуется обратное масштабирование по smart_resize. Пропуск
    этого шага — типичная ошибка, дающая систематический сдвиг боксов.
    """

    name = "qwen2.5-vl-7b"
    license = "Apache-2.0"
    open_vocabulary = True
    MODEL_CLASS = "Qwen2_5_VLForConditionalGeneration"

    PROMPT_TEMPLATE = (
        "Detect all objects of the following categories in this image: {classes}.\n"
        "Output a JSON array. Each element must have exactly these keys:\n"
        '  "bbox_2d": [x1, y1, x2, y2] (integers, pixel coordinates),\n'
        '  "label": one of the listed categories (exact string),\n'
        '  "confidence": a float between 0.0 and 1.0.\n'
        "Output only the JSON array, no explanation, no markdown fences.\n"
        "If no objects are present, output []."
    )

    def _load(
        self,
        checkpoint: str = "Qwen/Qwen2.5-VL-7B-Instruct",
        max_new_tokens: int = 2048,
        shots: list[dict] | None = None,
        lora: str | None = None,
        **kw: Any,
    ) -> None:
        import torch
        import transformers
        from transformers import AutoProcessor

        self.checkpoint = checkpoint
        self.max_new_tokens = max_new_tokens
        self.processor = AutoProcessor.from_pretrained(checkpoint)
        model = getattr(transformers, self.MODEL_CLASS).from_pretrained(
            checkpoint, torch_dtype=torch.bfloat16, device_map=self.device
        )
        if lora:
            model = merge_lora(model, lora)
            self.adaptation_meta = {"lora": lora}
        self.model = model.eval()
        self._torch = torch
        self.shots = shots or []
        if self.shots:
            self.adaptation_meta = {"shots": [s["image_id"] for s in self.shots]}

    @classmethod
    def user_turn(cls, image: Image.Image, prompts: list[str]) -> dict:
        """Сообщение пользователя: кадр и инструкция. Одно и то же в zero-shot,
        в примерах few-shot и при обучении LoRA."""
        return {
            "role": "user",
            "content": [
                {"type": "image", "image": image},
                {"type": "text", "text": cls.PROMPT_TEMPLATE.format(classes=", ".join(prompts))},
            ],
        }

    def _predict(self, image: Image.Image, prompts: list[str]):
        torch = self._torch
        # Few-shot: перед кадром — диалог из примеров «кадр → правильный ответ»
        messages, images = [], []
        for s in self.shots:
            messages += [
                self.user_turn(s["image"], prompts),
                {"role": "assistant", "content": [{"type": "text", "text": qwen_answer(
                    s["boxes"], s["labels"], prompts, qwen_input_scale(self.processor, s["image"]))}]},
            ]
            images.append(s["image"])
        messages.append(self.user_turn(image, prompts))
        images.append(image)
        chat = self.processor.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )
        inputs = self.processor(text=[chat], images=images, return_tensors="pt").to(self.device)

        with torch.no_grad():
            # temperature=0 через do_sample=False: воспроизводимость обязательна,
            # иначе повторный запуск даст другие числа и сравнение развалится.
            generated = self.model.generate(
                **inputs, max_new_tokens=self.max_new_tokens, do_sample=False
            )
        trimmed = generated[:, inputs.input_ids.shape[1]:]
        answer = self.processor.batch_decode(trimmed, skip_special_tokens=True)[0]

        sx, sy = self._answer_scale(inputs, image)
        boxes, scores, labels, n_bad = self._parse(answer, prompts, sx, sy)
        return boxes, scores, labels, {"answer": answer, "n_unparseable": n_bad}

    def _answer_scale(self, inputs, image: Image.Image) -> tuple[float, float]:
        # Масштаб входа: Qwen2.5-VL отдаёт координаты в сетке обработанного
        # изображения. Кадр, о котором вопрос, — последний среди изображений диалога.
        grid = inputs.get("image_grid_thw")
        if grid is None:
            return 1.0, 1.0
        return grid_scale(self.processor, grid[-1], image)

    def _parse(self, answer: str, prompts: list[str], sx: float, sy: float):
        """Разобрать текстовый ответ в боксы, считая неразобранные элементы."""
        import json
        import re

        norm = [p.lower().strip() for p in prompts]
        boxes, scores, labels = [], [], []
        n_bad = 0

        # Снять markdown-ограждение, если модель его добавила вопреки инструкции.
        cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", answer.strip(), flags=re.MULTILINE)
        match = re.search(r"\[.*\]", cleaned, re.DOTALL)
        if not match:
            # Ответ целиком невалиден: считаем это одним невалидным ответом,
            # поскольку заявленное число объектов неизвестно.
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int32), 1

        try:
            items = json.loads(match.group(0))
        except json.JSONDecodeError:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int32), 1

        if not isinstance(items, list):
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int32), 1

        for it in items:
            try:
                bb = it["bbox_2d"]
                if len(bb) != 4:
                    n_bad += 1
                    continue
                x1, y1, x2, y2 = (float(v) for v in bb)
                lbl = str(it.get("label", "")).lower().strip()
                idx = norm.index(lbl) if lbl in norm else -1
                conf = float(it.get("confidence", 1.0))
                boxes.append([x1 * sx, y1 * sy, x2 * sx, y2 * sy])
                scores.append(conf)
                labels.append(idx)
            except (KeyError, TypeError, ValueError):
                n_bad += 1

        if not boxes:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int32), n_bad
        return (
            np.array(boxes, dtype=np.float32),
            np.array(scores, dtype=np.float32),
            np.array(labels, dtype=np.int32),
            n_bad,
        )

    def predict(self, image, prompts=None, measure_vram: bool = True):
        """Переопределено, чтобы передать счётчик неразобранных элементов в отчёт."""
        dets, validity, lat, vram = super().predict(image, prompts, measure_vram)
        if dets.raw and "n_unparseable" in dets.raw:
            n_bad = dets.raw["n_unparseable"]
            validity.n_unparseable = n_bad
            validity.n_raw += n_bad
        return dets, validity, lat, vram


class Qwen3VLAdapter(QwenVLAdapter):
    """Qwen3-VL: та же обвязка, но координаты в ответе относительные, от 0 до 1000.

    Так модель обучена отвечать на запросы grounding, поэтому инструкция
    просит именно эту шкалу, а не пиксели.
    """

    name = "qwen3-vl-8b"
    MODEL_CLASS = "Qwen3VLForConditionalGeneration"
    PROMPT_TEMPLATE = QwenVLAdapter.PROMPT_TEMPLATE.replace(
        "integers, pixel coordinates", "integers, relative coordinates from 0 to 1000"
    )

    def _answer_scale(self, inputs, image: Image.Image) -> tuple[float, float]:
        return image.width / 1000, image.height / 1000


# --------------------------------------------------------------------------- #
#                      Closed-set baseline (контрольная точка)                #
# --------------------------------------------------------------------------- #

class ClosedSetAdapter(BaseDetector):
    """Обычный детектор с фиксированным словарём — контрольная точка сравнения.

    Нужен, чтобы отделить эффект открытого словаря от эффекта масштаба модели.
    Без него нельзя утверждать, что устойчивость к сдвигу домена связана именно
    с language-grounding, а не с размером обучающей выборки.

    Классы COCO сопоставляются классам эксперимента таблицей синонимов: COCO
    не различает rider и person, поэтому rider остаётся несопоставленным —
    это ограничение фиксируется в отчёте, а не маскируется.
    """

    name = "closed-set-baseline"
    license = "AGPL-3.0 (ultralytics)"
    open_vocabulary = False

    COCO_TO_EXP = {
        "person": "person",
        "car": "car",
        "bus": "bus",
        "truck": "truck",
        "bicycle": "bicycle",
        "motorcycle": "motorcycle",
        "train": "train",
    }

    def _load(self, checkpoint: str = "yolo11l.pt", **kw: Any) -> None:
        from ultralytics import YOLO

        self.checkpoint = checkpoint
        self.model = YOLO(checkpoint)
        names = self.model.names
        self._coco_idx_to_exp: dict[int, int] = {}
        for ci, cname in names.items():
            exp_name = self.COCO_TO_EXP.get(cname)
            if exp_name in self.class_names:
                self._coco_idx_to_exp[int(ci)] = self.class_names.index(exp_name)

    def _predict(self, image: Image.Image, prompts: list[str]):
        res = self.model.predict(
            image, conf=self.box_threshold, verbose=False, device=self.device
        )[0]
        if res.boxes is None or len(res.boxes) == 0:
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=np.int32), None

        boxes = res.boxes.xyxy.cpu().numpy()
        scores = res.boxes.conf.cpu().numpy()
        coco = res.boxes.cls.cpu().numpy().astype(int)
        labels = np.array([self._coco_idx_to_exp.get(c, -1) for c in coco], dtype=np.int32)

        keep = labels >= 0  # классы COCO вне словаря эксперимента не считаем ошибкой модели
        return boxes[keep], scores[keep], labels[keep], None


# --------------------------------------------------------------------------- #

REGISTRY: dict[str, type[BaseDetector]] = {
    "grounding-dino": GroundingDINOAdapter,
    "owlv2": OWLv2Adapter,
    "owlvit": OWLViTAdapter,
    "yolo-world": YOLOWorldAdapter,
    "yoloe": YOLOEAdapter,
    "qwen2.5-vl": QwenVLAdapter,
    "qwen3-vl": Qwen3VLAdapter,
    "closed-set": ClosedSetAdapter,
}


def build_detector(kind: str, class_names: list[str], **kwargs: Any) -> BaseDetector:
    if kind not in REGISTRY:
        raise KeyError(f"Неизвестная модель '{kind}'. Доступны: {sorted(REGISTRY)}")
    return REGISTRY[kind](class_names=class_names, **kwargs)
