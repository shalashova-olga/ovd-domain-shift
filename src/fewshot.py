"""
Данные для режимов адаптации: выбор примеров few-shot и путь к весам LoRA.

Примеры берутся из части train кадров адаптации (scripts/make_adapt_split.py)
в зафиксированном порядке, поэтому выбор детерминирован и вложен: примеры
1-shot входят в 3-shot и 5-shot. Сравнение режимов тогда отражает число
примеров, а не удачу при их выборе.

Два вида примеров:
    кадры целиком — для Qwen2.5-VL, которому в диалоге показывают размеченный
        кадр и правильный ответ;
    образцы объектов — для OWLv2 и OWL-ViT, которые ищут объекты, похожие на
        вырезанный образец, вместо текстового запроса или вместе с ним.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
from PIL import Image

from src.data.acdc import ACDCDetection, Sample, load_adaptation

# Кадр-пример для VLM должен показывать несколько объектов, но не раздувать
# ответ: при 4–12 объектах пример занимает ~300–700 токенов
IMAGE_SHOT_OBJECTS = (4, 12)

# Образец меньше 32×32 пикселей занимает на сетке признаков OWL одну-две
# клетки, и его вектор описывает скорее фон, чем объект
MIN_SHOT_AREA = 32 * 32


def image_shots(dataset: ACDCDetection, k: int,
                n_objects: tuple[int, int] = IMAGE_SHOT_OBJECTS) -> list[Sample]:
    """Первые k кадров (в порядке набора) с подходящим числом объектов."""
    lo, hi = n_objects
    picked = [s for s in dataset if lo <= len(s.boxes) <= hi][:k]
    if len(picked) < k:
        raise RuntimeError(f"Нашлось {len(picked)} кадров-примеров из {k} нужных")
    return picked


def box_shots(dataset: ACDCDetection, k: int, n_classes: int,
              min_area: float = MIN_SHOT_AREA) -> dict[int, list[tuple[Sample, np.ndarray]]]:
    """До k образцов каждого класса, каждый — с отдельного кадра.

    С кадра берётся самый крупный объект класса: крупный образец меньше
    зависит от соседних объектов и фона. Класс без подходящих образцов
    (в BDD100K так бывает с train) остаётся без них и ищется только по тексту.
    """
    out: dict[int, list[tuple[Sample, np.ndarray]]] = {c: [] for c in range(n_classes)}
    for s in dataset:
        if len(s.boxes) == 0:
            continue
        areas = (s.boxes[:, 2] - s.boxes[:, 0]) * (s.boxes[:, 3] - s.boxes[:, 1])
        for c in range(n_classes):
            if len(out[c]) >= k:
                continue
            idx = np.where((s.labels == c) & (areas >= min_area))[0]
            if len(idx):
                out[c].append((s, s.boxes[idx[np.argmax(areas[idx])]]))
        if all(len(v) >= k for v in out.values()):
            break
    return out


def adapter_kwargs(cfg: dict, model_cfg: dict) -> dict[str, Any]:
    """Дополнительные аргументы адаптера для режима модели из конфига.

    Для zero-shot — пусто. Изображения примеров загружаются здесь, чтобы
    адаптеры не зависели от устройства датасета.
    """
    mode = model_cfg.get("mode")
    if mode == "few-shot":
        train = load_adaptation(cfg, "train")
        k = int(model_cfg["shots"])
        if model_cfg["adapter"] == "qwen2.5-vl":
            shots = [
                {"image_id": s.image_id, "image": Image.open(s.image_path).convert("RGB"),
                 "boxes": s.boxes, "labels": s.labels}
                for s in image_shots(train, k)
            ]
            return {"shots": shots}
        if model_cfg["adapter"] in ("owlv2", "owlvit"):
            chosen = box_shots(train, k, len(train.classes))
            images: dict[str, Image.Image] = {}
            shots = {}
            for c, items in chosen.items():
                shots[c] = []
                for s, box in items:
                    if s.image_id not in images:
                        images[s.image_id] = Image.open(s.image_path).convert("RGB")
                    shots[c].append({"image_id": s.image_id, "image": images[s.image_id],
                                     "box": box})
            return {"shots": shots, "shot_mix": float(model_cfg.get("shot_mix", 0.5))}
        raise ValueError(f"У адаптера {model_cfg['adapter']} нет режима few-shot")
    if mode == "lora":
        path = Path(model_cfg["lora"])
        if not (path / "adapter_config.json").exists():
            raise FileNotFoundError(
                f"Нет весов LoRA в {path}. Сначала обучите:\n"
                f"    python -m src.train_lora --config <конфиг> --model <ключ модели>"
            )
        return {"lora": str(path)}
    return {}
