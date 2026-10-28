"""
Загрузка BDD100K (val) в том же интерфейсе, что и ACDC.

Используется открытая копия val-части на HuggingFace (dgural/bdd100k, формат
FiftyOne): 10 000 кадров 1280×720 с ручными боксами и метками погоды и
времени суток. Скачивается без ручного одобрения, в отличие от ACDC.

Домены собираются по меткам кадра. Ночь берётся только при ясной погоде, а
дождь и снег — только днём, чтобы эффекты не смешивались. Главное отличие от
ACDC: кадры разных доменов сняты в разных местах, поэтому сдвиг условий
смешан со сменой местности — это фиксируется в ограничениях.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path

import numpy as np

from .acdc import ACDC_CLASSES, ACDCDetection, Sample

# Классы BDD100K -> классы эксперимента. traffic light/sign, other vehicle,
# trailer, other person в эксперимент не входят: в словаре запросов их нет.
BDD_TO_EXP = {
    "pedestrian": "person",
    "rider": "rider",
    "car": "car",
    "truck": "truck",
    "bus": "bus",
    "train": "train",
    "motorcycle": "motorcycle",
    "bicycle": "bicycle",
}


@lru_cache(maxsize=2)
def _read_samples(path: str) -> list[dict]:
    with open(path, encoding="utf-8") as f:
        return json.load(f)["samples"]


def _label(sample: dict, field: str) -> str | None:
    return (sample.get(field) or {}).get("label")


class BDDDetection(ACDCDetection):
    """Домен BDD100K val, отобранный по погоде и времени суток."""

    def __init__(
        self,
        root: str | Path,
        domain: str,
        rule: dict,
        max_images: int | None = None,
        seed: int = 0,
        min_box_area: float = 16.0,
        classes: list[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.domain = domain
        self.rule = rule
        self.max_images = max_images
        self.seed = seed
        self.splits = ["val"]
        self.split = "val"
        self.subset = "all"
        self.min_box_area = min_box_area
        self.classes = classes or ACDC_CLASSES
        self.samples = self._load()

    def _load(self) -> list[Sample]:
        path = self.root / "samples.json"
        if not path.exists():
            raise FileNotFoundError(
                f"Нет {path}. Скачайте данные: bash scripts/download_bdd.sh"
            )
        raw = [
            s for s in _read_samples(str(path))
            if _label(s, "weather") in self.rule["weather"]
            and _label(s, "timeofday") in self.rule["timeofday"]
        ]
        # Подвыборка зависит только от списка кадров и seed — одинакова для
        # всех моделей и повторных запусков
        raw.sort(key=lambda s: s["filepath"])
        if self.max_images and len(raw) > self.max_images:
            idx = np.random.default_rng(self.seed).choice(len(raw), self.max_images, replace=False)
            raw = [raw[i] for i in sorted(idx)]
        return [self._to_sample(s) for s in raw]

    def _to_sample(self, s: dict) -> Sample:
        w, h = s["metadata"]["width"], s["metadata"]["height"]
        boxes, labels = [], []
        for d in (s.get("detections") or {}).get("detections") or []:
            name = BDD_TO_EXP.get(d["label"])
            if name not in self.classes:
                continue
            x, y, bw, bh = d["bounding_box"]          # доли кадра, [x, y, w, h]
            if bw * w * bh * h < self.min_box_area:
                continue
            boxes.append([x * w, y * h, (x + bw) * w, (y + bh) * h])
            labels.append(self.classes.index(name))
        stem = Path(s["filepath"]).stem
        return Sample(
            image_id=stem,
            image_path=self.root / s["filepath"],
            domain=self.domain,
            boxes=np.array(boxes, dtype=np.float32).reshape(-1, 4),
            labels=np.array(labels, dtype=np.int32),
            width=w,
            height=h,
            split="val",
            sequence=stem,        # в val каждый кадр — из своей видеозаписи
        )
