"""
Единый интерфейс детектора для сопоставимого сравнения разнородных моделей.

Ключевая идея протокола: все модели — open-vocabulary детекторы, VLM с grounding
и closed-set baseline — приводятся к одному контракту вход/выход. Это обязательное
условие корректного сравнения: любое расхождение в постобработке (NMS, порог,
формат координат) смещает метрики сильнее, чем разница между самими моделями.

Контракт:
    вход  : RGB-изображение (PIL.Image) + список текстовых запросов классов
    выход : Detections — боксы в абсолютных пикселях xyxy, confidence, id класса
            + поля диагностики (невалидные выходы, время, память)
"""

from __future__ import annotations

import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from PIL import Image


# --------------------------------------------------------------------------- #
#                              Структуры данных                               #
# --------------------------------------------------------------------------- #

@dataclass
class Detections:
    """Результат детекции на одном изображении.

    boxes   : (N, 4) float32, абсолютные пиксели, формат xyxy, x1<x2, y1<y2
    scores  : (N,)   float32 в [0, 1]
    labels  : (N,)   int32 — индекс в списке классов эксперимента
    raw     : сырой выход модели (для отладки и подсчёта невалидных ответов)
    """

    boxes: np.ndarray
    scores: np.ndarray
    labels: np.ndarray
    raw: Any = None

    def __post_init__(self) -> None:
        self.boxes = np.asarray(self.boxes, dtype=np.float32).reshape(-1, 4)
        self.scores = np.asarray(self.scores, dtype=np.float32).reshape(-1)
        self.labels = np.asarray(self.labels, dtype=np.int32).reshape(-1)
        n = len(self.boxes)
        assert len(self.scores) == n and len(self.labels) == n, (
            f"Рассогласование длин: boxes={n}, scores={len(self.scores)}, "
            f"labels={len(self.labels)}"
        )

    def __len__(self) -> int:
        return len(self.boxes)

    def filter_by_score(self, thr: float) -> "Detections":
        keep = self.scores >= thr
        return Detections(self.boxes[keep], self.scores[keep], self.labels[keep])


@dataclass
class ValidityReport:
    """Диагностика «невалидных ответов» — аналог доли невалидного JSON в LLM-задачах.

    Для генеративных VLM невалидность буквальна: модель вернула текст, который не
    парсится в боксы. Для детекторов она структурная: вырожденный бокс, выход за
    границы кадра, класс вне словаря. И то, и другое делает псевдоразметку
    непригодной без ручной правки, поэтому считается по единой шкале.
    """

    n_raw: int = 0                  # сколько объектов модель заявила
    n_unparseable: int = 0          # текст не распарсился (только VLM)
    n_degenerate: int = 0           # w<=0 или h<=0
    n_out_of_bounds: int = 0        # бокс целиком вне кадра
    n_clipped: int = 0              # бокс частично вне кадра, обрезан
    n_unknown_class: int = 0        # класс вне словаря эксперимента
    n_kept: int = 0                 # дошло до метрик

    def merge(self, other: "ValidityReport") -> "ValidityReport":
        return ValidityReport(
            **{
                f: getattr(self, f) + getattr(other, f)
                for f in self.__dataclass_fields__
            }
        )

    @property
    def invalid_rate(self) -> float:
        """Доля невалидных выходов среди заявленных моделью объектов."""
        if self.n_raw == 0:
            return 0.0
        bad = (
            self.n_unparseable
            + self.n_degenerate
            + self.n_out_of_bounds
            + self.n_unknown_class
        )
        return bad / self.n_raw


@dataclass
class InferenceRecord:
    """Одна строка журнала инференса — основа для таблиц метрик и профиля ресурсов."""

    image_id: str
    domain: str
    prompt_variant: str
    detections: Detections
    validity: ValidityReport
    latency_ms: float
    peak_vram_mb: float = 0.0
    meta: dict = field(default_factory=dict)


# --------------------------------------------------------------------------- #
#                              Базовый детектор                               #
# --------------------------------------------------------------------------- #

class BaseDetector(ABC):
    """Контракт, которому обязана следовать каждая модель эксперимента.

    Подкласс реализует только `_predict`. Санитизация, замер времени и памяти
    и подсчёт невалидных выходов происходят в базовом классе — одинаково для
    всех моделей. Это гарантирует, что различия в таблице метрик вызваны
    моделями, а не разной обвязкой.
    """

    #: человекочитаемое имя для таблиц
    name: str = "base"
    #: лицензия весов — учитывается при выборе модели для продакшена
    license: str = "unknown"
    #: открытый словарь или фиксированный набор классов
    open_vocabulary: bool = True
    #: что использовано при адаптации (примеры few-shot, веса LoRA) — пишется в журнал
    adaptation_meta: dict | None = None

    def __init__(
        self,
        class_names: list[str],
        device: str = "cuda",
        box_threshold: float = 0.05,
        max_detections: int | None = None,
        **kwargs: Any,
    ) -> None:
        self.class_names = class_names
        self.device = device
        # Предел боксов на кадр по скору, как maxDets в COCO. Нужен режимам,
        # где скоры смещены вверх и выше box_threshold остаются тысячи боксов
        # (few-shot OWL): хвост ниже первых сотен на mAP не влияет, а журналы
        # без предела вырастают в десятки раз. У zero-shot моделей не задан.
        self.max_detections = max_detections
        # Порог намеренно низкий: отсечение по confidence — предмет исследования,
        # поэтому сырые предсказания сохраняются, а пороги подбираются на этапе
        # анализа. Высокий порог на инференсе уничтожил бы кривые precision-recall
        # и сделал бы калибровку неизмеримой.
        self.box_threshold = box_threshold
        self._load(**kwargs)

    # ---------------------------- для подкласса ---------------------------- #

    @abstractmethod
    def _load(self, **kwargs: Any) -> None:
        """Загрузить веса и перевести модель в eval-режим."""

    @abstractmethod
    def _predict(self, image: Image.Image, prompts: list[str]) -> tuple[np.ndarray, np.ndarray, np.ndarray, Any]:
        """Сырое предсказание: (boxes_xyxy_abs, scores, labels, raw).

        Реализация обязана вернуть координаты в абсолютных пикселях исходного
        изображения. Любое внутреннее масштабирование разворачивается здесь же.
        """

    # ------------------------------ публичное ------------------------------ #

    def predict(
        self,
        image: Image.Image,
        prompts: list[str] | None = None,
        measure_vram: bool = True,
    ) -> tuple[Detections, ValidityReport, float, float]:
        """Предсказание с замером ресурсов и санитизацией выхода."""
        prompts = prompts if prompts is not None else self.class_names

        peak_vram = 0.0
        if measure_vram:
            try:
                import torch

                if torch.cuda.is_available():
                    torch.cuda.synchronize()
                    torch.cuda.reset_peak_memory_stats()
            except ImportError:
                measure_vram = False

        t0 = time.perf_counter()
        boxes, scores, labels, raw = self._predict(image, prompts)
        try:
            import torch

            if torch.cuda.is_available():
                torch.cuda.synchronize()
                if measure_vram:
                    peak_vram = torch.cuda.max_memory_allocated() / 1024**2
        except ImportError:
            pass
        latency_ms = (time.perf_counter() - t0) * 1000.0

        dets, validity = self._sanitize(boxes, scores, labels, image.size, raw)
        if self.max_detections and len(dets) > self.max_detections:
            top = np.argsort(-dets.scores, kind="stable")[: self.max_detections]
            dets = Detections(dets.boxes[top], dets.scores[top], dets.labels[top], dets.raw)
            validity.n_kept = len(dets)
        return dets, validity, latency_ms, peak_vram

    # ------------------------------ внутреннее ----------------------------- #

    def _sanitize(
        self,
        boxes: np.ndarray,
        scores: np.ndarray,
        labels: np.ndarray,
        image_size: tuple[int, int],
        raw: Any,
        n_unparseable: int = 0,
    ) -> tuple[Detections, ValidityReport]:
        """Привести выход к валидному виду и зафиксировать, что пришлось чинить.

        Порядок проверок важен: сначала отбрасываются принципиально непригодные
        боксы (вырожденные, вне кадра, неизвестный класс), затем остальные
        обрезаются по границам. Бокс, обрезанный до нулевой площади, считается
        вырожденным, а не обрезанным.
        """
        W, H = image_size
        rep = ValidityReport(n_raw=len(boxes) + n_unparseable, n_unparseable=n_unparseable)

        if len(boxes) == 0:
            return Detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0), raw), rep

        boxes = np.asarray(boxes, dtype=np.float32).reshape(-1, 4)
        scores = np.asarray(scores, dtype=np.float32).reshape(-1)
        labels = np.asarray(labels, dtype=np.int32).reshape(-1)

        # 1. класс вне словаря
        known = (labels >= 0) & (labels < len(self.class_names))
        rep.n_unknown_class = int((~known).sum())

        # 2. бокс целиком вне кадра
        inside = (boxes[:, 2] > 0) & (boxes[:, 0] < W) & (boxes[:, 3] > 0) & (boxes[:, 1] < H)
        rep.n_out_of_bounds = int((~inside & known).sum())

        keep = known & inside
        boxes, scores, labels = boxes[keep], scores[keep], labels[keep]
        if len(boxes) == 0:
            return Detections(np.zeros((0, 4)), np.zeros(0), np.zeros(0), raw), rep

        # 3. обрезка по границам кадра
        clipped_before = boxes.copy()
        boxes[:, [0, 2]] = boxes[:, [0, 2]].clip(0, W)
        boxes[:, [1, 3]] = boxes[:, [1, 3]].clip(0, H)
        rep.n_clipped = int((np.abs(boxes - clipped_before) > 1e-3).any(axis=1).sum())

        # 4. вырожденные после обрезки
        valid_wh = (boxes[:, 2] - boxes[:, 0] > 1.0) & (boxes[:, 3] - boxes[:, 1] > 1.0)
        rep.n_degenerate = int((~valid_wh).sum())
        boxes, scores, labels = boxes[valid_wh], scores[valid_wh], labels[valid_wh]

        # 5. confidence вне [0,1] — встречается у VLM, выводящих «уверенность» текстом
        scores = np.clip(scores, 0.0, 1.0)

        rep.n_kept = len(boxes)
        return Detections(boxes, scores, labels, raw), rep

    # ----------------------------- утилиты --------------------------------- #

    @staticmethod
    def warmup(fn, image: Image.Image, prompts: list[str], n: int = 3) -> None:
        """Прогрев перед замером latency.

        Без прогрева первый вызов включает компиляцию ядер CUDA и ленивую
        загрузку весов — это завышает p50 на порядок и делает сравнение
        моделей по скорости бессмысленным.
        """
        for _ in range(n):
            fn(image, prompts)
