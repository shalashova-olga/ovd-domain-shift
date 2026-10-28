"""
Типизация ошибок детекции на уровне отдельного кадра.

Сводные метрики говорят, насколько модель ошибается, но не говорят как. Для
выбора учителя это решающее различие: пропуск из-за низкого confidence
лечится порогом или калибровкой, ложный бокс на пустом месте — нет. Поэтому
каждое ложное срабатывание и каждый пропуск получают причину.

Типы ложных срабатываний (по мотивам TIDE, Bolya et al., ECCV 2020):
    dup   — дубль: объект своего класса уже найден более уверенным боксом
    cls   — путаница классов: бокс точный, но класс другой (rider -> person)
    loc   — плохая локализация: свой класс, IoU от 0.1 до порога
    bkg   — фон: рядом нет ни одного объекта разметки (галлюцинация)
    other — чужой класс и плохая локализация одновременно

Причины пропусков:
    low_score   — модель нашла объект, но с confidence ниже рабочего порога;
                  это ошибка калибровки, а не зрения
    wrong_class — на месте объекта есть бокс выше порога, но другого класса
    missed      — модель не увидела объект ни при каком пороге
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from .detection_metrics import box_iou

FP_TYPES = ("dup", "cls", "loc", "bkg", "other")
FN_REASONS = ("low_score", "wrong_class", "missed")
LOC_IOU_MIN = 0.1


@dataclass
class ImageErrors:
    """Разбор одного кадра при фиксированном пороге.

    Индексы указывают на исходные массивы предсказаний и разметки кадра.
    """

    tp: list[int] = field(default_factory=list)
    fp: list[tuple[int, str]] = field(default_factory=list)       # (индекс, тип)
    fn: list[tuple[int, str]] = field(default_factory=list)       # (индекс разметки, причина)
    #: для fp типа cls: (класс предсказания, класс объекта)
    confusions: list[tuple[int, int]] = field(default_factory=list)

    @property
    def n_errors(self) -> int:
        return len(self.fp) + len(self.fn)


def classify_image(
    p_boxes: np.ndarray,
    p_scores: np.ndarray,
    p_labels: np.ndarray,
    g_boxes: np.ndarray,
    g_labels: np.ndarray,
    score_threshold: float,
    iou_threshold: float = 0.5,
) -> ImageErrors:
    """Разобрать кадр: TP, тип каждого ложного срабатывания, причина каждого пропуска.

    Сопоставление то же, что в match_detections: жадно по убыванию confidence,
    внутри класса, объект разметки занимается не более одного раза. Поэтому
    число TP совпадает с precision_recall_at_threshold на том же пороге.
    """
    p_boxes = np.asarray(p_boxes, dtype=np.float32).reshape(-1, 4)
    p_scores = np.asarray(p_scores, dtype=np.float32).reshape(-1)
    p_labels = np.asarray(p_labels).reshape(-1)
    g_boxes = np.asarray(g_boxes, dtype=np.float32).reshape(-1, 4)
    g_labels = np.asarray(g_labels).reshape(-1)

    out = ImageErrors()
    ious = box_iou(p_boxes, g_boxes)              # (P, G)
    same = p_labels[:, None] == g_labels[None, :]

    kept = np.where(p_scores >= score_threshold)[0]
    kept = kept[np.argsort(-p_scores[kept], kind="stable")]
    matched = np.zeros(len(g_boxes), dtype=bool)

    for pi in kept:
        if len(g_boxes):
            row = np.where(same[pi] & ~matched, ious[pi], 0.0)
            gi = int(row.argmax())
            if row[gi] >= iou_threshold:
                matched[gi] = True
                out.tp.append(int(pi))
                continue

        if not len(g_boxes):
            out.fp.append((int(pi), "bkg"))
            continue
        own, other = ious[pi][same[pi]], ious[pi][~same[pi]]
        if own.size and own.max() >= iou_threshold:
            kind = "dup"
        elif other.size and other.max() >= iou_threshold:
            kind = "cls"
            gi = int(np.where(~same[pi], ious[pi], -1.0).argmax())
            out.confusions.append((int(p_labels[pi]), int(g_labels[gi])))
        elif own.size and own.max() >= LOC_IOU_MIN:
            kind = "loc"
        elif ious[pi].max() < LOC_IOU_MIN:
            kind = "bkg"
        else:
            kind = "other"
        out.fp.append((int(pi), kind))

    below = p_scores < score_threshold
    for gi in np.where(~matched)[0]:
        hit = ious[:, gi] >= iou_threshold
        if (hit & same[:, gi] & below).any():
            reason = "low_score"
        elif (hit & ~same[:, gi] & ~below).any():
            reason = "wrong_class"
        else:
            reason = "missed"
        out.fn.append((int(gi), reason))
    return out
