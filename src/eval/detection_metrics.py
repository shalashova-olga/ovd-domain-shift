"""
Метрики качества детекции: сопоставление предсказаний с разметкой, AP, P/R/F1.

Реализация самостоятельная, а не через pycocotools, по двум причинам.
Во-первых, для оценки пригодности модели как автоматического разметчика нужны
не только AP, но и precision/recall при конкретном рабочем пороге — pycocotools
их не отдаёт. Во-вторых, результаты сопоставления (какое предсказание с каким
объектом совпало) нужны дальше для калибровки и анализа ошибок, а внутрь
pycocotools не заглянуть.

Протокол сопоставления соответствует COCO: жадный матчинг по убыванию
confidence, один объект разметки матчится не более одного раза, IoU считается
внутри класса. Совпадение с pycocotools проверяется тестом в tests/.
"""

from __future__ import annotations

from collections import defaultdict
from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
#                                    IoU                                      #
# --------------------------------------------------------------------------- #

def box_iou(boxes_a: np.ndarray, boxes_b: np.ndarray) -> np.ndarray:
    """Попарный IoU. (N,4) x (M,4) -> (N,M). Формат xyxy."""
    if len(boxes_a) == 0 or len(boxes_b) == 0:
        return np.zeros((len(boxes_a), len(boxes_b)), dtype=np.float32)

    a = boxes_a[:, None, :]
    b = boxes_b[None, :, :]

    inter_x1 = np.maximum(a[..., 0], b[..., 0])
    inter_y1 = np.maximum(a[..., 1], b[..., 1])
    inter_x2 = np.minimum(a[..., 2], b[..., 2])
    inter_y2 = np.minimum(a[..., 3], b[..., 3])

    inter_w = np.clip(inter_x2 - inter_x1, 0, None)
    inter_h = np.clip(inter_y2 - inter_y1, 0, None)
    inter = inter_w * inter_h

    area_a = (a[..., 2] - a[..., 0]) * (a[..., 3] - a[..., 1])
    area_b = (b[..., 2] - b[..., 0]) * (b[..., 3] - b[..., 1])
    union = area_a + area_b - inter
    return (inter / np.maximum(union, 1e-9)).astype(np.float32)


# --------------------------------------------------------------------------- #
#                              Сопоставление                                  #
# --------------------------------------------------------------------------- #

@dataclass
class MatchResult:
    """Результат сопоставления по всему набору — основа для всех метрик.

    scores     : (N,) confidence каждого предсказания, отсортированы по убыванию
    tp         : (N,) bool, предсказание совпало с ещё не занятым объектом
    labels     : (N,) класс предсказания
    iou        : (N,) IoU с сопоставленным объектом (0 для ложных срабатываний)
    areas      : (N,) площадь бокса предсказания — для разбивки по размеру
    image_ids  : (N,) к какому изображению относится предсказание
    n_gt       : число объектов разметки по классам
    """

    scores: np.ndarray
    tp: np.ndarray
    labels: np.ndarray
    iou: np.ndarray
    areas: np.ndarray
    image_ids: np.ndarray
    n_gt: dict[int, int]
    #: объекты разметки, которые не нашло ни одно предсказание — для анализа пропусков
    missed_gt: list[tuple[str, int, np.ndarray]]


def match_detections(
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ground_truth: dict[str, tuple[np.ndarray, np.ndarray]],
    iou_threshold: float = 0.5,
) -> MatchResult:
    """Сопоставить предсказания с разметкой по протоколу COCO.

    predictions  : image_id -> (boxes, scores, labels)
    ground_truth : image_id -> (boxes, labels)

    Жадность по confidence принципиальна: она отражает порядок, в котором
    предсказания отбирались бы при построении псевдоразметки с порогом.
    """
    n_gt: dict[int, int] = defaultdict(int)
    for _, (gt_boxes, gt_labels) in ground_truth.items():
        for lbl in gt_labels:
            n_gt[int(lbl)] += 1

    all_scores, all_tp, all_labels, all_iou, all_areas, all_imgids = [], [], [], [], [], []
    missed: list[tuple[str, int, np.ndarray]] = []

    for image_id, (gt_boxes, gt_labels) in ground_truth.items():
        pred = predictions.get(image_id)
        matched_gt = np.zeros(len(gt_boxes), dtype=bool)

        if pred is None or len(pred[0]) == 0:
            for gi in range(len(gt_boxes)):
                missed.append((image_id, int(gt_labels[gi]), gt_boxes[gi]))
            continue

        p_boxes, p_scores, p_labels = pred
        order = np.argsort(-p_scores)  # по убыванию confidence
        p_boxes, p_scores, p_labels = p_boxes[order], p_scores[order], p_labels[order]

        ious = box_iou(p_boxes, gt_boxes) if len(gt_boxes) else np.zeros((len(p_boxes), 0))
        # IoU считается только внутри класса: пары разных классов обнуляются заранее
        cand = np.where(p_labels[:, None] == gt_labels[None, :], ious, 0.0)

        for pi in range(len(p_boxes)):
            best_iou, best_gi = 0.0, -1
            if len(gt_boxes):
                row = np.where(matched_gt, 0.0, cand[pi])
                gi = int(row.argmax())  # при равенстве — первый, как в переборе
                if row[gi] > 0.0:
                    best_iou, best_gi = float(row[gi]), gi

            is_tp = best_iou >= iou_threshold and best_gi >= 0
            if is_tp:
                matched_gt[best_gi] = True

            w = p_boxes[pi, 2] - p_boxes[pi, 0]
            h = p_boxes[pi, 3] - p_boxes[pi, 1]

            all_scores.append(p_scores[pi])
            all_tp.append(is_tp)
            all_labels.append(int(p_labels[pi]))
            all_iou.append(best_iou if is_tp else 0.0)
            all_areas.append(float(w * h))
            all_imgids.append(image_id)

        for gi in np.where(~matched_gt)[0]:
            missed.append((image_id, int(gt_labels[gi]), gt_boxes[gi]))

    if not all_scores:
        empty = np.zeros(0)
        return MatchResult(empty, empty.astype(bool), empty.astype(int), empty,
                           empty, np.array([], dtype=object), dict(n_gt), missed)

    scores = np.array(all_scores, dtype=np.float32)
    order = np.argsort(-scores)
    return MatchResult(
        scores=scores[order],
        tp=np.array(all_tp, dtype=bool)[order],
        labels=np.array(all_labels, dtype=np.int32)[order],
        iou=np.array(all_iou, dtype=np.float32)[order],
        areas=np.array(all_areas, dtype=np.float32)[order],
        image_ids=np.array(all_imgids, dtype=object)[order],
        n_gt=dict(n_gt),
        missed_gt=missed,
    )


# --------------------------------------------------------------------------- #
#                            Average Precision                                #
# --------------------------------------------------------------------------- #

def average_precision(tp: np.ndarray, n_gt: int, interpolate: bool = True) -> float:
    """AP по кривой precision-recall. tp отсортирован по убыванию confidence.

    interpolate=True даёт монотонную огибающую (протокол COCO/VOC2010+).
    Без интерполяции AP занижается на шумных кривых, характерных для малых
    выборок — различие доходит до 1-2 п.п. на 400 изображениях.
    """
    if n_gt == 0:
        return float("nan")  # класс отсутствует в домене — не 0, иначе mAP смещается
    if len(tp) == 0:
        return 0.0

    tp_cum = np.cumsum(tp)
    fp_cum = np.cumsum(~tp)
    recall = tp_cum / n_gt
    precision = tp_cum / np.maximum(tp_cum + fp_cum, 1e-9)

    if interpolate:
        precision = np.maximum.accumulate(precision[::-1])[::-1]

    recall = np.concatenate([[0.0], recall, [recall[-1]]])
    precision = np.concatenate([[precision[0] if len(precision) else 0.0], precision, [0.0]])
    return float(np.sum((recall[1:] - recall[:-1]) * precision[1:]))


def compute_map(
    predictions: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]],
    ground_truth: dict[str, tuple[np.ndarray, np.ndarray]],
    class_names: list[str],
    iou_thresholds: np.ndarray | None = None,
) -> dict:
    """mAP@50, mAP@50:95 и AP по классам.

    Классы, отсутствующие в домене, исключаются из усреднения (AP = nan).
    Это важно для ACDC: train встречается в единичных кадрах, и его нулевой AP
    при наивном усреднении сдвинул бы mAP на несколько пунктов, создав ложную
    картину деградации.
    """
    if iou_thresholds is None:
        iou_thresholds = np.arange(0.5, 1.0, 0.05)

    per_iou: dict[float, dict[int, float]] = {}
    for thr in iou_thresholds:
        m = match_detections(predictions, ground_truth, iou_threshold=float(thr))
        per_class: dict[int, float] = {}
        for cls_idx in range(len(class_names)):
            mask = m.labels == cls_idx
            per_class[cls_idx] = average_precision(m.tp[mask], m.n_gt.get(cls_idx, 0))
        per_iou[float(thr)] = per_class

    ap50 = per_iou[0.5]
    valid50 = [v for v in ap50.values() if not np.isnan(v)]

    all_vals = [
        v for thr_map in per_iou.values() for v in thr_map.values() if not np.isnan(v)
    ]

    return {
        "mAP@50": float(np.mean(valid50)) if valid50 else float("nan"),
        "mAP@50:95": float(np.mean(all_vals)) if all_vals else float("nan"),
        "AP@50_per_class": {class_names[i]: ap50[i] for i in range(len(class_names))},
        "n_gt_per_class": {class_names[i]: m.n_gt.get(i, 0) for i in range(len(class_names))},
    }


# --------------------------------------------------------------------------- #
#                  Precision / Recall при рабочем пороге                      #
# --------------------------------------------------------------------------- #

def precision_recall_at_threshold(m: MatchResult, score_threshold: float) -> dict:
    """P/R/F1 при конкретном пороге confidence.

    Для задачи автоматической разметки это главная метрика, а не AP: в
    продакшене порог один и фиксированный, и важно, какая доля отобранных
    боксов окажется верной. Precision здесь ценнее recall — ошибочный бокс
    в обучающей выборке ученика вреднее пропущенного.
    """
    keep = m.scores >= score_threshold
    tp = int(m.tp[keep].sum())
    fp = int((~m.tp[keep]).sum())
    total_gt = sum(m.n_gt.values())
    fn = total_gt - tp

    precision = tp / max(tp + fp, 1)
    recall = tp / max(total_gt, 1)
    f1 = 2 * precision * recall / max(precision + recall, 1e-9)

    return {
        "threshold": score_threshold,
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "n_kept": int(keep.sum()),
    }


def threshold_sweep(
    m: MatchResult, thresholds: np.ndarray | None = None
) -> list[dict]:
    """P/R/F1 по сетке порогов — для выбора рабочей точки и переноса её между доменами."""
    if thresholds is None:
        thresholds = np.arange(0.05, 1.0, 0.05)
    return [precision_recall_at_threshold(m, float(t)) for t in thresholds]


def threshold_for_target_precision(
    m: MatchResult, target_precision: float = 0.9
) -> tuple[float, dict]:
    """Минимальный порог, обеспечивающий заданную precision.

    Это и есть рабочая точка учителя: чтобы псевдоразметка годилась для
    обучения ученика, нужна высокая чистота, а recall — то, чем за неё платят.
    Возвращает (порог, метрики). Если точность недостижима ни при каком
    пороге, возвращает (1.0, метрики при максимальном пороге) — сигнал, что
    модель непригодна как учитель на этом домене.
    """
    sweep = threshold_sweep(m, np.arange(0.05, 1.0, 0.01))
    for row in sweep:
        if row["precision"] >= target_precision:
            return row["threshold"], row
    return 1.0, sweep[-1] if sweep else {}


# --------------------------------------------------------------------------- #
#                        Разбивка для анализа ошибок                          #
# --------------------------------------------------------------------------- #

# Границы COCO для разбиения по размеру объекта, в пикселях площади
AREA_RANGES = {"small": (0, 32**2), "medium": (32**2, 96**2), "large": (96**2, np.inf)}


def breakdown_by_area(m: MatchResult, score_threshold: float = 0.3) -> dict[str, dict]:
    """P/R по размеру объекта. Мелкие объекты — основной источник деградации
    при тумане и ночью, и разбивка показывает это количественно."""
    out = {}
    keep = m.scores >= score_threshold
    for name, (lo, hi) in AREA_RANGES.items():
        mask = keep & (m.areas >= lo) & (m.areas < hi)
        tp = int(m.tp[mask].sum())
        fp = int((~m.tp[mask]).sum())
        out[name] = {
            "n_pred": int(mask.sum()),
            "tp": tp,
            "fp": fp,
            "precision": tp / max(tp + fp, 1),
        }
    return out


def breakdown_by_class(m: MatchResult, class_names: list[str], score_threshold: float = 0.3) -> dict:
    out = {}
    keep = m.scores >= score_threshold
    for i, name in enumerate(class_names):
        mask = keep & (m.labels == i)
        tp = int(m.tp[mask].sum())
        fp = int((~m.tp[mask]).sum())
        n_gt = m.n_gt.get(i, 0)
        out[name] = {
            "n_gt": n_gt,
            "tp": tp,
            "fp": fp,
            "precision": tp / max(tp + fp, 1),
            "recall": tp / max(n_gt, 1),
        }
    return out


# --------------------------------------------------------------------------- #
#                      Доверительный интервал (bootstrap)                     #
# --------------------------------------------------------------------------- #

def _per_image_matches(
    predictions: dict, ground_truth: dict, n_classes: int, iou_threshold: float
) -> dict[str, tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]]:
    """Результат сопоставления, разложенный по кадрам:
    image_id -> (scores, tp, labels, число объектов разметки по классам)."""
    m = match_detections(predictions, ground_truth, iou_threshold=iou_threshold)
    rows: dict[str, list[int]] = defaultdict(list)
    for i, img in enumerate(m.image_ids):
        rows[img].append(i)

    parts = {}
    for img, (_, gt_labels) in ground_truth.items():
        idx = np.array(rows.get(img, []), dtype=np.int64)
        n_gt = np.bincount(np.asarray(gt_labels, dtype=np.int64), minlength=n_classes)
        parts[img] = (m.scores[idx], m.tp[idx], m.labels[idx], n_gt)
    return parts


def _map50_from_parts(parts: list[tuple], n_classes: int) -> float:
    """mAP по набору кадров, для которых сопоставление уже выполнено."""
    scores = np.concatenate([p[0] for p in parts])
    tp = np.concatenate([p[1] for p in parts]).astype(bool)
    labels = np.concatenate([p[2] for p in parts])
    n_gt = np.sum([p[3] for p in parts], axis=0)

    order = np.argsort(-scores, kind="stable")
    tp, labels = tp[order], labels[order]
    aps = [average_precision(tp[labels == c], int(n_gt[c])) for c in range(n_classes)]
    aps = [a for a in aps if not np.isnan(a)]
    return float(np.mean(aps)) if aps else float("nan")


def bootstrap_ci(
    predictions: dict,
    ground_truth: dict,
    class_names: list[str],
    metric_fn=None,
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
    iou_threshold: float = 0.5,
) -> tuple[float, float, float]:
    """Доверительный интервал метрики через бутстрэп по изображениям.

    Ресэмплинг идёт по изображениям, а не по отдельным предсказаниям: объекты
    внутри кадра коррелированы (одна погода, одно освещение, один ракурс), и
    бутстрэп по боксам занизил бы ширину интервала в разы.

    На 400 изображениях интервал mAP обычно оказывается шириной 3-5 п.п. Это
    и есть причина, по которой разницу моделей в 1 п.п. нельзя объявлять
    значимой — вывод, который в сравнительных работах часто опускают.

    По умолчанию считается mAP@50, и сопоставление выполняется один раз:
    итерации переиспользуют результаты по кадрам. Это точно, а не
    приближённо — жадный матчинг идёт внутри кадра и от остальной выборки не
    зависит. Пересчёт compute_map на каждой итерации (10 порогов IoU и
    сопоставление заново) при низком пороге инференса занимал бы часы на одну
    пару модель–домен. metric_fn задаёт произвольную метрику через общий,
    медленный путь.
    """
    rng = np.random.default_rng(seed)
    image_ids = list(ground_truth.keys())

    if metric_fn is None:
        n_classes = len(class_names)
        point = compute_map(predictions, ground_truth, class_names)["mAP@50"]
        parts = _per_image_matches(predictions, ground_truth, n_classes, iou_threshold)

        def stat(sampled):
            return _map50_from_parts([parts[i] for i in sampled], n_classes)
    else:
        point = metric_fn(predictions, ground_truth)

        def stat(sampled):
            # Дубликаты изображений получают уникальные ключи, иначе словарь схлопнет их
            gt_b, pred_b = {}, {}
            for k, img_id in enumerate(sampled):
                key = f"{img_id}__{k}"
                gt_b[key] = ground_truth[img_id]
                if img_id in predictions:
                    pred_b[key] = predictions[img_id]
            return metric_fn(pred_b, gt_b)

    samples = [
        stat(rng.choice(image_ids, size=len(image_ids), replace=True)) for _ in range(n_boot)
    ]

    samples = np.array([s for s in samples if not np.isnan(s)])
    if len(samples) == 0:
        return point, float("nan"), float("nan")
    lo = float(np.percentile(samples, 100 * alpha / 2))
    hi = float(np.percentile(samples, 100 * (1 - alpha / 2)))
    return float(point), lo, hi
