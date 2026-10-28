"""
Калибровка детектора: насколько confidence соответствует фактической точности.

Для автоматической разметки калибровка важнее, чем для обычной детекции.
Псевдоразметка строится отсечением по порогу confidence, и порог подбирается
на размеченном домене-источнике. Если на целевом домене модель остаётся
уверенной там, где уже ошибается, перенесённый порог пропустит шум в обучающую
выборку ученика. Деградация mAP это не покажет: падение AP и рост
переуверенности — разные величины, и вторая прямо определяет пригодность
модели как учителя.

Реализованы:
    D-ECE  — detection ECE, адаптация ECE к детекции (Küppers et al., CVPR-W 2020)
    Brier  — квадратичная ошибка вероятности
    MCE    — максимальная ошибка калибровки по бину
    reliability diagram — данные для графика надёжности
    temperature / vector scaling — посткалибровка на валидации источника
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


# --------------------------------------------------------------------------- #
#                                  D-ECE                                      #
# --------------------------------------------------------------------------- #

@dataclass
class CalibrationResult:
    ece: float                      # средневзвешенная ошибка калибровки
    mce: float                      # максимальная ошибка по бину
    brier: float
    bin_edges: np.ndarray
    bin_confidence: np.ndarray      # средний confidence в бине
    bin_accuracy: np.ndarray        # доля TP в бине
    bin_counts: np.ndarray
    signed_gap: float               # >0 переуверенность, <0 недоуверенность

    def as_row(self) -> dict:
        return {
            "D-ECE": self.ece,
            "MCE": self.mce,
            "Brier": self.brier,
            "signed_gap": self.signed_gap,
        }


def detection_ece(
    scores: np.ndarray,
    tp: np.ndarray,
    n_bins: int = 15,
    adaptive: bool = True,
) -> CalibrationResult:
    """D-ECE: ошибка калибровки по сопоставленным предсказаниям.

    scores : confidence предсказаний
    tp     : True, если предсказание сопоставлено с объектом разметки

    adaptive=True задаёт бины равной населённости вместо равной ширины.
    Это существенно: распределение confidence у детекторов резко скошено к
    нулю, и при равных по ширине бинах верхние содержат единицы примеров,
    а их шум доминирует в MCE. Равнонаселённые бины дают устойчивую оценку
    на выборке в сотни предсказаний.

    signed_gap отделяет переуверенность от недоуверенности. Обычный ECE
    берёт модуль и теряет направление, а для фильтрации псевдоразметки
    опасна именно переуверенность: порог тогда пропускает ложные боксы.
    """
    if len(scores) == 0:
        nan = float("nan")
        z = np.zeros(0)
        return CalibrationResult(nan, nan, nan, z, z, z, z, nan)

    scores = np.asarray(scores, dtype=np.float64)
    tp = np.asarray(tp, dtype=np.float64)

    if adaptive:
        quantiles = np.linspace(0, 1, n_bins + 1)
        edges = np.unique(np.quantile(scores, quantiles))
        if len(edges) < 2:
            edges = np.array([scores.min(), scores.max() + 1e-6])
    else:
        edges = np.linspace(0.0, 1.0, n_bins + 1)

    bin_idx = np.clip(np.digitize(scores, edges[1:-1], right=False), 0, len(edges) - 2)

    n_actual = len(edges) - 1
    bin_conf = np.zeros(n_actual)
    bin_acc = np.zeros(n_actual)
    bin_cnt = np.zeros(n_actual, dtype=int)

    for b in range(n_actual):
        mask = bin_idx == b
        bin_cnt[b] = int(mask.sum())
        if bin_cnt[b] > 0:
            bin_conf[b] = scores[mask].mean()
            bin_acc[b] = tp[mask].mean()

    nonempty = bin_cnt > 0
    weights = bin_cnt[nonempty] / bin_cnt.sum()
    gaps = bin_conf[nonempty] - bin_acc[nonempty]

    ece = float(np.sum(weights * np.abs(gaps)))
    mce = float(np.max(np.abs(gaps))) if nonempty.any() else float("nan")
    signed = float(np.sum(weights * gaps))
    brier = float(np.mean((scores - tp) ** 2))

    return CalibrationResult(
        ece=ece,
        mce=mce,
        brier=brier,
        bin_edges=edges,
        bin_confidence=bin_conf,
        bin_accuracy=bin_acc,
        bin_counts=bin_cnt,
        signed_gap=signed,
    )


# --------------------------------------------------------------------------- #
#                    Перенос рабочего порога между доменами                   #
# --------------------------------------------------------------------------- #

def threshold_transfer_penalty(
    source_scores: np.ndarray,
    source_tp: np.ndarray,
    target_scores: np.ndarray,
    target_tp: np.ndarray,
    target_precision: float = 0.9,
) -> dict:
    """Во что обходится перенос порога, подобранного на источнике, в целевой домен.

    Это прямое измерение практического риска, который не виден ни в mAP, ни в
    D-ECE по отдельности. Сценарий буквально воспроизводит продакшен: порог
    подбирают один раз на размеченных данных, а применяют там, где разметки нет.

    Возвращает фактическую precision на целевом домене при перенесённом пороге
    и её разрыв с оракульным порогом, подобранным по самому целевому домену.
    Большой разрыв означает, что фильтрация по фиксированному порогу
    непригодна и нужна адаптивная калибровка.
    """

    def _best_threshold(scores: np.ndarray, tp: np.ndarray, target_p: float) -> float:
        if len(scores) == 0:
            return 1.0
        order = np.argsort(-scores)
        s, t = scores[order], tp[order]
        tp_cum = np.cumsum(t)
        precision = tp_cum / np.arange(1, len(s) + 1)
        ok = np.where(precision >= target_p)[0]
        if len(ok) == 0:
            return 1.0
        return float(s[ok[-1]])  # самый низкий порог, ещё дающий нужную точность

    def _precision_at(scores: np.ndarray, tp: np.ndarray, thr: float) -> tuple[float, int]:
        keep = scores >= thr
        if keep.sum() == 0:
            return float("nan"), 0
        return float(tp[keep].mean()), int(keep.sum())

    thr_source = _best_threshold(source_scores, source_tp, target_precision)
    thr_oracle = _best_threshold(target_scores, target_tp, target_precision)

    p_transferred, n_transferred = _precision_at(target_scores, target_tp, thr_source)
    p_oracle, n_oracle = _precision_at(target_scores, target_tp, thr_oracle)

    return {
        "threshold_source": thr_source,
        "threshold_oracle_target": thr_oracle,
        "precision_transferred": p_transferred,
        "precision_oracle": p_oracle,
        "precision_gap": (p_oracle - p_transferred) if not np.isnan(p_transferred) else float("nan"),
        "n_boxes_transferred": n_transferred,
        "n_boxes_oracle": n_oracle,
        # Во сколько раз меняется объём псевдоразметки при переносе порога:
        # падение объёма — скрытая цена сохранения чистоты.
        # Если целевая precision в домене недостижима ни при каком пороге,
        # оракула нет и отношение не определено
        "yield_ratio": n_transferred / n_oracle if n_oracle else float("nan"),
    }


# --------------------------------------------------------------------------- #
#                            Посткалибровка                                   #
# --------------------------------------------------------------------------- #

def fit_temperature(scores: np.ndarray, tp: np.ndarray, max_iter: int = 200) -> float:
    """Temperature scaling в логит-пространстве, подбор по NLL.

    Калибруется на домене-источнике и переносится на целевой. Если после
    калибровки на источнике D-ECE на целевом домене не улучшается, это
    доказывает, что переуверенность вызвана сдвигом домена, а не свойством
    самой модели — содержательный результат, а не технический шаг.
    """
    eps = 1e-6
    s = np.clip(np.asarray(scores, dtype=np.float64), eps, 1 - eps)
    y = np.asarray(tp, dtype=np.float64)
    logits = np.log(s / (1 - s))

    lo, hi = 0.05, 10.0
    for _ in range(max_iter):
        mid1 = lo + (hi - lo) / 3
        mid2 = hi - (hi - lo) / 3
        if _nll(logits, y, mid1) < _nll(logits, y, mid2):
            hi = mid2
        else:
            lo = mid1
        if hi - lo < 1e-4:
            break
    return float((lo + hi) / 2)


def _nll(logits: np.ndarray, y: np.ndarray, T: float) -> float:
    p = 1.0 / (1.0 + np.exp(-logits / T))
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return float(-np.mean(y * np.log(p) + (1 - y) * np.log(1 - p)))


def apply_temperature(scores: np.ndarray, T: float) -> np.ndarray:
    eps = 1e-6
    s = np.clip(np.asarray(scores, dtype=np.float64), eps, 1 - eps)
    logits = np.log(s / (1 - s))
    return 1.0 / (1.0 + np.exp(-logits / T))


# --------------------------------------------------------------------------- #
#                         Данные для графика надёжности                       #
# --------------------------------------------------------------------------- #

def reliability_data(result: CalibrationResult) -> dict:
    """Точки для диаграммы надёжности. Пустые бины исключены, чтобы не
    рисовать ложные нули на графике."""
    mask = result.bin_counts > 0
    return {
        "confidence": result.bin_confidence[mask].tolist(),
        "accuracy": result.bin_accuracy[mask].tolist(),
        "counts": result.bin_counts[mask].tolist(),
        "gap": (result.bin_confidence[mask] - result.bin_accuracy[mask]).tolist(),
    }
