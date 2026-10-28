"""
Проверка корректности метрик на случаях с известным ответом.

Тесты здесь не формальность: ошибка в матчинге или в AP даёт правдоподобные,
но неверные числа, которые нечем заметить в итоговой таблице. Каждый случай
подобран так, чтобы ответ считался вручную.

Запуск:  python -m pytest tests/ -v
"""

import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.eval.calibration import (
    apply_temperature,
    detection_ece,
    fit_temperature,
    threshold_transfer_penalty,
)
from src.eval.detection_metrics import (
    average_precision,
    box_iou,
    compute_map,
    match_detections,
    precision_recall_at_threshold,
)


# --------------------------------------------------------------------------- #
#                                   IoU                                       #
# --------------------------------------------------------------------------- #

def test_iou_identical_boxes():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    assert box_iou(a, a)[0, 0] == pytest.approx(1.0)


def test_iou_no_overlap():
    a = np.array([[0, 0, 10, 10]], dtype=np.float32)
    b = np.array([[20, 20, 30, 30]], dtype=np.float32)
    assert box_iou(a, b)[0, 0] == pytest.approx(0.0)


def test_iou_half_overlap():
    # Пересечение 10x20=200, объединение 400+400-200=600 -> 1/3
    a = np.array([[0, 0, 20, 20]], dtype=np.float32)
    b = np.array([[10, 0, 30, 20]], dtype=np.float32)
    assert box_iou(a, b)[0, 0] == pytest.approx(1 / 3, abs=1e-5)


def test_iou_empty_input():
    assert box_iou(np.zeros((0, 4)), np.array([[0, 0, 1, 1]])).shape == (0, 1)


# --------------------------------------------------------------------------- #
#                               Сопоставление                                 #
# --------------------------------------------------------------------------- #

def test_match_perfect_detection():
    gt = {"img1": (np.array([[0, 0, 10, 10]], dtype=np.float32), np.array([0]))}
    pred = {"img1": (np.array([[0, 0, 10, 10]], dtype=np.float32),
                     np.array([0.9]), np.array([0]))}
    m = match_detections(pred, gt, iou_threshold=0.5)
    assert m.tp.tolist() == [True]
    assert m.n_gt == {0: 1}
    assert len(m.missed_gt) == 0


def test_match_wrong_class_is_false_positive():
    """Идеальный бокс, но другой класс — ложное срабатывание и пропуск объекта."""
    gt = {"img1": (np.array([[0, 0, 10, 10]], dtype=np.float32), np.array([0]))}
    pred = {"img1": (np.array([[0, 0, 10, 10]], dtype=np.float32),
                     np.array([0.9]), np.array([1]))}
    m = match_detections(pred, gt, iou_threshold=0.5)
    assert m.tp.tolist() == [False]
    assert len(m.missed_gt) == 1


def test_match_duplicate_detections_one_tp():
    """Два бокса на один объект: TP получает только более уверенный."""
    gt = {"img1": (np.array([[0, 0, 10, 10]], dtype=np.float32), np.array([0]))}
    pred = {"img1": (np.array([[0, 0, 10, 10], [0, 0, 10, 10]], dtype=np.float32),
                     np.array([0.9, 0.8]), np.array([0, 0]))}
    m = match_detections(pred, gt, iou_threshold=0.5)
    assert m.tp.tolist() == [True, False]


def test_match_greedy_by_confidence():
    """Жадность по confidence: более уверенный бокс забирает объект первым,
    даже если менее уверенный совпадает с ним точнее."""
    gt = {"img1": (np.array([[0, 0, 10, 10]], dtype=np.float32), np.array([0]))}
    pred = {"img1": (np.array([[1, 1, 11, 11], [0, 0, 10, 10]], dtype=np.float32),
                     np.array([0.9, 0.5]), np.array([0, 0]))}
    m = match_detections(pred, gt, iou_threshold=0.5)
    assert m.scores.tolist() == [pytest.approx(0.9), pytest.approx(0.5)]
    assert m.tp.tolist() == [True, False]


def test_match_missing_image_counts_as_missed():
    gt = {"img1": (np.array([[0, 0, 10, 10]], dtype=np.float32), np.array([0]))}
    m = match_detections({}, gt, iou_threshold=0.5)
    assert len(m.missed_gt) == 1
    assert len(m.tp) == 0


# --------------------------------------------------------------------------- #
#                           Average Precision                                 #
# --------------------------------------------------------------------------- #

def test_ap_all_correct_is_one():
    assert average_precision(np.array([True, True, True]), n_gt=3) == pytest.approx(1.0)


def test_ap_absent_class_is_nan():
    """Класс, которого нет в домене, даёт nan, а не 0: иначе mAP смещается вниз."""
    assert np.isnan(average_precision(np.array([False]), n_gt=0))


def test_ap_half_recall():
    """Один TP при двух объектах: precision 1.0 до recall 0.5, затем обрыв."""
    ap = average_precision(np.array([True, False]), n_gt=2)
    assert ap == pytest.approx(0.5, abs=1e-6)


def test_ap_order_matters():
    """Тот же набор TP/FP, но TP впереди даёт более высокий AP."""
    good = average_precision(np.array([True, True, False, False]), n_gt=2)
    bad = average_precision(np.array([False, False, True, True]), n_gt=2)
    assert good > bad
    assert good == pytest.approx(1.0)


def test_map_excludes_absent_classes():
    gt = {"img1": (np.array([[0, 0, 10, 10]], dtype=np.float32), np.array([0]))}
    pred = {"img1": (np.array([[0, 0, 10, 10]], dtype=np.float32),
                     np.array([0.9]), np.array([0]))}
    res = compute_map(pred, gt, class_names=["car", "bus", "train"])
    # Присутствует только 'car' с AP=1.0; bus и train -> nan и не усредняются
    assert res["mAP@50"] == pytest.approx(1.0)
    assert np.isnan(res["AP@50_per_class"]["bus"])


# --------------------------------------------------------------------------- #
#                            Precision / Recall                               #
# --------------------------------------------------------------------------- #

def test_precision_recall_at_threshold():
    gt = {"img1": (np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32),
                   np.array([0, 0]))}
    pred = {"img1": (
        np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float32),
        np.array([0.9, 0.8]), np.array([0, 0]))}
    m = match_detections(pred, gt, iou_threshold=0.5)
    r = precision_recall_at_threshold(m, 0.5)
    assert r["tp"] == 1 and r["fp"] == 1 and r["fn"] == 1
    assert r["precision"] == pytest.approx(0.5)
    assert r["recall"] == pytest.approx(0.5)
    assert r["f1"] == pytest.approx(0.5)


def test_threshold_raises_precision():
    """Повышение порога отсекает неуверенное ложное срабатывание."""
    gt = {"img1": (np.array([[0, 0, 10, 10]], dtype=np.float32), np.array([0]))}
    pred = {"img1": (np.array([[0, 0, 10, 10], [50, 50, 60, 60]], dtype=np.float32),
                     np.array([0.9, 0.2]), np.array([0, 0]))}
    m = match_detections(pred, gt, iou_threshold=0.5)
    assert precision_recall_at_threshold(m, 0.1)["precision"] == pytest.approx(0.5)
    assert precision_recall_at_threshold(m, 0.5)["precision"] == pytest.approx(1.0)


# --------------------------------------------------------------------------- #
#                               Калибровка                                    #
# --------------------------------------------------------------------------- #

def test_ece_perfect_calibration_is_zero():
    """Confidence 0.5 при ровно половине верных предсказаний: ошибки нет."""
    scores = np.full(1000, 0.5)
    tp = np.array([True, False] * 500)
    res = detection_ece(scores, tp, n_bins=5)
    assert res.ece == pytest.approx(0.0, abs=0.02)
    assert abs(res.signed_gap) < 0.02


def test_ece_detects_overconfidence():
    """Модель уверена на 0.95, права в 50% случаев: переуверенность ~0.45."""
    scores = np.full(1000, 0.95)
    tp = np.array([True, False] * 500)
    res = detection_ece(scores, tp, n_bins=5)
    assert res.ece == pytest.approx(0.45, abs=0.03)
    assert res.signed_gap > 0.4          # знак положительный = переуверенность


def test_ece_detects_underconfidence():
    scores = np.full(1000, 0.2)
    tp = np.array([True] * 800 + [False] * 200)
    res = detection_ece(scores, tp, n_bins=5)
    assert res.signed_gap < -0.5         # знак отрицательный = недоуверенность


def test_brier_bounds():
    res = detection_ece(np.full(100, 1.0), np.ones(100, dtype=bool), n_bins=5)
    assert res.brier == pytest.approx(0.0)
    res = detection_ece(np.full(100, 1.0), np.zeros(100, dtype=bool), n_bins=5)
    assert res.brier == pytest.approx(1.0)


def test_temperature_scaling_reduces_ece():
    """Посткалибровка должна снижать ошибку на переуверенных скорах."""
    rng = np.random.default_rng(0)
    tp = rng.random(2000) < 0.6
    scores = np.where(tp, rng.uniform(0.85, 0.99, 2000), rng.uniform(0.80, 0.97, 2000))
    before = detection_ece(scores, tp, n_bins=10).ece
    T = fit_temperature(scores, tp)
    after = detection_ece(apply_temperature(scores, T), tp, n_bins=10).ece
    assert after < before
    assert T > 1.0                        # T>1 «растягивает» переуверенные скоры вниз


def test_threshold_transfer_detects_penalty():
    """Перенос порога между доменами с разной калибровкой должен давать
    падение фактической точности."""
    rng = np.random.default_rng(1)
    # Источник: высокая точность при высоких скорах
    s_src = rng.uniform(0.5, 1.0, 2000)
    tp_src = rng.random(2000) < s_src
    # Целевой домен: те же скоры, но точность систематически ниже
    s_tgt = rng.uniform(0.5, 1.0, 2000)
    tp_tgt = rng.random(2000) < (s_tgt * 0.6)

    res = threshold_transfer_penalty(s_src, tp_src, s_tgt, tp_tgt, target_precision=0.9)
    assert res["precision_transferred"] < 0.9      # перенесённый порог не держит точность
    assert res["precision_gap"] > 0


# --------------------------------------------------------------------------- #
#                         Санитизация выхода модели                           #
# --------------------------------------------------------------------------- #

def test_sanitize_catches_invalid_boxes():
    from src.models.base import BaseDetector

    class Dummy(BaseDetector):
        name = "dummy"

        def _load(self, **kw):
            pass

        def _predict(self, image, prompts):
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0), None

    d = Dummy(class_names=["car", "person"], device="cpu")
    boxes = np.array([
        [0, 0, 10, 10],        # валидный
        [10, 10, 5, 5],        # вырожденный: x2 < x1
        [200, 200, 250, 250],  # целиком вне кадра 100x100
        [-10, -10, 50, 50],    # частично вне кадра -> обрезка
        [0, 0, 10, 10],        # класс вне словаря
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7, 0.6, 0.5])
    labels = np.array([0, 0, 0, 1, 99])

    dets, rep = d._sanitize(boxes, scores, labels, (100, 100), None)
    assert rep.n_unknown_class == 1
    assert rep.n_out_of_bounds == 1
    assert rep.n_degenerate == 1
    assert rep.n_clipped == 1
    assert rep.n_kept == 2
    assert rep.invalid_rate == pytest.approx(3 / 5)
    assert (dets.boxes >= 0).all() and (dets.boxes <= 100).all()


def test_sanitize_clips_scores_to_unit_interval():
    """VLM может вернуть confidence вне [0,1] текстом — обрезаем молча,
    но бокс не выбрасываем."""
    from src.models.base import BaseDetector

    class Dummy(BaseDetector):
        name = "dummy"

        def _load(self, **kw):
            pass

        def _predict(self, image, prompts):
            return np.zeros((0, 4)), np.zeros(0), np.zeros(0), None

    d = Dummy(class_names=["car"], device="cpu")
    dets, _ = d._sanitize(
        np.array([[0, 0, 10, 10], [20, 20, 30, 30]], dtype=np.float32),
        np.array([1.5, -0.3]), np.array([0, 0]), (100, 100), None,
    )
    assert dets.scores.tolist() == [1.0, 0.0]


# --------------------------------------------------------------------------- #
#                      Сопоставление: класс важнее IoU                        #
# --------------------------------------------------------------------------- #

def test_match_ignores_better_iou_of_other_class():
    """Объект другого класса с идеальным IoU не должен перехватывать матч."""
    gt = {"img1": (np.array([[0, 0, 10, 10], [1, 1, 11, 11]], dtype=np.float32),
                   np.array([1, 0]))}
    pred = {"img1": (np.array([[0, 0, 10, 10]], dtype=np.float32),
                     np.array([0.9]), np.array([0]))}
    m = match_detections(pred, gt, iou_threshold=0.5)
    assert m.tp.tolist() == [True]
    assert m.iou[0] < 1.0                 # совпал объект своего класса, а не ближайший


# --------------------------------------------------------------------------- #
#                                 Бутстрэп                                    #
# --------------------------------------------------------------------------- #

def _random_detection_set(n_images=30, n_classes=3, seed=0):
    rng = np.random.default_rng(seed)
    gt, pred = {}, {}
    for i in range(n_images):
        k = rng.integers(0, 5)
        xy = rng.uniform(0, 80, (k, 2))
        g_boxes = np.hstack([xy, xy + rng.uniform(10, 20, (k, 2))]).astype(np.float32)
        g_labels = rng.integers(0, n_classes, k)
        gt[f"img{i}"] = (g_boxes, g_labels)
        # предсказания: зашумлённые копии разметки плюс случайные ложные боксы
        jitter = g_boxes + rng.normal(0, 3, g_boxes.shape).astype(np.float32)
        n_fp = rng.integers(0, 4)
        xy = rng.uniform(0, 80, (n_fp, 2))
        fp = np.hstack([xy, xy + 15]).astype(np.float32)
        boxes = np.vstack([jitter, fp])
        labels = np.concatenate([g_labels, rng.integers(0, n_classes, n_fp)])
        pred[f"img{i}"] = (boxes, rng.random(len(boxes)).astype(np.float32), labels)
    return pred, gt


def test_bootstrap_fast_path_equals_full_recompute():
    """Быстрый путь переиспользует сопоставление по кадрам; результат обязан
    совпадать с пересчётом compute_map на каждой итерации."""
    from src.eval.detection_metrics import bootstrap_ci

    pred, gt = _random_detection_set()
    classes = ["a", "b", "c"]
    fast = bootstrap_ci(pred, gt, classes, n_boot=30, seed=3)
    slow = bootstrap_ci(pred, gt, classes, n_boot=30, seed=3,
                        metric_fn=lambda p, g: compute_map(p, g, classes)["mAP@50"])
    assert fast == pytest.approx(slow, abs=1e-9)
    assert fast[1] <= fast[0] <= fast[2]


# --------------------------------------------------------------------------- #
#                  Grounding DINO: класс по токенам фразы                     #
# --------------------------------------------------------------------------- #

def test_grounding_text_format_and_spans():
    from src.models.adapters import build_grounding_text

    text, spans = build_grounding_text(["Person", "a car."])
    assert text == "person. a car."
    assert [text[s:e] for s, e in spans] == ["person", "a car"]


def test_phrase_scores_separate_prompts_with_shared_words():
    """Варианты 'a photo of a X on a road' почти целиком совпадают. Класс
    должен определяться токеном, различающим фразы, а общие слова не должны
    приписывать бокс первому классу."""
    from src.models.adapters import build_grounding_text, phrase_scores, phrase_token_indices

    prompts = ["a photo of a person on a road", "a photo of a car on a road"]
    text, spans = build_grounding_text(prompts)
    # пословная «токенизация» с [CLS] и [SEP], как у fast-токенизатора
    offsets = [(0, 0)]
    pos = 0
    for word in text.replace(".", " .").split():
        start = text.index(word, pos)
        offsets.append((start, start + len(word)))
        pos = start + len(word)
    offsets.append((0, 0))

    idx = phrase_token_indices(offsets, spans)
    assert all(len(i) == 8 for i in idx)          # точки и служебные токены не входят

    car_token = next(i for i, (a, b) in enumerate(offsets) if text[a:b] == "car")
    probs = np.full((2, len(offsets)), 0.1)
    probs[0, car_token] = 0.8                     # запрос 0 нашёл машину
    shared = [i for i, (a, b) in enumerate(offsets) if text[a:b] in ("a", "photo", "of")]
    probs[1, shared] = 0.6                        # запрос 1 активировал только 'a photo of a'
    s = phrase_scores(probs, idx)
    assert s[0].argmax() == 1
    assert s[1, 0] == pytest.approx(s[1, 1])      # общие слова не дают перевеса ни одному классу


# --------------------------------------------------------------------------- #
#                         Формулировки запросов                               #
# --------------------------------------------------------------------------- #

def test_build_prompts_applies_class_overrides():
    from src.run_inference import build_prompts

    cfg = {
        "variants": {"plain": "{cls}", "contextual": "a photo of a {cls} on a road"},
        "class_overrides": {"rider": {"contextual": "a person riding a bicycle"}},
    }
    assert build_prompts(cfg, "plain", ["car", "rider"]) == ["car", "rider"]
    assert build_prompts(cfg, "contextual", ["car", "rider"]) == [
        "a photo of a car on a road", "a person riding a bicycle"]


def test_prompt_sensitivity_end_to_end(tmp_path):
    """Сквозная проверка: журналы трёх вариантов -> таблицы разброса и рангов."""
    import json

    import pandas as pd
    import yaml

    from src.run_eval import prompt_sensitivity

    pred, gt = _random_detection_set(n_images=10, n_classes=2)

    class FakeDataset:
        classes = ["a", "b"]

        def ground_truth(self):
            return gt

    variants = {"plain": "{cls}", "article": "a {cls}", "contextual": "a {cls} on a road"}
    (tmp_path / "prompts.yaml").write_text(yaml.safe_dump({"variants": variants}))
    models = {"m1": 0.0, "m2": 5.0, "m3": 15.0}   # шум боксов задаёт разное качество
    rng = np.random.default_rng(0)
    for model, noise in models.items():
        for variant in variants:
            path = tmp_path / "pred" / model / f"night__{variant}.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            with open(path, "w") as f:
                for img, (b, s, lbl) in pred.items():
                    b = b + rng.normal(0, noise + 1, b.shape)
                    f.write(json.dumps({
                        "image_id": img, "boxes": b.tolist(), "scores": s.tolist(),
                        "labels": lbl.tolist(), "validity": {"n_raw": len(b)},
                        "latency_ms": 1.0, "peak_vram_mb": 0.0}) + "\n")

    cfg = {"prompts_file": str(tmp_path / "prompts.yaml"), "output_dir": str(tmp_path / "pred"),
           "models": models, "prompt_sensitivity_domains": ["night"],
           "default_prompt_variant": "plain", "evaluation": {"bootstrap_iterations": 50}}
    prompt_sensitivity(cfg, {"night": FakeDataset()}, tmp_path)

    summary = pd.read_csv(tmp_path / "prompt_sensitivity_summary.csv")
    assert len(summary) == 3 and (summary["count"] == 3).all()
    assert (summary["range"] >= 0).all()
    assert summary["mAP50_ci_width"].between(0, 1).all()   # интервал посчитан на общих кадрах
    assert set(summary["best_variant"]) <= set(variants)
    corr = pd.read_csv(tmp_path / "prompt_rank_correlation.csv")
    assert set(corr["domain"]) == {"night"}


# --------------------------------------------------------------------------- #
#                         Раскладка каталога ACDC                             #
# --------------------------------------------------------------------------- #

def _write_coco(path, file_names, cats=("car", "person")):
    import json
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps({
        "categories": [{"id": i + 1, "name": c} for i, c in enumerate(cats)],
        "images": [{"id": i + 1, "file_name": f, "width": 100, "height": 100}
                   for i, f in enumerate(file_names)],
        "annotations": [{"id": i + 1, "image_id": i + 1, "category_id": 1,
                         "bbox": [0, 0, 20, 20], "area": 400.0, "iscrowd": 0}
                        for i in range(len(file_names))],
    }))


def test_acdc_official_layout_and_reference_merge(tmp_path):
    """Официальная раскладка: разметка по условиям, пути кадров относительно
    rgb_anon, reference склеивается из файлов разных условий с пересекающимися id."""
    from src.data.acdc import ACDCDetection

    root = tmp_path / "ACDC"
    for cond in ("fog", "night"):
        img = f"{cond}/val/GOPR01/GOPR01_frame_{cond}_rgb_anon.png"
        ref = f"{cond}/val_ref/GOPR01/GOPR01_frame_{cond}_rgb_ref_anon.png"
        for f in (img, ref):
            (root / "rgb_anon" / f).parent.mkdir(parents=True, exist_ok=True)
            (root / "rgb_anon" / f).touch()
        _write_coco(root / "gt_detection" / cond / f"instancesonly_{cond}_val_gt_detection.json", [img])
        _write_coco(root / "gt_detection" / cond / f"instancesonly_{cond}_val_ref_gt_detection.json", [ref])

    fog = ACDCDetection(root, "fog", classes=["car", "person"])
    assert len(fog) == 1                          # ref-файл того же условия не подмешан
    assert fog.samples[0].image_path.exists()

    ref = ACDCDetection(root, "reference", classes=["car", "person"])
    assert len(ref) == 2                          # по одному кадру из fog и night
    assert len({s.image_id for s in ref}) == 2    # одинаковые числовые id не схлопнулись
    assert all(s.image_path.exists() for s in ref)

    with pytest.raises(FileNotFoundError):
        ACDCDetection(root, "snow")


# --------------------------------------------------------------------------- #
#                            Типизация ошибок                                 #
# --------------------------------------------------------------------------- #

def test_classify_image_error_types():
    """Каждый тип ложного срабатывания и каждая причина пропуска на одном кадре."""
    from src.eval.error_analysis import classify_image

    gt_boxes = np.array([
        [0, 0, 10, 10],        # 0: car — найден, плюс дубль
        [20, 0, 30, 10],       # 1: person — назван bicycle (не тот класс)
        [40, 0, 50, 10],       # 2: car — найден только с низким скором
        [60, 0, 70, 10],       # 3: bus — не найден вовсе
        [80, 0, 90, 10],       # 4: car — бокс сильно смещён (локализация)
    ], dtype=np.float32)
    gt_labels = np.array([0, 1, 0, 2, 0])
    pred = np.array([
        [0, 0, 10, 10],        # TP
        [0, 0, 10, 11],        # dup
        [20, 0, 30, 10],       # cls: bicycle(3) на месте person
        [40, 0, 50, 10],       # ниже порога -> пропуск low_score
        [86, 0, 96, 10],       # loc: IoU 4/16 = 0.25
        [200, 200, 210, 210],  # bkg
    ], dtype=np.float32)
    scores = np.array([0.9, 0.8, 0.7, 0.1, 0.6, 0.5])
    labels = np.array([0, 0, 3, 0, 0, 1])

    e = classify_image(pred, scores, labels, gt_boxes, gt_labels, score_threshold=0.3)
    assert e.tp == [0]
    assert dict(e.fp) == {1: "dup", 2: "cls", 4: "loc", 5: "bkg"}
    assert dict(e.fn) == {1: "wrong_class", 2: "low_score", 3: "missed", 4: "missed"}
    assert e.confusions == [(3, 1)]


def test_classify_image_agrees_with_threshold_metrics():
    """Число TP и FP совпадает с precision_recall_at_threshold на том же пороге."""
    from src.eval.error_analysis import classify_image

    pred, gt = _random_detection_set(n_images=20)
    m = match_detections(pred, gt, iou_threshold=0.5)
    ref = precision_recall_at_threshold(m, 0.4)
    tp = fp = 0
    for img, (gb, gl) in gt.items():
        e = classify_image(*pred[img], gb, gl, score_threshold=0.4)
        tp, fp = tp + len(e.tp), fp + len(e.fp)
    assert (tp, fp) == (ref["tp"], ref["fp"])


def test_classify_image_without_ground_truth():
    from src.eval.error_analysis import classify_image

    e = classify_image(np.array([[0, 0, 5, 5]]), np.array([0.9]), np.array([0]),
                       np.zeros((0, 4)), np.zeros(0, dtype=int), score_threshold=0.3)
    assert e.fp == [(0, "bkg")] and e.fn == []


def test_dev_split_keeps_sequences_whole_and_uses_train(tmp_path):
    """dev набирается целыми видеозаписями из train; test не делит с ним ни
    одной записи; load_domain отдаёт непересекающиеся dev и test."""
    import json

    from src.data.acdc import load_domain, make_dev_split

    root = tmp_path / "ACDC"
    for split, seqs in (("train", 6), ("val", 2)):
        names = [f"fog/{split}/S{split}{q}/S{split}{q}_frame_{i:06d}_rgb_anon.png"
                 for q in range(seqs) for i in range(5)]
        for n in names:
            (root / "rgb_anon" / n).parent.mkdir(parents=True, exist_ok=True)
            (root / "rgb_anon" / n).touch()
        _write_coco(root / "gt_detection" / "fog" / f"instancesonly_fog_{split}_gt_detection.json",
                    names)

    cfg = {"data_root": str(root), "splits": ["train", "val"], "domains": ["fog"],
           "split_file": str(tmp_path / "split.json"), "eval_subset": "test"}
    full = load_domain(cfg, "fog", subset="all")
    assert len(full) == 40 and {s.split for s in full} == {"train", "val"}

    dev = make_dev_split(full, per_domain=8, seed=0)
    assert len(dev) == 10                            # две записи по 5 кадров, а не 8 кадров
    seq = {s.image_id: s.sequence for s in full}
    split = {s.image_id: s.split for s in full}
    assert all(split[i] == "train" for i in dev)

    (tmp_path / "split.json").write_text(json.dumps({"dev": {"fog": dev}}))
    d, t = load_domain(cfg, "fog", "dev"), load_domain(cfg, "fog")
    assert len(d) + len(t) == 40 and t.subset == "test"
    assert not {seq[s.image_id] for s in d} & {seq[s.image_id] for s in t}
