"""
Проверка режимов адаптации на случаях с известным ответом.

Ошибки здесь незаметны в итоговых таблицах: перепутанный номер класса при
обучении LoRA или сдвиг координат в примере few-shot дают модель, которая
просто чуть хуже, и это легко принять за свойство метода.

Запуск:  python -m pytest tests/ -v
"""

import json
import sys
from pathlib import Path

import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.acdc import ACDCDetection, Sample  # noqa: E402
from src.fewshot import box_shots, image_shots  # noqa: E402

CLASSES = ["person", "rider", "car"]


def _dataset(frames: list[list[tuple[int, float]]]) -> ACDCDetection:
    """Набор из кадров; кадр — список (класс, сторона квадратного бокса)."""
    ds = ACDCDetection.__new__(ACDCDetection)
    ds.classes, ds.domain, ds.subset = CLASSES, "reference", "adapt"
    ds.samples = []
    for i, objs in enumerate(frames):
        boxes = np.array([[10, 10, 10 + a, 10 + a] for _, a in objs], dtype=np.float32).reshape(-1, 4)
        labels = np.array([c for c, _ in objs], dtype=np.int32)
        ds.samples.append(Sample(f"f{i}", Path(f"f{i}.jpg"), "reference", boxes, labels, 1280, 720))
    return ds


# --------------------------------------------------------------------------- #
#                            Выбор примеров                                   #
# --------------------------------------------------------------------------- #

def test_box_shots_distinct_frames_largest_box_and_nested():
    ds = _dataset([
        [(0, 40), (0, 80), (2, 50)],     # person: берётся крупный (80)
        [(0, 20)],                        # слишком мелкий образец — пропуск
        [(0, 60), (2, 100)],
        [(2, 70)],
    ])
    s2 = box_shots(ds, 2, len(CLASSES))
    assert [s.image_id for s, _ in s2[0]] == ["f0", "f2"]
    assert s2[0][0][1][2] - s2[0][0][1][0] == 80
    assert s2[1] == []                    # rider не встретился — класс ищется по тексту
    assert [s.image_id for s, _ in s2[2]] == ["f0", "f2"]
    s1 = box_shots(ds, 1, len(CLASSES))
    ids = lambda shots: {c: [s.image_id for s, _ in v] for c, v in shots.items()}  # noqa: E731
    assert all(ids(s1)[c] == ids(s2)[c][:1] for c in range(len(CLASSES)))   # 1-shot входит в 2-shot


def test_image_shots_respects_object_count_and_order():
    ds = _dataset([[(2, 50)] * 3, [(2, 50)] * 5, [(2, 50)] * 20, [(0, 50)] * 4])
    assert [s.image_id for s in image_shots(ds, 2)] == ["f1", "f3"]
    with pytest.raises(RuntimeError):
        image_shots(ds, 3)


# --------------------------------------------------------------------------- #
#                         Ответ-пример для Qwen                               #
# --------------------------------------------------------------------------- #

def test_qwen_answer_scale_order_and_labels():
    """Координаты переводятся в сетку входа Qwen, объекты — сверху вниз,
    подписи — строки запроса, которыми модель должна ответить."""
    from src.models.adapters import QwenVLAdapter, qwen_answer

    boxes = np.array([[200, 300, 400, 500], [20, 40, 60, 80]], dtype=np.float32)
    labels = np.array([2, 0])
    ans = json.loads(qwen_answer(boxes, labels, CLASSES, (2.0, 2.0)))
    assert [a["label"] for a in ans] == ["person", "car"]
    assert ans[0]["bbox_2d"] == [10, 20, 30, 40]
    assert ans[1]["bbox_2d"] == [100, 150, 200, 250]

    # Ответ-пример разбирается тем же парсером, что и ответ модели, и
    # после обратного масштабирования совпадает с разметкой
    parser = QwenVLAdapter.__new__(QwenVLAdapter)
    got, scores, lbl, n_bad = parser._parse(json.dumps(ans), CLASSES, 2.0, 2.0)
    assert n_bad == 0 and list(lbl) == [0, 2]
    np.testing.assert_allclose(got, boxes[::-1])


def test_qwen3_relative_coordinates():
    """Qwen3-VL отвечает в шкале 0–1000 по обеим осям, независимо от сетки входа."""
    from PIL import Image

    from src.models.adapters import Qwen3VLAdapter

    a = Qwen3VLAdapter.__new__(Qwen3VLAdapter)
    assert "0 to 1000" in a.user_turn(Image.new("RGB", (8, 8)), CLASSES)["content"][1]["text"]
    sx, sy = a._answer_scale({"image_grid_thw": None}, Image.new("RGB", (1280, 720)))
    got, _, lbl, _ = a._parse('[{"bbox_2d": [500, 500, 1000, 1000], "label": "car"}]', CLASSES, sx, sy)
    np.testing.assert_allclose(got, [[640, 360, 1280, 720]])
    assert list(lbl) == [2]


# --------------------------------------------------------------------------- #
#                 Номера классов при обучении Grounding DINO                  #
# --------------------------------------------------------------------------- #

def test_grounding_label_groups_follow_prompt_order():
    """Модель делит текст запроса на группы токенов по точкам, и в labels
    при обучении номер класса — номер группы. Проверяется, что группа i —
    это токены i-й фразы, в том числе для фраз из нескольких токенов."""
    pytest.importorskip("torch")
    transformers = pytest.importorskip("transformers")
    from transformers.models.grounding_dino.modeling_grounding_dino import build_label_maps

    from src.models.adapters import build_grounding_text, phrase_token_indices

    import torch

    try:
        tok = transformers.AutoTokenizer.from_pretrained(
            "IDEA-Research/grounding-dino-tiny", local_files_only=True)
    except OSError:
        pytest.skip("нет локальной копии токенизатора Grounding DINO")

    prompts = ["person", "rider", "car", "a person riding a bicycle or motorcycle"]
    text, spans = build_grounding_text(prompts)
    enc = tok(text, return_offsets_mapping=True, return_tensors="pt")
    # от logits функции нужна только длина последней оси
    groups = build_label_maps(torch.zeros(1, 1, enc["input_ids"].shape[1]), enc["input_ids"])[0]
    expected = phrase_token_indices(enc["offset_mapping"][0].tolist(), spans)
    assert groups.shape[0] == len(prompts)
    for i, idx in enumerate(expected):
        assert sorted(np.nonzero(groups[i].numpy())[0].tolist()) == idx.tolist()
