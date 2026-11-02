"""Выбор dense/MoE по конфигу без torch, скачивания или загрузки весов."""

import sys
from pathlib import Path
from types import ModuleType

import pytest
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.models.adapters import build_detector  # noqa: E402


@pytest.mark.parametrize(
    ("model_key", "expected_class", "expected_checkpoint"),
    [
        ("qwen3-vl-8b", "Qwen3VLForConditionalGeneration", "Qwen/Qwen3-VL-8B-Instruct"),
        (
            "qwen3-vl-30b-a3b",
            "Qwen3VLMoeForConditionalGeneration",
            "Qwen/Qwen3-VL-30B-A3B-Instruct",
        ),
    ],
)
def test_qwen_config_loads_correct_architecture(
    monkeypatch, model_key, expected_class, expected_checkpoint
):
    cfg_path = Path(__file__).resolve().parents[1] / "configs/experiment_bdd.yaml"
    cfg = yaml.safe_load(cfg_path.read_text(encoding="utf-8"))["models"][model_key]
    calls = []

    class FakeModel:
        def eval(self):
            self.is_eval = True
            return self

    def loader(class_name):
        class Factory:
            @staticmethod
            def from_pretrained(checkpoint, **kwargs):
                calls.append((class_name, checkpoint, kwargs))
                return FakeModel()

        return Factory

    torch = ModuleType("torch")
    torch.bfloat16 = object()
    transformers = ModuleType("transformers")
    transformers.AutoProcessor = loader("processor")
    transformers.Qwen3VLForConditionalGeneration = loader("Qwen3VLForConditionalGeneration")
    transformers.Qwen3VLMoeForConditionalGeneration = loader("Qwen3VLMoeForConditionalGeneration")
    monkeypatch.setitem(sys.modules, "torch", torch)
    monkeypatch.setitem(sys.modules, "transformers", transformers)

    detector = build_detector(
        cfg["adapter"],
        ["person", "car"],
        device="cuda:0",
        checkpoint=cfg["checkpoint"],
        box_threshold=cfg["box_threshold"],
    )

    assert calls == [
        ("processor", expected_checkpoint, {}),
        (expected_class, expected_checkpoint, {"torch_dtype": torch.bfloat16, "device_map": "cuda:0"}),
    ]
    assert detector.model.is_eval
    assert detector.max_new_tokens == 2048
