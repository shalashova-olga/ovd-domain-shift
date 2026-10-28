"""
Кадры для адаптации (few-shot и LoRA) — создаются один раз и больше не меняются.

Берутся из домена-источника (ясный день) вне тестовых кадров: train — для
обучения LoRA и как запас примеров few-shot, val — для контроля
переобучения LoRA. Порядок в train фиксирован: примеры few-shot выбираются
по нему, поэтому примеры 1-shot входят в 3-shot и 5-shot.

Файл configs/adapt_bdd.json кладётся в репозиторий вместе с результатами.

Запуск:
    python scripts/make_adapt_split.py --config configs/experiment_bdd.yaml
    python scripts/make_adapt_split.py --config ... --force   # только до обучения!
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.acdc import adaptation_pool  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Кадры для few-shot и LoRA")
    ap.add_argument("--config", default="configs/experiment_bdd.yaml")
    ap.add_argument("--force", action="store_true", help="перезаписать существующий файл")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    if "adaptation" not in cfg:
        print("В конфиге нет раздела adaptation — нечего делать.")
        return 0
    ad = cfg["adaptation"]
    path = Path(ad["file"])
    if path.exists() and not args.force:
        print(f"{path} уже существует — кадры адаптации зафиксированы, ничего не меняю.")
        return 0

    pool = adaptation_pool(cfg)
    need = ad["n_train"] + ad["n_val"]
    ids = sorted(s.image_id for s in pool)
    if len(ids) < need:
        print(f"[FAIL] в пуле {len(ids)} кадров, а нужно {need} — уменьшите n_train / n_val")
        return 1
    order = np.random.default_rng(cfg.get("seed", 0)).permutation(len(ids))
    chosen = [ids[i] for i in order[:need]]
    train, val = chosen[: ad["n_train"]], chosen[ad["n_train"]:]

    path.write_text(json.dumps({
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "domain": ad["domain"],
        "seed": cfg.get("seed", 0),
        "pool_size": len(ids),
        "rule": "кадры домена-источника вне теста; порядок train задаёт выбор примеров few-shot",
        "train": train,
        "val": val,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"{ad['domain']}: пул {len(ids)} кадров вне теста · train {len(train)} · val {len(val)}")
    print(f"Записано: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
