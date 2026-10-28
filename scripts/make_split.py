"""
Разбиение dev / test — создаётся один раз до любых прогонов и больше не меняется.

dev — небольшая часть train (по умолчанию ~40 кадров на домен) для подбора
промпта VLM и примеров few-shot. test — всё остальное, на нём считаются все
метрики. Если подбирать промпт, глядя на ошибки на тех же кадрах, на которых
потом считается качество, результат VLM окажется завышен.

Деление идёт целыми видеозаписями (GOPRxxxx): соседние кадры одной записи
почти одинаковы, и кадр в dev при соседнем кадре в test — та же утечка.

Файл configs/split.json кладётся в репозиторий вместе с результатами.

Запуск:
    python scripts/make_split.py
    python scripts/make_split.py --force     # пересоздать (только до первых прогонов!)
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.acdc import load_domain, make_dev_split  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description="Фиксированное разбиение dev / test")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--force", action="store_true", help="перезаписать существующий файл")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    path = Path(cfg["split_file"])
    if path.exists() and not args.force:
        print(f"{path} уже существует — разбиение зафиксировано, ничего не меняю.")
        print("Пересоздавать его после прогонов нельзя: dev и test перемешаются.")
        return 0

    per_domain, seed = cfg.get("dev_per_domain", 40), cfg.get("seed", 0)
    dev, info = {}, {}
    for domain in cfg["domains"]:
        ds = load_domain(cfg, domain, subset="all")
        ids = make_dev_split(ds, per_domain, seed)
        dev[domain] = ids
        seqs = sorted({s.sequence for s in ds if s.image_id in set(ids)})
        info[domain] = {"n_dev": len(ids), "n_test": len(ds) - len(ids), "dev_sequences": seqs}
        print(f"{domain:<10} dev {len(ids):>4} кадров из {len(seqs):>2} записей · "
              f"test {len(ds) - len(ids):>5}")

    path.write_text(json.dumps({
        "created": time.strftime("%Y-%m-%dT%H:%M:%S"),
        "splits": cfg["splits"],
        "seed": seed,
        "dev_per_domain": per_domain,
        "rule": "dev — целые видеозаписи из train; test — все остальные кадры",
        "summary": info,
        "dev": dev,
    }, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\nЗаписано: {path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
