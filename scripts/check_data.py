"""
Проверка данных перед прогоном: что нашлось в ACDC и все ли кадры на месте.

Запускается первым на сервере. Полный прогон идёт часами, и падение на
отсутствующем файле через три часа после старта — самая дорогая ошибка.
Скрипт ничего не меняет, только печатает отчёт.

Запуск:
    python scripts/check_data.py
    python scripts/check_data.py --data-root /path/to/ACDC
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.acdc import load_adaptation, load_domain  # noqa: E402


def _diagnose(root: Path) -> None:
    """Подсказать, где данные на самом деле: содержимое каталога и находки рядом."""
    for d in (root, root.parent):
        if d.is_dir():
            items = sorted(p.name + ("/" if p.is_dir() else "") for p in d.iterdir())
            print(f"\nСодержимое {d}: {', '.join(items[:30]) or '(пусто)'}")

    print("\nПоиск разметки и архивов ACDC (домашний каталог и текущий проект, глубина до 5):")
    found = []
    for base in {Path.home(), Path.cwd()}:
        for pattern in ("gt_detection", "*gt_detection*.zip", "rgb_anon*.zip", "rgb_anon"):
            for depth in range(1, 6):
                found += base.glob("/".join(["*"] * (depth - 1) + [pattern]))
    for p in sorted(set(found))[:20]:
        print(f"  {p}")
    if not found:
        print("  ничего не найдено — архивы, видимо, ещё не скачаны")
    print("\nЕсли данные лежат в другом месте: python scripts/check_data.py --data-root <путь>")
    print("и тот же путь пропишите в data_root в configs/experiment.yaml")


def main() -> int:
    ap = argparse.ArgumentParser(description="Проверка данных (ACDC или BDD100K)")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--data-root", default=None, help="переопределить data_root из конфига")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    root = Path(args.data_root or cfg["data_root"])
    print(f"data_root: {root.resolve()}")

    bdd = cfg.get("dataset") == "bdd100k"
    if bdd:
        if not (root / "samples.json").exists():
            print(f"[FAIL] нет {root / 'samples.json'} — скачайте: bash scripts/download_bdd.sh")
            return 1
        print("BDD100K val: разметка samples.json найдена")
    else:
        gt = root / "gt_detection"
        if not gt.is_dir():
            print(f"[FAIL] нет каталога {gt}")
            print("Ожидается распакованный gt_detection_trainval.zip и rgb_anon_trainvaltest.zip")
            _diagnose(root)
            return 1

        print("\nФайлы разметки:")
        for p in sorted(gt.rglob("*.json")):
            if any(sp in p.name for sp in cfg["splits"]):
                print(f"  {p.relative_to(root)}")

    # Ожидаемое по статье ACDC (Sakaridis et al.): размеченные train + val
    expected = {"reference": 1003, "fog": 500, "night": 506, "rain": 500, "snow": 500}
    ok = True
    print(f"\n{'домен':<10} {'кадров':>7} {'ожид.':>6} {'записей':>8} {'объектов':>9} "
          f"{'нет файла':>10}  по частям")
    loaded = {}
    for domain in cfg["domains"]:
        try:
            ds = load_domain(cfg | {"data_root": str(root)}, domain, subset="all")
        except FileNotFoundError as e:
            print(f"{domain:<10} [FAIL] {str(e).splitlines()[0]}")
            ok = False
            continue
        loaded[domain] = ds
        missing = [s.image_path for s in ds if not s.image_path.exists()]
        parts = Counter(s.split for s in ds)
        st = ds.stats()
        exp = (cfg.get("max_images_per_domain") or "—") if bdd else (
            expected.get(domain, "—") if cfg["splits"] == ["train", "val"] else "—")
        print(f"{domain:<10} {len(ds):>7} {exp:>6} {st['n_sequences']:>8} {st['n_objects']:>9} "
              f"{len(missing):>10}  " + ", ".join(f"{k} {v}" for k, v in sorted(parts.items())))
        if missing:
            ok = False
            print(f"           пример отсутствующего: {missing[0]}")
        dup = len(ds) - len({s.image_id for s in ds})
        if dup:
            ok = False
            print(f"           [FAIL] {dup} кадров встречаются в разметке дважды")

    print("\nРазбиение dev / test:")
    split_path = Path(cfg["split_file"])
    if not split_path.exists():
        print(f"  {split_path} ещё нет — создайте: python scripts/make_split.py")
    else:
        dev = json.loads(split_path.read_text(encoding="utf-8"))["dev"]
        for domain, ds in loaded.items():
            ids = set(dev.get(domain, []))
            dev_seq = {s.sequence for s in ds if s.image_id in ids}
            test_seq = {s.sequence for s in ds if s.image_id not in ids}
            shared = dev_seq & test_seq
            unknown = ids - {s.image_id for s in ds}
            status = "ok" if not shared and not unknown else "[FAIL]"
            print(f"  {domain:<10} dev {len(ids) - len(unknown):>4} · test {len(ds) - len(ids) + len(unknown):>5}"
                  f" · общих записей {len(shared)} · неизвестных кадров {len(unknown)}  {status}")
            ok &= status == "ok"

    if "adaptation" in cfg:
        print("\nКадры адаптации (few-shot и LoRA):")
        if not Path(cfg["adaptation"]["file"]).exists():
            print(f"  {cfg['adaptation']['file']} ещё нет — создайте: python scripts/make_adapt_split.py")
        else:
            test_ids = set()
            for ds in loaded.values():
                test_ids |= {s.image_id for s in ds}
            for part in ("train", "val"):
                try:
                    ds = load_adaptation(cfg | {"data_root": str(root)}, part)
                except RuntimeError as e:
                    print(f"  {part:<6} [FAIL] {str(e).splitlines()[0]}")
                    ok = False
                    continue
                missing = sum(not s.image_path.exists() for s in ds)
                shared = len({s.image_id for s in ds} & test_ids)
                status = "ok" if not missing and not shared else "[FAIL]"
                print(f"  {part:<6} {len(ds):>4} кадров · нет файла {missing} · "
                      f"общих с тестом {shared}  {status}")
                if missing:
                    print("         докачайте: bash scripts/download_bdd.sh")
                ok &= status == "ok"

    if ok:
        print("\n[OK] данные готовы к прогону")
        return 0
    print("\n[FAIL] пришлите этот вывод — по нему видно, что поправить")
    return 1


if __name__ == "__main__":
    sys.exit(main())
