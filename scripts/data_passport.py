"""
Паспорт данных для отчёта (задание 2): источник, лицензия, единица наблюдения,
классы, объём, пропуски, дубликаты, дисбаланс, разбиение и проверка утечки.

Всё считается по фактически загруженной разметке — числа в отчёте совпадают с
тем, на чём считались метрики.

Запуск:
    python scripts/data_passport.py --config configs/experiment_quick.yaml
Результат: results/data_passport.md
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.acdc import load_domain  # noqa: E402


def md_table(header: list[str], rows: list[list]) -> str:
    lines = ["| " + " | ".join(header) + " |", "|" + "---|" * len(header)]
    lines += ["| " + " | ".join(str(v) for v in r) + " |" for r in rows]
    return "\n".join(lines)


def main() -> int:
    ap = argparse.ArgumentParser(description="Паспорт данных")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--out", default="results/data_passport.md")
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))

    split_path = Path(cfg["split_file"])
    dev = json.loads(split_path.read_text(encoding="utf-8"))["dev"] if split_path.exists() else {}

    domains, classes = {}, None
    for d in cfg["domains"]:
        try:
            domains[d] = load_domain(cfg, d, subset="all")
            classes = domains[d].classes
        except FileNotFoundError as e:
            print(f"[warn] домен '{d}' пропущен: {e}")
    if not domains:
        print("Нет данных — паспорт не построен.")
        return 1

    vol, cls_rows, size_rows, quality = [], [], [], []
    total_cls = Counter()
    for d, ds in domains.items():
        st = ds.stats()
        dev_ids = set(dev.get(d, []))
        n_dev = sum(1 for s in ds if s.image_id in dev_ids)
        parts = Counter(s.split for s in ds)
        vol.append([d, st["n_images"], ", ".join(f"{k} {v}" for k, v in sorted(parts.items())),
                    st["n_sequences"], st["n_objects"], f"{st['objects_per_image']:.1f}",
                    n_dev, st["n_images"] - n_dev])
        cls_rows.append([d] + [st["per_class"][c] for c in classes])
        total_cls.update(st["per_class"])
        sz = st["size_distribution"]
        n = max(st["n_objects"], 1)
        size_rows.append([d, f"{sz['small'] / n:.0%}", f"{sz['medium'] / n:.0%}",
                          f"{sz['large'] / n:.0%}", f"{st['area_median']:.0f}"])

        missing = sum(1 for s in ds if not s.image_path.exists())
        dup = len(ds) - len({s.image_id for s in ds})
        dev_seq = {s.sequence for s in ds if s.image_id in dev_ids}
        test_seq = {s.sequence for s in ds if s.image_id not in dev_ids}
        quality.append([d, st["n_empty_images"], missing, dup, len(dev_seq & test_seq)])

    total = sum(total_cls.values()) or 1
    rare = [c for c in classes if total_cls[c] / total < 0.01]

    if cfg.get("dataset") == "bdd100k":
        rules = "; ".join(f"{d} — погода {', '.join(r['weather'])}, время {', '.join(r['timeofday'])}"
                          for d, r in cfg["bdd_domains"].items())
        head = [
            "| Источник | BDD100K: A Diverse Driving Dataset for Heterogeneous Multitask Learning "
            "(Yu et al., CVPR 2020), val-часть; открытая копия HuggingFace dgural/bdd100k |",
            "| Лицензия | BDD100K License: использование в образовательных и исследовательских "
            "целях без коммерческого применения |",
            "| Единица наблюдения | Кадр 1280×720 с видеорегистратора и ручные боксы объектов на нём |",
            "| Части разметки | val (10 000 кадров, по одному кадру из видеозаписи) |",
            f"| Классы | {', '.join(classes)} (pedestrian → person; traffic light/sign, "
            "other vehicle, trailer, other person исключены) |",
            f"| Домены | по меткам погоды и времени суток: {rules}. Подвыборка до "
            f"{cfg.get('max_images_per_domain')} кадров на домен, seed {cfg.get('seed', 0)} |",
            "| Временная структура | Каждый кадр val — из своей видеозаписи; почти одинаковых "
            "кадров между доменами нет |",
        ]
        limitation = ("- **Ограничение:** домены сняты в разных местах и разными машинами; "
                      "сдвиг погоды смешан со сменой местности. Тумана в val 13 кадров — "
                      "домен fog не используется. Метки погоды проставлены разметчиками BDD.")
    else:
        head = [
            "| Источник | ACDC: The Adverse Conditions Dataset with Correspondences "
            "(Sakaridis et al., ICCV 2021; версия 2024), acdc.vision.ee.ethz.ch |",
            "| Лицензия | Только некоммерческое исследовательское использование, доступ по "
            "регистрации; перераспространение данных запрещено условиями |",
            "| Единица наблюдения | Кадр 1920×1080 с дорожной камеры и боксы объектов на нём |",
            f"| Части разметки | {' + '.join(cfg['splits'])} (разметка test закрыта организаторами) |",
            f"| Классы | {', '.join(classes)} (классы Cityscapes с экземплярами) |",
            "| Домены | reference — ясный день на тех же маршрутах; fog, night, rain, snow |",
            "| Временная структура | Кадры — выборки из видеозаписей (GOPRxxxx); соседние кадры "
            "одной записи почти одинаковы, поэтому разбиение идёт по записям |",
        ]
        limitation = ("- **Ограничение:** кадры ясного дня сняты на тех же маршрутах, но в "
                      "другие дни; состав сцены у пары разный. Один город, одна камера.")
    L = [
        "# Паспорт данных", "",
        "| Поле | Значение |", "|---|---|",
        *head,
        f"| Отсечение | объекты площадью < {cfg.get('min_box_area', 16)} пикс.² и crowd-регионы "
        "исключены |",
        "",
        "## Объём", "",
        md_table(["домен", "кадров", "по частям", "видеозаписей", "объектов", "объектов/кадр",
                  "dev", "test"], vol), "",
        "## Классы (число объектов)", "",
        md_table(["домен"] + classes, cls_rows), "",
        f"Всего: " + ", ".join(f"{c} {total_cls[c]} ({total_cls[c] / total:.1%})"
                               for c in classes) + ".", "",
        f"**Дисбаланс:** классы с долей меньше 1 % — {', '.join(rare) or 'нет'}. "
        "Их AP опирается на единицы объектов и в выводах не используется; в mAP они "
        "входят, только если присутствуют в домене (иначе AP = nan, а не 0).", "",
        "## Размер объектов", "",
        "Границы COCO: мелкие < 32² пикс., крупные > 96² пикс.", "",
        md_table(["домен", "мелкие", "средние", "крупные", "медиана площади, пикс.²"],
                 size_rows), "",
        "## Качество и утечка", "",
        md_table(["домен", "кадров без объектов", "нет файла кадра", "дубликаты кадров",
                  "общих записей dev/test"], quality), "",
        "- **Пропуски:** кадры без объектов оставлены — на них измеряются ложные "
        "срабатывания. Отсутствующих файлов быть не должно.",
        "- **Дубликаты:** проверяются по имени кадра (уникально во всём ACDC).",
        "- **Утечка:** dev и test не должны делить ни одной видеозаписи — иначе почти "
        "одинаковые кадры попали бы в обе части." +
        (" Разбиение dev в этом прогоне пустое: промпт не подбирался, все кадры — test."
         if not any(dev.values()) else ""),
        limitation,
    ]
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text("\n".join(L) + "\n", encoding="utf-8")
    print(f"Паспорт данных: {out}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
