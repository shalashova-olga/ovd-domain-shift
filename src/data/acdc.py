"""
Загрузка ACDC в формате задачи детекции.

ACDC поставляет instance-разметку в формате COCO, боксы выводятся из масок
экземпляров. Это важная деталь протокола: бокс из маски плотнее, чем
нарисованный вручную, поэтому абсолютные значения AP систематически ниже,
чем на датасетах с ручными боксами. Сравнивать с опубликованными числами
на COCO напрямую нельзя — сопоставимы только относительные изменения между
доменами внутри этой работы.

Домены ACDC: fog, night, rain, snow (неблагоприятные) и reference (ясный день,
тот же маршрут). Наличие reference-кадров и есть причина выбора ACDC: сдвиг
домена измеряется при почти неизменной географии и составе сцены, что
отделяет эффект условий съёмки от эффекта смены местности.
"""

from __future__ import annotations

import copy
import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

# Классы Cityscapes с instance-разметкой. train и rider оставлены осознанно:
# train почти отсутствует в ACDC, а rider не имеет аналога в COCO. Оба случая
# дают содержательные ограничения, которые фиксируются в отчёте.
ACDC_CLASSES = ["person", "rider", "car", "truck", "bus", "train", "motorcycle", "bicycle"]

DOMAINS = ["reference", "fog", "night", "rain", "snow"]


@dataclass
class Sample:
    image_id: str
    image_path: Path
    domain: str
    boxes: np.ndarray          # (N,4) xyxy абсолютные
    labels: np.ndarray         # (N,) индексы в ACDC_CLASSES
    width: int
    height: int
    split: str = ""            # train / val — из какого файла разметки
    sequence: str = ""         # видеозапись (GOPRxxxx): соседние кадры почти одинаковы


def sequence_of(file_name: str) -> str:
    """Идентификатор видеозаписи по имени кадра: GOPR0475_frame_000041_rgb_anon -> GOPR0475."""
    stem = Path(file_name).stem
    return stem.split("_frame")[0] if "_frame" in stem else stem


class ACDCDetection:
    """Набор ACDC как список Sample с фильтрацией по домену.

    min_box_area отсекает объекты площадью меньше порога. Такие экземпляры
    в ACDC встречаются (далёкие пешеходы в несколько пикселей), и ни одна
    модель их не находит. Оставленные, они занижают recall у всех моделей
    одинаково, но смещают оценку абсолютной пригодности учителя, создавая
    впечатление, что задача безнадёжна.
    """

    def __init__(
        self,
        root: str | Path,
        domain: str,
        split: str | list[str] = "val",
        min_box_area: float = 16.0,
        classes: list[str] | None = None,
    ) -> None:
        self.root = Path(root)
        self.domain = domain
        self.splits = [split] if isinstance(split, str) else list(split)
        self.split = "+".join(self.splits)
        self.subset = "all"
        self.min_box_area = min_box_area
        self.classes = classes or ACDC_CLASSES
        self.samples = self._load()

    def _load(self) -> list[Sample]:
        """Все запрошенные части разметки. Отсутствие любой из них — ошибка, а не
        молчаливый пропуск: иначе объём выборки тихо меняется между запусками."""
        samples = []
        for split in self.splits:
            ann_paths = self._annotation_paths(split)
            if not ann_paths:
                raise FileNotFoundError(
                    f"Не найдена разметка домена '{self.domain}' ({split}) в "
                    f"{self.root / 'gt_detection'}.\n"
                    f"Запустите scripts/check_data.py — он покажет, какие файлы есть."
                )
            for path in ann_paths:
                samples.extend(self._load_file(path, split))
        return samples

    def filtered(self, image_ids, subset: str | None = None) -> "ACDCDetection":
        """Копия набора только с указанными кадрами."""
        ids = set(image_ids)
        out = copy.copy(self)
        out.samples = [s for s in self.samples if s.image_id in ids]
        out.subset = subset or self.subset
        return out

    def _annotation_paths(self, split: str) -> list[Path]:
        """Файлы разметки домена.

        Официальный архив раскладывает разметку по условиям:
            gt_detection/fog/instancesonly_fog_val_gt_detection.json
        Кадры ясного дня сняты на маршрутах всех четырёх условий, поэтому домен
        reference собирается из нескольких файлов — по одному на условие.
        Шаблоны перебираются по порядку, побеждает первый, давший файлы.
        """
        gt = self.root / "gt_detection"
        d, s = self.domain, split
        if d == "reference":
            patterns = [
                f"*/instancesonly_*_{s}_ref_gt_detection.json",
                f"**/*{s}_ref*.json",
                f"**/*ref*{s}*.json",
            ]
        else:
            patterns = [
                f"{d}/instancesonly_{d}_{s}_gt_detection.json",
                f"instancesonly_{d}_{s}.json",
                f"**/instancesonly_{d}_{s}*.json",
            ]
        for pat in patterns:
            found = sorted(
                p for p in gt.glob(pat)
                if d == "reference" or "ref" not in p.name  # кадры ясного дня — отдельный домен
            )
            if found:
                return found
        return []

    def _resolve_image(self, file_name: str) -> Path:
        """Путь к кадру. В разных выпусках ACDC file_name указан относительно
        корня или относительно rgb_anon; в крайнем случае ищем по имени файла."""
        for cand in (self.root / file_name, self.root / "rgb_anon" / file_name):
            if cand.exists():
                return cand
        if self._image_index is None:
            self._image_index = {p.name: p for p in self.root.rglob("*.png")}
        return self._image_index.get(Path(file_name).name, self.root / file_name)

    _image_index: dict[str, Path] | None = None

    def _load_file(self, ann_path: Path, split: str) -> list[Sample]:
        with open(ann_path, encoding="utf-8") as f:
            coco = json.load(f)

        cat_id_to_idx: dict[int, int] = {}
        for cat in coco["categories"]:
            name = cat["name"].lower()
            if name in self.classes:
                cat_id_to_idx[cat["id"]] = self.classes.index(name)

        images = {img["id"]: img for img in coco["images"]}
        by_image: dict[int, list[dict]] = {}
        for ann in coco["annotations"]:
            if ann.get("iscrowd", 0) == 1:
                continue  # crowd-регионы не матчатся по COCO-протоколу
            if ann["category_id"] not in cat_id_to_idx:
                continue
            if ann.get("area", 0) < self.min_box_area:
                continue
            by_image.setdefault(ann["image_id"], []).append(ann)

        samples = []
        for img_id, img in images.items():
            anns = by_image.get(img_id, [])
            if anns:
                boxes = np.array(
                    [[a["bbox"][0], a["bbox"][1],
                      a["bbox"][0] + a["bbox"][2], a["bbox"][1] + a["bbox"][3]]
                     for a in anns], dtype=np.float32
                )
                labels = np.array(
                    [cat_id_to_idx[a["category_id"]] for a in anns], dtype=np.int32
                )
            else:
                boxes = np.zeros((0, 4), dtype=np.float32)
                labels = np.zeros(0, dtype=np.int32)

            samples.append(
                Sample(
                    # Имя файла уникально во всём ACDC, а числовой id — только
                    # внутри одного JSON, и домен reference склеен из четырёх
                    image_id=Path(img["file_name"]).stem,
                    image_path=self._resolve_image(img["file_name"]),
                    domain=self.domain,
                    boxes=boxes,
                    labels=labels,
                    width=img["width"],
                    height=img["height"],
                    split=split,
                    sequence=sequence_of(img["file_name"]),
                )
            )
        return samples

    def __len__(self) -> int:
        return len(self.samples)

    def __iter__(self):
        return iter(self.samples)

    def ground_truth(self) -> dict[str, tuple[np.ndarray, np.ndarray]]:
        """Разметка в формате, который ждут метрики."""
        return {s.image_id: (s.boxes, s.labels) for s in self.samples}

    def stats(self) -> dict:
        """Описательная статистика — основа паспорта данных в отчёте."""
        n_obj = sum(len(s.boxes) for s in self.samples)
        per_class = {c: 0 for c in self.classes}
        areas = []
        for s in self.samples:
            for lbl, box in zip(s.labels, s.boxes):
                per_class[self.classes[lbl]] += 1
                areas.append(float((box[2] - box[0]) * (box[3] - box[1])))

        areas_arr = np.array(areas) if areas else np.zeros(1)
        small = int((areas_arr < 32**2).sum())
        medium = int(((areas_arr >= 32**2) & (areas_arr < 96**2)).sum())
        large = int((areas_arr >= 96**2).sum())

        return {
            "domain": self.domain,
            "split": self.split,
            "subset": self.subset,
            "n_images": len(self.samples),
            "n_sequences": len({s.sequence for s in self.samples}),
            "n_objects": n_obj,
            "objects_per_image": n_obj / max(len(self.samples), 1),
            "n_empty_images": sum(1 for s in self.samples if len(s.boxes) == 0),
            "per_class": per_class,
            "area_median": float(np.median(areas_arr)),
            "size_distribution": {"small": small, "medium": medium, "large": large},
        }


# --------------------------------------------------------------------------- #
#                    Разбиение dev / test по видеозаписям                      #
# --------------------------------------------------------------------------- #

def make_dev_split(dataset: ACDCDetection, per_domain: int, seed: int) -> list[str]:
    """Выбрать кадры dev из train-части целыми видеозаписями.

    dev нужен для подбора промпта и примеров few-shot; всё остальное — test.
    Делить приходится по записям, а не по кадрам: соседние кадры одной записи
    почти одинаковы, и кадр в dev при его соседе в test — утечка, из-за
    которой подобранный промпт выглядел бы лучше, чем есть.
    """
    pool = [s for s in dataset if s.split == "train"] or list(dataset)
    by_seq: dict[str, list[str]] = {}
    for s in pool:
        by_seq.setdefault(s.sequence, []).append(s.image_id)
    rng = np.random.default_rng(seed)
    order = sorted(by_seq)
    rng.shuffle(order)
    dev: list[str] = []
    for seq in order:
        if len(dev) >= per_domain:
            break
        dev.extend(sorted(by_seq[seq]))
    return dev


def load_split_file(path: str | Path) -> dict:
    path = Path(path)
    if not path.exists():
        raise RuntimeError(
            f"Нет файла разбиения {path}. Создайте его один раз до любых прогонов:\n"
            f"    python scripts/make_split.py"
        )
    return json.loads(path.read_text(encoding="utf-8"))


def load_domain(cfg: dict, domain: str, subset: str | None = None) -> ACDCDetection:
    """Домен по конфигу эксперимента с фильтрацией dev / test / all.

    По умолчанию — eval_subset из конфига (test): все метрики считаются на
    кадрах, которые не участвовали в подборе промпта.
    """
    if cfg.get("dataset") == "bdd100k":
        from .bdd import BDDDetection

        if domain not in cfg["bdd_domains"]:
            raise FileNotFoundError(f"Домен '{domain}' не описан в bdd_domains")
        ds = BDDDetection(cfg["data_root"], domain, cfg["bdd_domains"][domain],
                          max_images=cfg.get("max_images_per_domain"),
                          seed=cfg.get("seed", 0),
                          min_box_area=cfg.get("min_box_area", 16.0))
    else:
        ds = ACDCDetection(cfg["data_root"], domain=domain, split=cfg["splits"],
                           min_box_area=cfg.get("min_box_area", 16.0))
    subset = subset or cfg.get("eval_subset", "all")
    if subset == "all":
        return ds
    dev = set(load_split_file(cfg["split_file"])["dev"].get(domain, []))
    keep = [s.image_id for s in ds if (s.image_id in dev) == (subset == "dev")]
    return ds.filtered(keep, subset)


def load_domains(cfg: dict, subset: str | None = None) -> dict[str, ACDCDetection]:
    """Все домены конфига. Отсутствующий домен пропускается с предупреждением,
    чтобы эксперимент можно было прогнать на частично скачанных данных."""
    out = {}
    for d in cfg["domains"]:
        try:
            out[d] = load_domain(cfg, d, subset)
        except FileNotFoundError as e:
            print(f"[warn] домен '{d}' пропущен: {e}")
    return out


# --------------------------------------------------------------------------- #
#               Размеченные кадры для адаптации (few-shot, LoRA)               #
# --------------------------------------------------------------------------- #

def adaptation_pool(cfg: dict) -> ACDCDetection:
    """Кадры домена-источника, не входящие в тест, — из них берутся данные адаптации.

    У BDD100K тест — подвыборка max_images_per_domain из всех кадров домена,
    поэтому пул собирается из полного домена без этого ограничения. У ACDC
    тестом служит всё, кроме dev, и пулом оказывается dev. Тестовые кадры
    исключаются по всем доменам, а не только по источнику, — на случай
    пересечения правил отбора доменов.
    """
    domain = cfg["adaptation"]["domain"]
    full = load_domain({**cfg, "max_images_per_domain": None}, domain, subset="all")
    test_ids: set[str] = set()
    for d in cfg["domains"]:
        test_ids |= {s.image_id for s in load_domain(cfg, d, subset=cfg.get("eval_subset", "test"))}
    return full.filtered([s.image_id for s in full if s.image_id not in test_ids], "adapt")


def load_adaptation(cfg: dict, part: str) -> ACDCDetection:
    """Часть данных адаптации: train (обучение LoRA и примеры few-shot) или val.

    Порядок кадров берётся из файла разбиения, а не из датасета: примеры
    few-shot выбираются по этому порядку, и он не должен зависеть от того,
    как разметка отсортирована на диске.
    """
    path = Path(cfg["adaptation"]["file"])
    if not path.exists():
        raise RuntimeError(
            f"Нет файла {path}. Создайте его один раз до обучения и few-shot:\n"
            f"    python scripts/make_adapt_split.py --config <конфиг>"
        )
    ids = json.loads(path.read_text(encoding="utf-8"))[part]
    pool = adaptation_pool(cfg)
    by_id = {s.image_id: s for s in pool}
    missing = [i for i in ids if i not in by_id]
    if missing:
        raise RuntimeError(
            f"{len(missing)} кадров адаптации ({part}) нет в пуле — например {missing[0]}.\n"
            f"Файл {cfg['adaptation']['file']} создан для других данных или теста."
        )
    out = pool.filtered(ids, part)
    out.samples = [by_id[i] for i in ids]
    return out


def predictions_root(cfg: dict, subset: str | None = None) -> Path:
    """Журналы dev пишутся отдельно от test, чтобы их нельзя было смешать."""
    root = Path(cfg["output_dir"])
    return root / "_dev" if (subset or cfg.get("eval_subset")) == "dev" else root
