"""
Анализ ошибок и сводный отчёт по готовым журналам инференса и таблицам метрик.

Запускается после run_eval и ничего не пересчитывает в моделях — только
читает results/predictions и results/tables. Результат в results/analysis:

    summary.md                 сводка: все ключевые таблицы и ссылки на картинки
    error_types.csv            типы ложных срабатываний и причины пропусков
    fn_breakdown.csv           пропуски по классу, размеру объекта и причине
    class_confusions.csv       какие классы с какими путаются
    vlm_answers.md             сырые ответы VLM: невалидные, обрезанные, примеры
    plots/*.png                тепловая карта mAP, перенос порога, диаграммы надёжности,
                               zero-shot против few-shot и LoRA
    gallery/<модель>/<домен>/  кадры с наибольшим числом ошибок, разметка ошибок на кадре
    compare/<домен>_*.jpg      самые трудные кадры домена: разметка и все модели рядом

Ошибки размечаются при рабочем пороге учителя — пороге, дающем precision 0.90
на домене-источнике. Это ровно та картина, которую увидел бы ученик, если бы
псевдоразметку собирали по порогу, подобранному на ясном дне.

Запуск:
    python -m src.analyze --config configs/experiment.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import yaml  # noqa: E402
from matplotlib import font_manager  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from PIL import Image, ImageDraw, ImageFont  # noqa: E402

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.acdc import ACDCDetection, load_domains, predictions_root  # noqa: E402
from src.eval.detection_metrics import (  # noqa: E402
    AREA_RANGES,
    match_detections,
    threshold_for_target_precision,
)
from src.eval.error_analysis import FN_REASONS, FP_TYPES, classify_image  # noqa: E402
from src.run_eval import load_predictions  # noqa: E402

FP_RU = {"dup": "дубль", "cls": "не тот класс", "loc": "локализация",
         "bkg": "фон", "other": "прочее"}
FN_RU = {"low_score": "низкий скор", "wrong_class": "не тот класс", "missed": "не найден"}

# Цвета разметки ошибок на кадрах: семантические, а не категориальные
TP_COLOR, FP_COLOR, FN_COLOR = "#0ca30c", "#d03b3b", "#fab219"

# Категориальная палитра доменов, фиксированный порядок (проверена валидатором)
DOMAIN_SLOTS = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4"]
SURFACE, INK, INK_2, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#898781", "#e8e7e3"
SEQ_BLUE = ["#cde2fb", "#86b6ef", "#3987e5", "#256abf", "#104281"]


# --------------------------------------------------------------------------- #
#                               Разбор ошибок                                 #
# --------------------------------------------------------------------------- #

def operating_threshold(preds_src: dict | None, ds_src: ACDCDetection | None, ev: dict):
    """Порог под целевую precision на источнике; если недостижим — рабочий порог."""
    if preds_src is not None and ds_src is not None:
        m = match_detections(preds_src, ds_src.ground_truth(), ev["iou_threshold"])
        thr, _ = threshold_for_target_precision(m, ev["target_precision"])
        if thr < 1.0:
            return thr, f"P≥{ev['target_precision']} на источнике"
    return ev["working_threshold"], "рабочий (P90 недостижима)"


def size_bucket(box: np.ndarray) -> str:
    area = float((box[2] - box[0]) * (box[3] - box[1]))
    for name, (lo, hi) in AREA_RANGES.items():
        if lo <= area < hi:
            return name
    return "large"


def analyse_domain(preds: dict, ds: ACDCDetection, thr: float, iou: float):
    """Разобрать все кадры домена. Возвращает счётчики и разбор по кадрам."""
    counts = Counter()
    fn_rows, confusions, per_image = Counter(), Counter(), []
    empty = (np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int))
    for s in ds:
        p = preds.get(s.image_id, empty)
        e = classify_image(*p, s.boxes, s.labels, thr, iou)
        counts["tp"] += len(e.tp)
        counts.update(f"fp_{k}" for _, k in e.fp)
        counts.update(f"fn_{r}" for _, r in e.fn)
        for gi, reason in e.fn:
            fn_rows[(ds.classes[s.labels[gi]], size_bucket(s.boxes[gi]), reason)] += 1
        confusions.update((ds.classes[a], ds.classes[b]) for a, b in e.confusions)
        per_image.append((s, p, e))
    return counts, fn_rows, confusions, per_image


# --------------------------------------------------------------------------- #
#                               Картинки                                      #
# --------------------------------------------------------------------------- #

def _font(size: int) -> ImageFont.FreeTypeFont:
    # DejaVu идёт вместе с matplotlib и содержит кириллицу — на сервере может
    # не быть системных шрифтов
    return ImageFont.truetype(font_manager.findfont("DejaVu Sans"), size)


def _box(b) -> list[float]:
    """Бокс с упорядоченными углами: PIL падает на x2 < x1."""
    x1, y1, x2, y2 = (float(v) for v in b)
    return [min(x1, x2), min(y1, y2), max(x1, x2), max(y1, y2)]


def _label(d: ImageDraw.ImageDraw, xy, text: str, color: str, font, below: bool = False) -> None:
    """Подпись над точкой xy, или под ней при below=True — так подписи пропусков
    не слипаются с подписями ложных боксов, которые стоят над рамкой."""
    x, y = xy
    l, t, r, b = d.textbbox((x, y), text, font=font)
    y = y + 3 if below else max(0, y - (b - t) - 4)
    l, t, r, b = d.textbbox((x, y), text, font=font)
    d.rectangle((l - 2, t - 2, r + 2, b + 2), fill=color)
    d.text((x, y), text, fill="black" if color == FN_COLOR else "white", font=font)


def draw_errors(image: Image.Image, sample, pred, e, classes: list[str],
                title: str, width: int = 1280, legend: str | None = None) -> Image.Image:
    """Кадр с разметкой: зелёный — верно, красный — ложное (тип), жёлтый — пропуск."""
    scale = width / image.width
    im = image.resize((width, round(image.height * scale)))
    d = ImageDraw.Draw(im)
    font = _font(max(12, width // 95))
    lw = max(2, width // 500)
    p_boxes, p_scores, p_labels = pred

    for gi, reason in e.fn:
        box = _box(sample.boxes[gi] * scale)
        d.rectangle(box, outline=FN_COLOR, width=lw + 1)
        _label(d, (box[0], box[3]), f"{classes[sample.labels[gi]]}: {FN_RU[reason]}",
               FN_COLOR, font, below=True)
    for pi in e.tp:
        d.rectangle(_box(p_boxes[pi] * scale), outline=TP_COLOR, width=lw)
    for pi, kind in e.fp:
        box = _box(p_boxes[pi] * scale)
        d.rectangle(box, outline=FP_COLOR, width=lw)
        _label(d, box[:2], f"{classes[p_labels[pi]]} {p_scores[pi]:.2f} {FP_RU[kind]}",
               FP_COLOR, font)

    head_font = _font(max(14, width // 70))
    head_h = int(head_font.size * 3.2)
    canvas = Image.new("RGB", (im.width, im.height + head_h), SURFACE)
    canvas.paste(im, (0, head_h))
    hd = ImageDraw.Draw(canvas)
    hd.text((10, 6), title, fill=INK, font=head_font)
    legend = legend or (f"верно {len(e.tp)} (зелёный)  ·  ложных {len(e.fp)} (красный)  ·  "
                        f"пропусков {len(e.fn)} (жёлтый)")
    hd.text((10, 8 + head_font.size * 1.4), legend, fill=INK_2, font=_font(head_font.size - 3))
    return canvas


def compare_grid(tiles: list[Image.Image], cols: int = 3) -> Image.Image:
    w, h = tiles[0].size
    rows = -(-len(tiles) // cols)
    grid = Image.new("RGB", (cols * w + (cols - 1) * 6, rows * h + (rows - 1) * 6), "white")
    for i, t in enumerate(tiles):
        grid.paste(t.resize((w, h)), ((i % cols) * (w + 6), (i // cols) * (h + 6)))
    return grid


def draw_ground_truth(image: Image.Image, sample, classes, width: int) -> Image.Image:
    from src.eval.error_analysis import ImageErrors

    e = ImageErrors()
    tile = draw_errors(image, sample, (np.zeros((0, 4)), np.zeros(0), np.zeros(0, dtype=int)),
                       e, classes, f"разметка · {sample.image_id}", width,
                       legend=f"объектов разметки {len(sample.boxes)}")
    d = ImageDraw.Draw(tile)
    scale = width / image.width
    head = tile.height - round(image.height * scale)
    font = _font(max(12, width // 60))
    for box, lbl in zip(sample.boxes, sample.labels):
        b = _box(box * scale)
        b = [b[0], b[1] + head, b[2], b[3] + head]
        d.rectangle(b, outline=TP_COLOR, width=2)
        _label(d, b[:2], classes[lbl], TP_COLOR, font)
    return tile


# --------------------------------------------------------------------------- #
#                                Графики                                      #
# --------------------------------------------------------------------------- #

def _style(ax) -> None:
    ax.set_facecolor(SURFACE)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(MUTED)
    ax.tick_params(colors=INK_2, labelsize=9)
    ax.grid(color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def plot_map_heatmap(metrics: pd.DataFrame, deg: pd.DataFrame, domains, path: Path) -> None:
    pv = metrics.pivot(index="model", columns="domain", values="mAP@50").reindex(columns=domains)
    pv = pv.loc[pv.mean(axis=1).sort_values(ascending=False).index]
    drop = (deg.pivot(index="model", columns="domain", values="relative_drop_%")
            if len(deg) else pd.DataFrame())

    fig, ax = plt.subplots(figsize=(1.5 * len(domains) + 2.5, 0.55 * len(pv) + 1.4), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    cmap = LinearSegmentedColormap.from_list("seq", SEQ_BLUE)
    ax.imshow(pv.values, cmap=cmap, aspect="auto", vmin=0, vmax=max(np.nanmax(pv.values), 1e-6))
    for i, model in enumerate(pv.index):
        for j, dom in enumerate(pv.columns):
            v = pv.iloc[i, j]
            if np.isnan(v):
                continue
            txt = f"{v:.2f}"
            if dom in drop.columns and model in drop.index and not np.isnan(drop.loc[model, dom]):
                txt += f"\n−{drop.loc[model, dom]:.0f}%" if drop.loc[model, dom] >= 0 \
                    else f"\n+{-drop.loc[model, dom]:.0f}%"
            light = v > 0.55 * np.nanmax(pv.values)
            ax.text(j, i, txt, ha="center", va="center", fontsize=8,
                    color="white" if light else INK)
    ax.set_xticks(range(len(pv.columns)), pv.columns)
    ax.set_yticks(range(len(pv.index)), pv.index)
    ax.tick_params(length=0, colors=INK_2, labelsize=9)
    for s in ax.spines.values():
        s.set_visible(False)
    ax.set_title("mAP@50 по доменам и падение относительно ясного дня",
                 loc="left", color=INK, fontsize=11)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_threshold_transfer(transfer: pd.DataFrame, domains, target: float, path: Path) -> None:
    models = (transfer.groupby("model")["precision_transferred"].mean()
              .sort_values().index.tolist())
    fig, ax = plt.subplots(figsize=(7.5, 0.45 * len(models) + 1.6), dpi=150)
    fig.patch.set_facecolor(SURFACE)
    _style(ax)
    ax.grid(axis="y", visible=False)
    ax.axvline(target, color=MUTED, linewidth=1.2, linestyle="--")
    ax.text(target, len(models) - 0.4, f" цель {target:.2f}", color=INK_2, fontsize=8,
            va="bottom")
    for slot, dom in zip(DOMAIN_SLOTS, domains):
        part = transfer[transfer.domain == dom].set_index("model").reindex(models)
        if part["precision_transferred"].notna().any():
            ax.scatter(part["precision_transferred"], range(len(models)), s=42, color=slot,
                       edgecolor=SURFACE, linewidth=1.5, label=dom, zorder=3)
    ax.set_yticks(range(len(models)), models)
    ax.set_xlabel("фактическая precision при пороге, подобранном на ясном дне",
                  color=INK_2, fontsize=9)
    ax.set_title("Перенос порога: держится ли чистота псевдоразметки",
                 loc="left", color=INK, fontsize=11)
    ax.legend(frameon=False, fontsize=8, ncol=len(domains), loc="upper left",
              bbox_to_anchor=(0, -0.18 if len(models) > 4 else -0.3), labelcolor=INK_2)
    fig.tight_layout()
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_reliability(details: dict, models, domains, path: Path) -> None:
    cols = 4
    rows = -(-len(models) // cols)
    fig, axes = plt.subplots(rows, cols, figsize=(3.1 * cols, 3.0 * rows + 0.6), dpi=150,
                             sharex=True, sharey=True, squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    for ax, model in zip(axes.flat, models):
        _style(ax)
        ax.plot([0, 1], [0, 1], color=MUTED, linewidth=1, linestyle="--")
        for slot, dom in zip(DOMAIN_SLOTS, domains):
            r = details.get(f"{model}__{dom}", {}).get("reliability")
            if r and r["confidence"]:
                ax.plot(r["confidence"], r["accuracy"], color=slot, linewidth=2,
                        marker="o", markersize=4, markeredgecolor=SURFACE, label=dom)
        ax.set_title(model, fontsize=9, color=INK, loc="left")
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)
    for ax in list(axes.flat)[len(models):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(domains), frameon=False,
               fontsize=9, labelcolor=INK_2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("Диаграммы надёжности: confidence против доли верных боксов "
                 "(ниже диагонали — переуверенность)", x=0.01, ha="left", color=INK, fontsize=11)
    fig.supxlabel("confidence", color=INK_2, fontsize=9, y=0.075)
    fig.supylabel("доля верных", color=INK_2, fontsize=9)
    fig.tight_layout(rect=(0, 0.1, 1, 0.97))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


def plot_adaptation(adapt: pd.DataFrame, metrics: pd.DataFrame, domains, path: Path) -> None:
    """mAP@50 исходной модели и её режимов адаптации, по панели на модель."""
    bases = list(dict.fromkeys(adapt["base"]))
    cols = min(len(bases), 3)
    rows = -(-len(bases) // cols)
    # Шкала у каждой панели своя: mAP у Qwen на порядок ниже, чем у детекторов,
    # и на общей шкале его столбцы не видны
    fig, axes = plt.subplots(rows, cols, figsize=(4.2 * cols, 3.2 * rows + 0.7), dpi=150,
                             squeeze=False)
    fig.patch.set_facecolor(SURFACE)
    m = metrics.set_index(["model", "domain"])["mAP@50"]
    width = 0.8 / len(domains)
    for ax, base in zip(axes.flat, bases):
        _style(ax)
        ax.grid(axis="x", visible=False)
        variants = [base] + list(dict.fromkeys(adapt[adapt.base == base]["model"]))
        names = ["zero-shot"] + [v[len(base) + 1:] if v.startswith(base + "-") else v
                                 for v in variants[1:]]
        for j, (slot, dom) in enumerate(zip(DOMAIN_SLOTS, domains)):
            vals = [m.get((v, dom), np.nan) for v in variants]
            ax.bar(np.arange(len(variants)) + (j - (len(domains) - 1) / 2) * width, vals,
                   width * 0.92, color=slot, label=dom)
        ax.set_xticks(range(len(variants)), names)
        ax.set_title(base, fontsize=9, color=INK, loc="left")
    for ax in list(axes.flat)[len(bases):]:
        ax.axis("off")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=len(domains), frameon=False,
               fontsize=9, labelcolor=INK_2, bbox_to_anchor=(0.5, 0.0))
    fig.suptitle("mAP@50: zero-shot, few-shot (fs — число примеров) и LoRA",
                 x=0.01, ha="left", color=INK, fontsize=11)
    fig.tight_layout(rect=(0, 0.07, 1, 0.96))
    fig.savefig(path, facecolor=SURFACE)
    plt.close(fig)


# --------------------------------------------------------------------------- #
#                               Ответы VLM                                    #
# --------------------------------------------------------------------------- #

def read_raw_answers(path: Path) -> list[dict]:
    rows = []
    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "raw_answer" in row:
                rows.append({"image_id": row["image_id"], "answer": row["raw_answer"],
                             "n_unparseable": row["validity"].get("n_unparseable", 0),
                             "n_kept": row["validity"].get("n_kept", 0),
                             "latency_ms": row["latency_ms"]})
    return rows


def vlm_report(answers: dict[tuple[str, str], list[dict]], out: Path) -> pd.DataFrame:
    stats, lines = [], ["# Ответы VLM", ""]
    for (model, domain), rows in sorted(answers.items()):
        bad = [r for r in rows if r["n_unparseable"] > 0]
        cut = [r for r in rows if not r["answer"].rstrip().rstrip("`").rstrip().endswith("]")]
        stats.append({
            "model": model, "domain": domain, "ответов": len(rows),
            "с ошибкой разбора": len(bad), "обрезано (нет ']')": len(cut),
            "средняя длина, симв.": int(np.mean([len(r["answer"]) for r in rows])),
            "боксов на кадр": round(float(np.mean([r["n_kept"] for r in rows])), 1),
        })
        lines += [f"## {model} · {domain}", ""]
        for title, sample in (("Невалидные ответы", bad[:8]), ("Обрезанные ответы", cut[:4]),
                              ("Обычные ответы", [r for r in rows if r not in bad][:3])):
            if not sample:
                continue
            lines += [f"### {title}", ""]
            for r in sample:
                text = r["answer"] if len(r["answer"]) < 1500 else r["answer"][:1500] + " …"
                lines += [f"`{r['image_id']}` · {r['latency_ms'] / 1000:.1f} с", "",
                          "```", text, "```", ""]
    df = pd.DataFrame(stats)
    out.write_text("\n".join(lines[:2] + [md_table(df), ""] + lines[2:]), encoding="utf-8")
    return df


# --------------------------------------------------------------------------- #
#                                 Сводка                                      #
# --------------------------------------------------------------------------- #

def md_table(df: pd.DataFrame, index: bool = False, digits: int = 3) -> str:
    if df is None or len(df) == 0:
        return "_нет данных_"
    if index:
        df = df.reset_index()

    def fmt(v):
        if isinstance(v, (float, np.floating)):
            return "—" if np.isnan(v) else f"{v:.{digits}f}"
        return str(v)

    head = "| " + " | ".join(map(str, df.columns)) + " |"
    sep = "|" + "|".join("---" for _ in df.columns) + "|"
    body = ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join([head, sep, *body])


def _read(path: Path) -> pd.DataFrame | None:
    return pd.read_csv(path) if path.exists() else None


def write_summary(out: Path, tables: Path, cfg: dict, err: pd.DataFrame,
                  conf: pd.DataFrame, fn: pd.DataFrame, vlm: pd.DataFrame | None,
                  gallery_links: list[str], compare_links: list[str],
                  sizes: dict[str, int] | None = None, subset: str = "test") -> None:
    src, ev = cfg["source_domain"], cfg["evaluation"]

    def img(name: str) -> str:
        return f"![](plots/{name})" if (out / "plots" / name).exists() else ""

    L = ["# Сводка эксперимента", "",
         f"Сгенерировано `src/analyze.py`. Все числа — подвыборка **{subset}** разбиения "
         f"{'BDD100K' if cfg.get('dataset') == 'bdd100k' else 'ACDC'} "
         f"({' + '.join(cfg['splits'])}), формулировка запроса по умолчанию "
         f"(`{cfg['default_prompt_variant']}`). Кадров: "
         + ", ".join(f"{d} {n}" for d, n in (sizes or {}).items()) + ".", ""]

    rank = _read(tables / "composite_ranking.csv")
    if rank is not None:
        L += ["## Сводный рейтинг", "",
              "Усреднение по целевым доменам, веса заданы до эксперимента "
              "(`configs/experiment.yaml`). Пустой балл — модель не прошла отсев.", "",
              md_table(rank), ""]

    m = _read(tables / "metrics_by_domain.csv")
    if m is not None:
        m["mAP@50 [95% ДИ]"] = m.apply(
            lambda r: f"{r['mAP@50']:.3f} [{r['mAP50_ci_low']:.3f}; {r['mAP50_ci_high']:.3f}]",
            axis=1)
        pv = m.pivot(index="model", columns="domain", values="mAP@50 [95% ДИ]")
        pv = pv.reindex(columns=[d for d in cfg["domains"] if d in pv.columns])
        L += ["## Качество по доменам", "", img("map_heatmap.png"), "",
              md_table(pv, index=True), "",
              "Разница моделей внутри пересекающихся интервалов незначима.", ""]
        res = (m.groupby("model")[["latency_p50_ms", "latency_p95_ms", "peak_vram_mb",
                                   "invalid_rate"]].agg(["max"]).droplevel(1, axis=1))
        L += ["## Ресурсы и валидность выхода", "", md_table(res, index=True, digits=2), ""]

    tr = _read(tables / "threshold_transfer.csv")
    if tr is not None and len(tr):
        L += ["## Перенос порога с ясного дня", "",
              f"Порог подобран на `{src}` под precision {ev['target_precision']} и без "
              "изменений применён к остальным доменам. `precision_gap` — недобор чистоты "
              "против порога, подобранного по самому домену; `yield_ratio` — объём "
              "псевдоразметки относительно оракульного порога.", "",
              img("threshold_transfer.png"), "",
              md_table(tr[["model", "domain", "threshold_source", "precision_transferred",
                           "precision_gap", "yield_ratio"]]), ""]

    ts = _read(tables / "temperature_scaling.csv")
    if ts is not None and len(ts):
        L += ["## Калибровка", "", img("reliability.png"), "",
              "Temperature scaling подобран на источнике и перенесён на домены. Если "
              "улучшение на целевых доменах мало, переуверенность вызвана сдвигом домена, "
              "а не свойством модели.", "",
              md_table(ts), ""]

    L += ["## Типы ошибок", "",
          "Порог — рабочая точка учителя (precision 0.90 на источнике). Ложные: "
          "**дубль** — объект уже найден; **не тот класс** — бокс точный, класс другой; "
          "**локализация** — свой класс, IoU 0.1–0.5; **фон** — рядом нет объектов. "
          "Пропуски: **низкий скор** — модель нашла объект ниже порога (лечится "
          "калибровкой); **не тот класс**; **не найден** — не видела ни при каком пороге.", "",
          md_table(err), ""]
    if len(conf):
        top = conf.sort_values("count", ascending=False).head(15)
        L += ["### Частые путаницы классов", "", md_table(top), ""]
    if len(fn):
        sz = (fn.groupby(["domain", "size"])["count"].sum().unstack(fill_value=0)
              .reindex(columns=list(AREA_RANGES), fill_value=0))
        sz = sz.div(sz.sum(axis=1), axis=0)
        L += ["### Доля пропусков по размеру объекта", "",
              "Мелкие — меньше 32² пикселей, крупные — больше 96².", "",
              md_table(sz, index=True, digits=2), ""]

    ad = _read(tables / "adaptation.csv")
    if ad is not None and len(ad):
        cols = ["model", "domain", "mAP@50_base", "mAP@50", "delta_mAP50", "significant",
                "precision_transferred_base", "precision_transferred",
                "invalid_rate_base", "invalid_rate"]
        L += ["## Zero-shot, few-shot и LoRA", "",
              "Примеры few-shot и обучающие кадры LoRA — с ясного дня вне теста, "
              "целевые домены при адаптации не используются. `significant` — интервалы "
              "mAP@50 режима и исходной модели не пересекаются.", "",
              img("adaptation.png"), "",
              md_table(ad[[c for c in cols if c in ad.columns]]), ""]

    ps = _read(tables / "prompt_sensitivity_summary.csv")
    if ps is not None:
        L += ["## Чувствительность к формулировке запроса", "",
              "Если размах больше ширины доверительного интервала, ранжирование моделей "
              "при одной формулировке не доказано.", "",
              md_table(ps[["model", "domain", "min", "max", "range", "mAP50_ci_width",
                           "best_variant"]]), ""]
        rc = _read(tables / "prompt_rank_correlation.csv")
        if rc is not None:
            L += ["Ранговая корреляция моделей между формулировками (1 — порядок не меняется):",
                  "", md_table(rc.rename(columns={"variant": "вариант"}), digits=2), ""]

    if vlm is not None and len(vlm):
        L += ["## Ответы VLM", "", "Разбор с примерами — в [vlm_answers.md](vlm_answers.md).",
              "", md_table(vlm), ""]

    L += ["## Кадры", "",
          "Самые трудные кадры каждого домена — разметка и все модели рядом:", ""]
    L += [f"- [{Path(p).stem}]({p})" for p in compare_links]
    L += ["", "Кадры с наибольшим числом ошибок по каждой модели и домену:", ""]
    L += [f"- [{p}]({p})" for p in gallery_links]
    (out / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")


# --------------------------------------------------------------------------- #

def main() -> None:
    ap = argparse.ArgumentParser(description="Анализ ошибок и сводный отчёт")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--subset", choices=["dev", "test", "all"], default=None,
                    help="часть разбиения; по умолчанию eval_subset из конфига (test)")
    ap.add_argument("--tables", default=None, help="по умолчанию results/tables[_dev]")
    ap.add_argument("--out", default=None, help="по умолчанию results/analysis[_dev]")
    ap.add_argument("--top-k", type=int, default=8, help="кадров в галерее на модель и домен")
    ap.add_argument("--hard-frames", type=int, default=4, help="трудных кадров на домен")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    ev = cfg["evaluation"]
    subset = args.subset or cfg.get("eval_subset", "test")
    sfx = "_dev" if subset == "dev" else ""
    pred_root = predictions_root(cfg, subset)
    tables = Path(args.tables or f"results/tables{sfx}")
    out = Path(args.out or f"results/analysis{sfx}")
    (out / "plots").mkdir(parents=True, exist_ok=True)
    variant, src = cfg["default_prompt_variant"], cfg["source_domain"]

    datasets = load_domains(cfg, subset)
    domains = [d for d in cfg["domains"] if d in datasets]
    # Сетки трудных кадров и диаграммы надёжности — только по исходным
    # моделям, иначе с режимами адаптации в них по 18 картинок; режимы
    # сравниваются с исходными моделями отдельным графиком
    zero_shot = [k for k, v in cfg["models"].items() if not v.get("base")]

    err_rows, fn_rows, conf_rows = [], [], []
    answers: dict[tuple[str, str], list[dict]] = {}
    hard: dict[str, Counter] = {d: Counter() for d in domains}
    per_model_images: dict[tuple[str, str], dict] = {}
    gallery_links = []

    for model in cfg["models"]:
        paths = {d: pred_root / model / f"{d}__{variant}.jsonl" for d in domains}
        paths = {d: p for d, p in paths.items() if p.exists()}
        if not paths:
            continue
        preds = {d: load_predictions(p)[0] for d, p in paths.items()}
        thr, thr_src = operating_threshold(preds.get(src), datasets.get(src), ev)
        print(f"{model}: порог {thr:.2f} ({thr_src})", flush=True)

        for dom, p in preds.items():
            ds = datasets[dom]
            counts, fn_c, conf_c, per_image = analyse_domain(p, ds, thr, ev["iou_threshold"])
            n_fp = sum(counts[f"fp_{k}"] for k in FP_TYPES)
            n_fn = sum(counts[f"fn_{r}"] for r in FN_REASONS)
            err_rows.append({
                "model": model, "domain": dom, "порог": thr, "TP": counts["tp"],
                **{f"FP {FP_RU[k]}": counts[f"fp_{k}"] for k in FP_TYPES},
                **{f"FN {FN_RU[r]}": counts[f"fn_{r}"] for r in FN_REASONS},
                "precision": counts["tp"] / max(counts["tp"] + n_fp, 1),
                "recall": counts["tp"] / max(counts["tp"] + n_fn, 1),
            })
            fn_rows += [{"model": model, "domain": dom, "class": c, "size": s,
                         "reason": FN_RU[r], "count": n} for (c, s, r), n in fn_c.items()]
            conf_rows += [{"model": model, "domain": dom, "предсказан": a, "на самом деле": b,
                           "count": n} for (a, b), n in conf_c.items()]

            raw_rows = read_raw_answers(paths[dom])
            raw = {r["image_id"]: r["answer"] for r in raw_rows}
            if raw_rows:
                answers[(model, dom)] = raw_rows

            by_id = {s.image_id: (s, pr, e) for s, pr, e in per_image}
            per_model_images[(model, dom)] = {"by_id": by_id, "thr": thr}
            for s, _, e in per_image:
                hard[dom][s.image_id] += e.n_errors

            # Галерея: кадры с наибольшим числом ошибок
            gdir = out / "gallery" / model / dom
            gdir.mkdir(parents=True, exist_ok=True)
            worst = sorted(per_image, key=lambda t: (-t[2].n_errors, t[0].image_id))
            for rank, (s, pr, e) in enumerate(worst[: args.top_k], 1):
                if e.n_errors == 0 or not s.image_path.exists():
                    continue
                img = Image.open(s.image_path).convert("RGB")
                title = f"{model} · {dom} · {s.image_id} · порог {thr:.2f}"
                name = f"{rank:02d}_{s.image_id}"
                draw_errors(img, s, pr, e, ds.classes, title).save(
                    gdir / f"{name}.jpg", quality=85)
                if s.image_id in raw:
                    (gdir / f"{name}.txt").write_text(raw[s.image_id], encoding="utf-8")
            gallery_links.append(str(gdir.relative_to(out)) + "/")

    if not err_rows:
        print("Нет журналов инференса — анализировать нечего.")
        return

    err = pd.DataFrame(err_rows)
    fn = pd.DataFrame(fn_rows, columns=["model", "domain", "class", "size", "reason", "count"])
    conf = pd.DataFrame(conf_rows,
                        columns=["model", "domain", "предсказан", "на самом деле", "count"])
    err.to_csv(out / "error_types.csv", index=False, encoding="utf-8")
    fn.to_csv(out / "fn_breakdown.csv", index=False, encoding="utf-8")
    conf.to_csv(out / "class_confusions.csv", index=False, encoding="utf-8")

    # Трудные кадры: разметка и все модели на одном листе
    compare_links = []
    cdir = out / "compare"
    cdir.mkdir(exist_ok=True)
    for dom in domains:
        for rank, (img_id, _) in enumerate(hard[dom].most_common(args.hard_frames), 1):
            s = next(x for x in datasets[dom] if x.image_id == img_id)
            if not s.image_path.exists():
                continue
            img = Image.open(s.image_path).convert("RGB")
            tiles = [draw_ground_truth(img, s, datasets[dom].classes, 640)]
            for model in zero_shot:
                info = per_model_images.get((model, dom))
                if info and img_id in info["by_id"]:
                    _, pr, e = info["by_id"][img_id]
                    tiles.append(draw_errors(img, s, pr, e, datasets[dom].classes,
                                             f"{model} · порог {info['thr']:.2f}", 640))
            path = cdir / f"{dom}_{rank:02d}_{img_id}.jpg"
            compare_grid(tiles).save(path, quality=85)
            compare_links.append(str(path.relative_to(out)))

    vlm = vlm_report(answers, out / "vlm_answers.md") if answers else None

    # Графики по таблицам run_eval
    m = _read(tables / "metrics_by_domain.csv")
    deg = _read(tables / "degradation.csv")
    tr = _read(tables / "threshold_transfer.csv")
    if m is not None:
        plot_map_heatmap(m, deg if deg is not None else pd.DataFrame(), domains,
                         out / "plots" / "map_heatmap.png")
    if tr is not None and tr["precision_transferred"].notna().any():
        plot_threshold_transfer(tr, domains, ev["target_precision"],
                                out / "plots" / "threshold_transfer.png")
    details_path = tables / "details.json"
    if details_path.exists():
        details = json.loads(details_path.read_text(encoding="utf-8"))
        models = [mk for mk in zero_shot if any(k.startswith(mk + "__") for k in details)]
        plot_reliability(details, models, domains, out / "plots" / "reliability.png")

    ad = _read(tables / "adaptation.csv")
    if m is not None and ad is not None and len(ad):
        plot_adaptation(ad, m, domains, out / "plots" / "adaptation.png")

    write_summary(out, tables, cfg, err, conf, fn, vlm, gallery_links, compare_links,
                  {d: len(ds) for d, ds in datasets.items()}, subset)
    print(f"\nАнализ записан в {out}/ — начните с summary.md")


if __name__ == "__main__":
    main()
