"""
Расчёт всех метрик по журналам инференса и сборка итоговых таблиц.

Порядок намеренный: сначала качество, затем калибровка, затем перенос порога,
затем ресурсы, и только потом сводный балл. Сводный балл считается последним,
потому что его веса задаются до эксперимента и не должны подбираться под
полученные числа — иначе выбор лучшей модели перестаёт быть проверяемым.

Запуск:
    python -m src.run_eval --config configs/experiment.yaml
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.acdc import load_domains, predictions_root  # noqa: E402
from src.eval.calibration import (  # noqa: E402
    apply_temperature,
    detection_ece,
    fit_temperature,
    reliability_data,
    threshold_transfer_penalty,
)
from src.eval.detection_metrics import (  # noqa: E402
    bootstrap_ci,
    breakdown_by_area,
    breakdown_by_class,
    compute_map,
    match_detections,
    precision_recall_at_threshold,
    threshold_for_target_precision,
)


def load_predictions(path: Path) -> tuple[dict, dict, dict]:
    """Прочитать JSONL: (предсказания, агрегированная валидность, профиль ресурсов)."""
    preds, lat, vram = {}, [], []
    val_total = {
        "n_raw": 0, "n_unparseable": 0, "n_degenerate": 0,
        "n_out_of_bounds": 0, "n_clipped": 0, "n_unknown_class": 0, "n_kept": 0,
    }
    meta = {}

    with open(path, encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            if "_meta" in row:
                meta = row["_meta"]
                continue
            preds[row["image_id"]] = (
                np.array(row["boxes"], dtype=np.float32).reshape(-1, 4),
                np.array(row["scores"], dtype=np.float32),
                np.array(row["labels"], dtype=np.int32),
            )
            for k in val_total:
                val_total[k] += row["validity"].get(k, 0)
            lat.append(row["latency_ms"])
            vram.append(row["peak_vram_mb"])

    bad = (val_total["n_unparseable"] + val_total["n_degenerate"]
           + val_total["n_out_of_bounds"] + val_total["n_unknown_class"])
    resources = {
        "latency_p50_ms": float(np.percentile(lat, 50)) if lat else float("nan"),
        "latency_p95_ms": float(np.percentile(lat, 95)) if lat else float("nan"),
        "peak_vram_mb": float(np.max(vram)) if vram else float("nan"),
        "throughput_img_s": 1000.0 / float(np.mean(lat)) if lat else float("nan"),
    }
    validity = {
        **val_total,
        "invalid_rate": bad / max(val_total["n_raw"], 1),
        "meta": meta,
    }
    return preds, validity, resources


def evaluate_combination(
    preds: dict, dataset: ACDCDetection, cfg: dict
) -> tuple[dict, object]:
    """Все метрики качества и калибровки для одной пары (модель, домен)."""
    gt = dataset.ground_truth()
    classes = dataset.classes

    quality = compute_map(preds, gt, classes)
    m = match_detections(preds, gt, iou_threshold=cfg["iou_threshold"])

    work_thr = cfg["working_threshold"]
    pr = precision_recall_at_threshold(m, work_thr)

    # Порог под целевую точность: рабочая точка учителя
    thr_target, pr_target = threshold_for_target_precision(m, cfg["target_precision"])

    cal = detection_ece(m.scores, m.tp, n_bins=cfg["calibration_bins"], adaptive=True)

    point, lo, hi = bootstrap_ci(
        preds, gt, classes, n_boot=cfg["bootstrap_iterations"], seed=cfg["seed"]
    )

    row = {
        **quality,
        **{f"PR@{work_thr}_{k}": v for k, v in pr.items() if k != "threshold"},
        "threshold_for_P90": thr_target,
        "recall_at_P90": pr_target.get("recall", float("nan")),
        "n_boxes_at_P90": pr_target.get("n_kept", 0),
        **cal.as_row(),
        "mAP50_ci_low": lo,
        "mAP50_ci_high": hi,
        "mAP50_ci_width": hi - lo,
        "breakdown_class": breakdown_by_class(m, classes, work_thr),
        "breakdown_area": breakdown_by_area(m, work_thr),
        "reliability": reliability_data(cal),
    }
    return row, m


def composite_score(df: pd.DataFrame, weights: dict, gates: dict) -> pd.DataFrame:
    """Сводный балл по заранее заданным весам с отсевом по жёстким условиям.

    Логика отличается от типового «взвесить метрики качества»: для задачи
    автоматической разметки precision при рабочем пороге весит больше mAP,
    а калибровка входит через разрыв переноса порога, а не через голый D-ECE.
    Модель, не прошедшая отсев, получает NaN, а не низкий балл — её нельзя
    использовать как учителя ни при каком весе остальных метрик.
    """
    out = df.copy()

    gate_mask = pd.Series(True, index=out.index)
    reasons = pd.Series("", index=out.index)
    for metric, rule in gates.items():
        if metric not in out.columns:
            continue
        if rule["direction"] == "max":   # метрика не должна превышать порог
            bad = out[metric] > rule["value"]
        else:                            # метрика не должна быть ниже порога
            bad = out[metric] < rule["value"]
        gate_mask &= ~bad
        reasons[bad] += f"{metric}; "

    def _norm(col: str, direction: str) -> pd.Series:
        v = out[col].astype(float)
        valid = v[gate_mask]
        if len(valid) == 0 or valid.max() == valid.min():
            return pd.Series(0.5, index=out.index)
        n = (v - valid.min()) / (valid.max() - valid.min())
        return n if direction == "higher_better" else 1 - n

    score = pd.Series(0.0, index=out.index)
    for metric, spec in weights.items():
        if metric not in out.columns:
            print(f"[warn] метрика '{metric}' отсутствует, вес {spec['weight']} не учтён")
            continue
        score += spec["weight"] * _norm(metric, spec["direction"])

    out["composite_score"] = score.where(gate_mask, np.nan)
    out["gate_failed"] = reasons.str.rstrip("; ")
    return out


def prompt_sensitivity(
    cfg: dict, datasets: dict, out_dir: Path, pred_root: Path | None = None
) -> None:
    """Разброс mAP@50 по вариантам запроса и устойчивость ранжирования моделей.

    Разброс сравнивается с шириной доверительного интервала mAP: если смена
    формулировки сдвигает mAP сильнее, чем шум выборки, ранжирование моделей
    при одной формулировке нельзя считать доказанным. Корреляция рангов между
    вариантами отвечает на тот же вопрос для всего набора моделей сразу.

    Альтернативные варианты гоняются на подвыборке кадров, поэтому все
    варианты, включая основной, сравниваются на общих кадрах, и интервал
    считается на них же — иначе размах и интервал были бы о разных выборках.
    """
    prompt_cfg = yaml.safe_load(open(cfg["prompts_file"], encoding="utf-8"))
    pred_root = pred_root or Path(cfg["output_dir"])
    ev = cfg.get("evaluation", {})

    rows, ci_rows = [], {}
    for model_key in cfg["models"]:
        for domain in cfg["prompt_sensitivity_domains"]:
            if domain not in datasets:
                continue
            runs = {}
            for variant in prompt_cfg["variants"]:
                path = pred_root / model_key / f"{domain}__{variant}.jsonl"
                if path.exists():
                    runs[variant] = load_predictions(path)
            if len(runs) < 2:
                continue
            ds = datasets[domain]
            common = set.intersection(*(set(p) for p, _, _ in runs.values()))
            gt = {k: v for k, v in ds.ground_truth().items() if k in common}
            for variant, (preds, validity, _) in runs.items():
                rows.append({
                    "model": model_key,
                    "domain": domain,
                    "variant": variant,
                    "n_images": len(gt),
                    "mAP@50": compute_map(preds, gt, ds.classes)["mAP@50"],
                    "invalid_rate": validity["invalid_rate"],
                })
            base = runs.get(cfg.get("default_prompt_variant"), next(iter(runs.values())))[0]
            _, lo, hi = bootstrap_ci(base, gt, ds.classes,
                                     n_boot=ev.get("bootstrap_iterations", 1000),
                                     seed=ev.get("seed", 0))
            ci_rows[(model_key, domain)] = hi - lo

    if not rows:
        print("[skip] нет прогонов с альтернативными формулировками запроса")
        return

    sens = pd.DataFrame(rows)
    sens.to_csv(out_dir / "prompt_sensitivity.csv", index=False, encoding="utf-8")

    g = sens.groupby(["model", "domain"])["mAP@50"]
    summary = g.agg(["min", "max", "std", "count"])
    summary["range"] = summary["max"] - summary["min"]
    summary["best_variant"] = sens.loc[g.idxmax().dropna().astype(int)].set_index(
        ["model", "domain"])["variant"]
    summary["mAP50_ci_width"] = pd.Series(ci_rows).reindex(summary.index)
    summary.to_csv(out_dir / "prompt_sensitivity_summary.csv", encoding="utf-8")

    # Спирмен как корреляция Пирсона рангов — без зависимости от scipy
    corr = []
    for domain, part in sens.groupby("domain"):
        ranks = part.pivot(index="model", columns="variant", values="mAP@50").rank(ascending=False)
        if len(ranks) < 3:
            continue
        c = ranks.corr()
        c.insert(0, "domain", domain)
        corr.append(c)
    if corr:
        pd.concat(corr).to_csv(out_dir / "prompt_rank_correlation.csv", encoding="utf-8")

    print("\nЧувствительность к формулировке запроса (размах mAP@50):")
    print(summary[["range", "mAP50_ci_width", "best_variant"]].to_string())


def adaptation_table(cfg: dict, df: pd.DataFrame, transfer: pd.DataFrame) -> pd.DataFrame:
    """Режимы адаптации рядом с исходной моделью: что дали few-shot и LoRA.

    Для каждого режима и домена — mAP@50 с интервалом, прирост к исходной
    модели и precision при пороге, перенесённом с источника. Прирост считается
    значимым, только если интервалы режима и исходной модели не пересекаются:
    обе оценки получены на одних и тех же кадрах, но интервалы независимы,
    поэтому это осторожная проверка.
    """
    rows = []
    by = df.set_index(["model", "domain"])
    tr = transfer.set_index(["model", "domain"]) if len(transfer) else None
    for key, mcfg in cfg["models"].items():
        base = mcfg.get("base")
        if not base:
            continue
        for domain in cfg["domains"]:
            if (key, domain) not in by.index or (base, domain) not in by.index:
                continue
            a, b = by.loc[(key, domain)], by.loc[(base, domain)]
            row = {
                "model": key, "base": base, "mode": mcfg["mode"],
                "shots": mcfg.get("shots"), "domain": domain,
                "mAP@50_base": b["mAP@50"], "mAP@50": a["mAP@50"],
                "delta_mAP50": a["mAP@50"] - b["mAP@50"],
                "significant": bool(a["mAP50_ci_low"] > b["mAP50_ci_high"]
                                    or a["mAP50_ci_high"] < b["mAP50_ci_low"]),
                "recall_at_P90_base": b["recall_at_P90"], "recall_at_P90": a["recall_at_P90"],
                "D-ECE_base": b["D-ECE"], "D-ECE": a["D-ECE"],
                "invalid_rate_base": b["invalid_rate"], "invalid_rate": a["invalid_rate"],
                "latency_p95_ms_base": b["latency_p95_ms"], "latency_p95_ms": a["latency_p95_ms"],
            }
            if tr is not None and (key, domain) in tr.index and (base, domain) in tr.index:
                row["precision_transferred_base"] = tr.loc[(base, domain), "precision_transferred"]
                row["precision_transferred"] = tr.loc[(key, domain), "precision_transferred"]
            rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser(description="Расчёт метрик и сборка таблиц")
    ap.add_argument("--config", default="configs/experiment.yaml")
    ap.add_argument("--out", default=None, help="по умолчанию results/tables[_dev]")
    ap.add_argument("--subset", choices=["dev", "test", "all"], default=None,
                    help="часть разбиения; по умолчанию eval_subset из конфига (test)")
    args = ap.parse_args()

    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    ev = cfg["evaluation"]
    subset = args.subset or cfg.get("eval_subset", "test")
    pred_root = predictions_root(cfg, subset)
    out_dir = Path(args.out or ("results/tables_dev" if subset == "dev" else "results/tables"))
    out_dir.mkdir(parents=True, exist_ok=True)

    datasets = load_domains(cfg, subset)
    print(f"Подвыборка: {subset}; кадров: "
          + ", ".join(f"{d} {len(ds)}" for d, ds in datasets.items()))

    rows, matches, details = [], {}, {}
    default_variant = cfg["default_prompt_variant"]

    for model_key in cfg["models"]:
        for domain in datasets:
            path = pred_root / model_key / f"{domain}__{default_variant}.jsonl"
            if not path.exists():
                print(f"[skip] нет файла {path}")
                continue

            preds, validity, resources = load_predictions(path)
            metrics, m = evaluate_combination(preds, datasets[domain], ev)
            matches[(model_key, domain)] = m

            rows.append({
                "model": model_key,
                "domain": domain,
                "mAP@50": metrics["mAP@50"],
                "mAP@50:95": metrics["mAP@50:95"],
                "mAP50_ci_low": metrics["mAP50_ci_low"],
                "mAP50_ci_high": metrics["mAP50_ci_high"],
                "precision": metrics[f"PR@{ev['working_threshold']}_precision"],
                "recall": metrics[f"PR@{ev['working_threshold']}_recall"],
                "f1": metrics[f"PR@{ev['working_threshold']}_f1"],
                "threshold_for_P90": metrics["threshold_for_P90"],
                "recall_at_P90": metrics["recall_at_P90"],
                "D-ECE": metrics["D-ECE"],
                "MCE": metrics["MCE"],
                "Brier": metrics["Brier"],
                "signed_gap": metrics["signed_gap"],
                "invalid_rate": validity["invalid_rate"],
                "latency_p50_ms": resources["latency_p50_ms"],
                "latency_p95_ms": resources["latency_p95_ms"],
                "peak_vram_mb": resources["peak_vram_mb"],
            })
            details[f"{model_key}__{domain}"] = {
                "breakdown_class": metrics["breakdown_class"],
                "breakdown_area": metrics["breakdown_area"],
                "reliability": metrics["reliability"],
                "validity": {k: v for k, v in validity.items() if k != "meta"},
            }

    if not rows:
        print("Нет данных для оценки. Сначала запустите src.run_inference.")
        return

    df = pd.DataFrame(rows)
    df.to_csv(out_dir / "metrics_by_domain.csv", index=False, encoding="utf-8")

    # ---- деградация относительно домена-источника ---- #
    src = cfg["source_domain"]
    base = df[df.domain == src].set_index("model")
    deg_rows = []
    for _, r in df[df.domain != src].iterrows():
        if r.model not in base.index:
            continue
        b = base.loc[r.model]
        deg_rows.append({
            "model": r.model,
            "domain": r.domain,
            "mAP@50": r["mAP@50"],
            "delta_mAP50": r["mAP@50"] - b["mAP@50"],
            "relative_drop_%": 100 * (b["mAP@50"] - r["mAP@50"]) / max(b["mAP@50"], 1e-9),
            "delta_precision": r["precision"] - b["precision"],
            "delta_D-ECE": r["D-ECE"] - b["D-ECE"],
            "delta_signed_gap": r["signed_gap"] - b["signed_gap"],
        })
    pd.DataFrame(deg_rows).to_csv(out_dir / "degradation.csv", index=False, encoding="utf-8")

    # ---- перенос рабочего порога источник -> целевой домен ---- #
    transfer = []
    for model_key in cfg["models"]:
        m_src = matches.get((model_key, src))
        if m_src is None:
            continue
        for domain in cfg["domains"]:
            if domain == src:
                continue
            m_tgt = matches.get((model_key, domain))
            if m_tgt is None:
                continue
            t = threshold_transfer_penalty(
                m_src.scores, m_src.tp, m_tgt.scores, m_tgt.tp,
                target_precision=ev["target_precision"],
            )
            transfer.append({"model": model_key, "domain": domain, **t})
    pd.DataFrame(transfer).to_csv(out_dir / "threshold_transfer.csv", index=False, encoding="utf-8")

    # ---- эффект посткалибровки, настроенной на источнике ---- #
    temp_rows = []
    for model_key in cfg["models"]:
        m_src = matches.get((model_key, src))
        if m_src is None or len(m_src.scores) == 0:
            continue
        T = fit_temperature(m_src.scores, m_src.tp)
        for domain in cfg["domains"]:
            m_t = matches.get((model_key, domain))
            if m_t is None or len(m_t.scores) == 0:
                continue
            before = detection_ece(m_t.scores, m_t.tp, ev["calibration_bins"])
            after = detection_ece(apply_temperature(m_t.scores, T), m_t.tp, ev["calibration_bins"])
            temp_rows.append({
                "model": model_key, "domain": domain, "temperature": T,
                "D-ECE_before": before.ece, "D-ECE_after": after.ece,
                "improvement": before.ece - after.ece,
            })
    pd.DataFrame(temp_rows).to_csv(out_dir / "temperature_scaling.csv", index=False, encoding="utf-8")

    # ---- чувствительность к формулировке запроса ---- #
    prompt_sensitivity(cfg, datasets, out_dir, pred_root)

    # ---- сводный балл: усреднение по целевым доменам ---- #
    target = df[df.domain != src].groupby("model").agg({
        "mAP@50": "mean", "precision": "mean", "recall_at_P90": "mean",
        "D-ECE": "mean", "invalid_rate": "max",
        "latency_p95_ms": "mean", "peak_vram_mb": "max",
    })
    scored = composite_score(target, ev["composite_weights"], ev["gates"])
    # Режимы адаптации участвуют в выборе наравне с исходными моделями:
    # правило выбирает лучшую модель или способ её применения
    scored.insert(0, "mode", [cfg["models"][k].get("mode", "zero-shot") for k in scored.index])
    scored.sort_values("composite_score", ascending=False).to_csv(
        out_dir / "composite_ranking.csv", encoding="utf-8"
    )

    # ---- режимы адаптации против исходных моделей ---- #
    adapt = adaptation_table(cfg, df, pd.DataFrame(transfer))
    if len(adapt):
        adapt.to_csv(out_dir / "adaptation.csv", index=False, encoding="utf-8")
        print("\nАдаптация (прирост mAP@50 к исходной модели):")
        print(adapt.pivot(index="model", columns="domain", values="delta_mAP50")
              .round(3).to_string())

    with open(out_dir / "details.json", "w", encoding="utf-8") as f:
        json.dump(details, f, ensure_ascii=False, indent=2)

    print(f"\nТаблицы записаны в {out_dir}")
    print("\nСводный рейтинг (усреднение по целевым доменам):")
    print(scored[["composite_score", "gate_failed"]].sort_values(
        "composite_score", ascending=False).to_string())


if __name__ == "__main__":
    main()
