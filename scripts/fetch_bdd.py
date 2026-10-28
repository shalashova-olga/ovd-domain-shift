"""
Скачивание из копии BDD100K на HuggingFace только тех кадров, что нужны эксперименту.

Сначала скачивается разметка (samples.json), по ней тем же загрузчиком, что и
в эксперименте, отбираются кадры доменов, и качаются только они — ~1200
файлов вместо 10 000. Без входа в аккаунт HuggingFace ограничивает частоту
запросов (ошибка 429), поэтому потоков немного, а при отказе запрос
повторяется с нарастающей паузой. Уже скачанные кадры пропускаются, так что
повторный запуск докачивает.

Если задан HF_TOKEN (бесплатный токен с huggingface.co), лимит выше.

Если уже создан файл кадров адаптации (scripts/make_adapt_split.py), качаются
и они — ещё ~700 кадров ясного дня для few-shot и LoRA.

Запуск:
    python scripts/fetch_bdd.py --config configs/experiment_bdd.yaml
"""

from __future__ import annotations

import argparse
import concurrent.futures as cf
import os
import sys
import time
import urllib.error
import urllib.request
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.data.acdc import load_adaptation, load_domain  # noqa: E402

BASE = "https://huggingface.co/datasets/dgural/bdd100k/resolve/main/"


def fetch(url: str, dest: Path, tries: int = 8) -> bool:
    """Скачать файл с повтором при 429 и сетевых сбоях. True — если скачан сейчас."""
    if dest.exists() and dest.stat().st_size > 0:
        return False
    dest.parent.mkdir(parents=True, exist_ok=True)
    headers = {"User-Agent": "ovd-domain-shift"}
    if os.environ.get("HF_TOKEN"):
        headers["Authorization"] = f"Bearer {os.environ['HF_TOKEN']}"
    tmp = dest.with_suffix(dest.suffix + ".part")
    for attempt in range(tries):
        try:
            req = urllib.request.Request(url, headers=headers)
            with urllib.request.urlopen(req, timeout=60) as r, open(tmp, "wb") as f:
                while chunk := r.read(1 << 20):
                    f.write(chunk)
            tmp.replace(dest)
            return True
        except (urllib.error.URLError, TimeoutError, ConnectionError) as e:
            wait = min(5 * 2 ** attempt, 120)
            code = getattr(e, "code", "")
            print(f"  [{code or 'сеть'}] {dest.name}: повтор через {wait} с", flush=True)
            time.sleep(wait)
    raise RuntimeError(f"Не удалось скачать {url} за {tries} попыток")


def main() -> int:
    ap = argparse.ArgumentParser(description="Скачать нужные кадры BDD100K")
    ap.add_argument("--config", default="configs/experiment_bdd.yaml")
    ap.add_argument("--workers", type=int, default=4)
    args = ap.parse_args()
    cfg = yaml.safe_load(open(args.config, encoding="utf-8"))
    root = Path(cfg["data_root"])

    print("Разметка samples.json (71 МБ)...", flush=True)
    fetch(BASE + "samples.json", root / "samples.json")

    paths = []
    for d in cfg["domains"]:
        ds = load_domain(cfg, d, subset="all")
        paths += [s.image_path for s in ds]
        print(f"  {d}: {len(ds)} кадров", flush=True)
    if "adaptation" in cfg and Path(cfg["adaptation"]["file"]).exists():
        for part in ("train", "val"):
            ds = load_adaptation(cfg, part)
            paths += [s.image_path for s in ds]
            print(f"  адаптация {part}: {len(ds)} кадров", flush=True)
    todo = [p for p in paths if not (p.exists() and p.stat().st_size > 0)]
    print(f"Нужно кадров: {len(paths)}, уже есть: {len(paths) - len(todo)}, качаю: {len(todo)}",
          flush=True)

    done = 0
    with cf.ThreadPoolExecutor(args.workers) as ex:
        futures = [ex.submit(fetch, BASE + "data/" + p.name, p) for p in todo]
        for f in cf.as_completed(futures):
            f.result()
            done += 1
            if done % 100 == 0 or done == len(todo):
                print(f"  {done}/{len(todo)}", flush=True)
    print("Готово.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
