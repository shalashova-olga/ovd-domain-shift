#!/usr/bin/env bash
# Скачивание нужной части BDD100K val (открытая копия на HuggingFace) и проверка.
#
# Качаются только кадры, отобранные в эксперимент (~1200 файлов, ~70 МБ).
# Ручного одобрения не нужно. Скачивая, вы соглашаетесь с лицензией BDD100K
# (образовательное и исследовательское использование).
#
# Если HuggingFace отвечает 429 (слишком много запросов) — скрипт сам ждёт и
# повторяет; повторный запуск докачивает с места остановки.
#
# Запуск из корня репозитория:
#     bash scripts/download_bdd.sh

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=configs/experiment_bdd.yaml

python scripts/fetch_bdd.py --config "$CONFIG"
python scripts/make_split.py --config "$CONFIG"
python scripts/check_data.py --config "$CONFIG"
