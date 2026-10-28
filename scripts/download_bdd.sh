#!/usr/bin/env bash
# Скачивание BDD100K val (открытая копия на HuggingFace, ~700 МБ) и проверка.
#
# Ручного одобрения не нужно. Скачивая, вы соглашаетесь с лицензией BDD100K
# (образовательное и исследовательское использование).
#
# Запуск из корня репозитория:
#     bash scripts/download_bdd.sh

set -euo pipefail
cd "$(dirname "$0")/.."

CONFIG=configs/experiment_bdd.yaml

python - <<'EOF'
from huggingface_hub import snapshot_download

snapshot_download(repo_id="dgural/bdd100k", repo_type="dataset",
                  local_dir="data/bdd100k", max_workers=8)
EOF

echo "Кадров скачано: $(ls data/bdd100k/data | wc -l) (ожидается 10000)"
python scripts/make_split.py --config "$CONFIG"
python scripts/check_data.py --config "$CONFIG"
