#!/usr/bin/env bash
# Установка окружения на сервере с GPU и предзагрузка всех весов.
#
# Веса скачиваются заранее, а не по ходу прогона: если сеть пропадёт или
# HuggingFace ответит ошибкой на седьмой модели, полный прогон упадёт через
# несколько часов после старта.
#
# Запуск из корня репозитория:
#     bash scripts/setup_server.sh

set -euo pipefail
cd "$(dirname "$0")/.."

# Активное conda-окружение используется как есть, иначе создаётся .venv
if [ -n "${CONDA_PREFIX:-}" ]; then
    echo "Ставлю в conda-окружение: $CONDA_PREFIX"
else
    [ -d .venv ] || python3 -m venv .venv
    source .venv/bin/activate
fi

pip install --upgrade pip
pip install -r requirements.txt

echo "== GPU =="
python - <<'EOF'
import torch
assert torch.cuda.is_available(), "CUDA недоступна: проверьте драйвер и сборку torch"
p = torch.cuda.get_device_properties(0)
print(f"{p.name}, {p.total_memory / 1024**3:.0f} ГБ, torch {torch.__version__}, CUDA {torch.version.cuda}")
EOF

echo "== Тесты =="
python -m pytest tests/ -q

echo "== Предзагрузка весов =="
python - <<'EOF'
import yaml
from huggingface_hub import snapshot_download

cfg = yaml.safe_load(open("configs/experiment.yaml", encoding="utf-8"))
for key, m in cfg["models"].items():
    ckpt = m["checkpoint"]
    if ckpt.endswith(".pt"):
        from ultralytics import YOLO
        model = YOLO(ckpt)                       # скачивает веса в текущий каталог
        if m["adapter"] == "yolo-world":
            model.set_classes(["car"])           # подтягивает текстовый энкодер CLIP
    else:
        snapshot_download(ckpt)
    print(f"[ok] {key}: {ckpt}")
EOF

echo
echo "Готово. Дальше: python scripts/check_data.py"
