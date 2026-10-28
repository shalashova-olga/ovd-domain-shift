#!/usr/bin/env bash
# Полный прогон: проверка данных, короткая проверка каждой модели, инференс, метрики.
#
# Запускать внутри tmux, чтобы прогон пережил отключение от сервера:
#     tmux new -s ovd
#     bash scripts/run_all.sh
#     (отсоединиться: Ctrl+B, затем D; вернуться: tmux attach -t ovd)
#
# Повторный запуск продолжает с места остановки: готовые журналы пропускаются.
#
# Без доступа к ACDC — BDD100K (2–3 часа на A100):
#     CONFIG=configs/experiment_bdd.yaml bash scripts/run_all.sh   (BDD100K, без ACDC)

set -uo pipefail
cd "$(dirname "$0")/.."
# Своё окружение (conda и т.п.) используется как есть, .venv — только если он есть
if [ -z "${CONDA_PREFIX:-}" ] && [ -f .venv/bin/activate ]; then
    source .venv/bin/activate
fi
echo "python: $(command -v python)"
CONFIG="${CONFIG:-configs/experiment.yaml}"

mkdir -p logs
LOG="logs/run_$(date +%Y%m%d_%H%M%S).log"
exec > >(tee -a "$LOG") 2>&1
echo "Журнал: $LOG"
echo "Конфиг: $CONFIG"

python scripts/make_split.py --config "$CONFIG" || exit 1          # если разбиение уже есть — не трогает
python scripts/check_data.py --config "$CONFIG" || { echo "Данные не готовы — прогон остановлен"; exit 1; }
python scripts/data_passport.py --config "$CONFIG"

# Быстрые модели первыми: если что-то не так с протоколом, это станет видно
# через минуты, а не после многих часов работы Qwen. Список можно задать
# снаружи, например только детекторы, пока подбирается промпт Qwen:
#     MODELS="yolo11l-coco grounding-dino-base" bash scripts/run_all.sh
DETECTORS="yolo11l-coco yolo-world-l owlvit-base owlv2-base grounding-dino-tiny grounding-dino-base owlv2-large"
MODELS="${MODELS:-$DETECTORS qwen2.5-vl-7b}"
echo "Модели: $MODELS"

# Модель, которая не запускается, исключается из прогона, а не останавливает
# его: остальные досчитаются, а о проблеме будет запись в журнале.
# Короткая проверка идёт на домене-источнике из конфига: набор доменов у
# ACDC и BDD100K разный (в BDD100K нет тумана).
SMOKE_DOMAIN=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['source_domain'])" "$CONFIG")
echo "== Проверка каждой модели на 3 кадрах ($SMOKE_DOMAIN) =="
OK_MODELS=""
FAILED=""
for m in $MODELS; do
    if python -m src.run_inference --config "$CONFIG" --model "$m" --domain "$SMOKE_DOMAIN" --limit 3 --overwrite; then
        OK_MODELS="$OK_MODELS $m"
    else
        echo "[!] Модель $m не запускается — пропускаю её"
        FAILED="$FAILED $m"
    fi
done
MODELS="$OK_MODELS"

echo "== Полный прогон: $MODELS =="
for m in $MODELS; do
    echo "---- $m: $(date +%H:%M:%S)"
    python -m src.run_inference --config "$CONFIG" --model "$m" || FAILED="$FAILED $m"
done

echo "== Метрики =="
python -m src.run_eval --config "$CONFIG"

echo "== Анализ ошибок, картинки, графики =="
python -m src.analyze --config "$CONFIG"

# Два архива: лёгкий — для отчёта (таблицы, анализ, логи), тяжёлый — сырые
# журналы предсказаний, чтобы пересчитать что угодно без повторного инференса.
STAMP=$(date +%Y%m%d_%H%M)
PRED_DIR=$(python -c "import yaml,sys; print(yaml.safe_load(open(sys.argv[1]))['output_dir'])" "$CONFIG")
# В архив попадает только то, что реально посчитано
PARTS=$(for p in results/tables results/analysis results/data_passport.md logs configs docs; do [ -e "$p" ] && echo "$p"; done)
tar -czf "results_${STAMP}_analysis.tar.gz" $PARTS
[ -d "$PRED_DIR" ] && tar -czf "results_${STAMP}_predictions.tar.gz" --exclude="_debug" "$PRED_DIR"
echo
ls -lh results_${STAMP}_*.tar.gz

if [ -n "$FAILED" ]; then
    echo "Модели с ошибками:$FAILED — перезапустите скрипт, готовое пропустится"
    exit 1
fi
echo "Готово. Начните с results/analysis/summary.md"
