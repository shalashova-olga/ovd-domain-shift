#!/usr/bin/env bash
# Скачивание ACDC прямо на сервер, распаковка, фиксация разбиения и проверка.
#
# ACDC выдаётся только после регистрации на acdc.vision.ee.ethz.ch и одобрения
# заявки, поэтому ссылки берутся со страницы пакетов после входа: правой
# кнопкой по архиву -> «Копировать ссылку». Нужны rgb_anon_trainvaltest.zip
# и gt_detection_trainval.zip (не gt_trainval.zip — это сегментация).
#
# Запуск из корня репозитория:
#     bash scripts/download_acdc.sh "<ссылка на rgb_anon>" "<ссылка на gt_detection>"
#
# Если ссылки работают только в браузере с входом, выгрузите cookies сайта в
# формате Netscape (расширение «Get cookies.txt») и передайте файл:
#     COOKIES=cookies.txt bash scripts/download_acdc.sh <ссылки...>
#
# Скачивание докачивается при обрыве (wget -c): повторный запуск продолжит.

set -euo pipefail
cd "$(dirname "$0")/.."

if [ $# -eq 0 ]; then
    sed -n '2,16p' "$0"
    exit 1
fi

DL=data/downloads
mkdir -p "$DL" data/ACDC
WGET_OPTS=(-c --content-disposition --tries=20 --timeout=60 -P "$DL")
[ -n "${COOKIES:-}" ] && WGET_OPTS+=(--load-cookies "$COOKIES")

for url in "$@"; do
    echo "== $url"
    wget "${WGET_OPTS[@]}" "$url"
done

echo "== Проверка и распаковка"
for z in "$DL"/*.zip; do
    # Страница входа вместо архива — частая причина «битого zip»
    if ! unzip -tq "$z" >/dev/null 2>&1; then
        echo "[FAIL] $z не является целым zip-архивом (обрыв или ссылка вела на страницу входа)."
        echo "       Начало файла:"; head -c 300 "$z"; echo
        exit 1
    fi
    md5sum "$z" | tee -a "$DL/md5.txt"
    unzip -qo "$z" -d data/ACDC
    echo "[ok] $z"
done

# Архивы бывают с лишним верхним каталогом: поднимаем gt_detection и rgb_anon
for d in gt_detection rgb_anon; do
    if [ ! -d "data/ACDC/$d" ]; then
        found=$(find data/ACDC -mindepth 2 -maxdepth 3 -type d -name "$d" | head -1)
        [ -n "$found" ] && mv "$found" "data/ACDC/$d" && echo "поднят $found -> data/ACDC/$d"
    fi
done

echo "== Содержимое data/ACDC: $(ls data/ACDC | tr '\n' ' ')"
echo "md5 архивов записаны в $DL/md5.txt — сверьте с сайтом, если он их публикует,"
echo "и внесите в паспорт данных."

python scripts/make_split.py
python scripts/check_data.py
