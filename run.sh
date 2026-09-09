#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/src"

# Необязательный аргумент выбирает, какое приложение запускать:
#   ./run.sh        — текущее приложение (неделя 2 и дальше)
#   ./run.sh week1  — замороженный артефакт недели 1 (дни 1-5, вкладки)
APP="app.py"
case "${1:-}" in
    "")      APP="app.py" ;;
    week1)   APP="app_week1.py" ;;
    *)
        echo "Неизвестный аргумент: $1. Допустимо: ./run.sh или ./run.sh week1" >&2
        exit 1
        ;;
esac

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install -q -r requirements.txt

if [ ! -f ".env" ]; then
    echo "Внимание: src/.env не найден. Создайте его с DEEPSEEK_API_KEY=... перед запуском (см. README, шаг 2)." >&2
fi

python "$APP"
