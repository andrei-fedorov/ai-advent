#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/src"

# Необязательный аргумент выбирает, что запускать:
#   ./run.sh                  — текущее приложение (неделя 2 и дальше)
#   ./run.sh week1            — замороженный артефакт недели 1 (дни 1-5, вкладки)
#   ./run.sh faq-watch [...]  — сторож FAQ (день 18): MCP-сервер по Streamable HTTP;
#                               аргументы после faq-watch дописываются к командной
#                               строке из presets.py и перекрывают её
#                               (--interval 5, --port 8766, --db …)
APP="app.py"
case "${1:-}" in
    "")        APP="app.py" ;;
    week1)     APP="app_week1.py" ;;
    faq-watch) APP="" ;;
    *)
        echo "Неизвестный аргумент: $1. Допустимо: ./run.sh, ./run.sh week1 или ./run.sh faq-watch [аргументы сервера]" >&2
        exit 1
        ;;
esac

if [ ! -d ".venv" ]; then
    python3 -m venv .venv
fi

source .venv/bin/activate
pip install -q -r requirements.txt

if [ -z "$APP" ]; then
    # Командная строка сторожа живёт в одном месте — presets.FAQ_WATCH_ARGV.
    # Ключей у сервера нет, поэтому src/.env здесь не проверяется и серверу не
    # передаётся: окружение снимается до import presets — импорт подхватывает
    # src/.env (load_dotenv в mcp_client), и через execv ключи уехали бы в
    # процесс сторожа.
    exec python -c 'import os, sys; env = dict(os.environ); import presets; argv = presets.FAQ_WATCH_ARGV + sys.argv[1:]; os.execve(argv[0], argv, env)' "${@:2}"
fi

if [ ! -f ".env" ]; then
    echo "Внимание: src/.env не найден. Создайте его с DEEPSEEK_API_KEY=... перед запуском (см. README, шаг 2)." >&2
fi

python "$APP"
