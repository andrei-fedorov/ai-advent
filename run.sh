#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "${BASH_SOURCE[0]}")/src"

# Необязательный аргумент выбирает, что запускать:
#   ./run.sh                  — текущее приложение (неделя 2 и дальше)
#   ./run.sh week1            — замороженный артефакт недели 1 (дни 1-5, вкладки)
#   ./run.sh faq-watch [...]  — ручной запуск сервера FAQ (дни 18-20); обычно его
#                               поднимает приложение. Аргументы после faq-watch
#                               дописываются к командной строке из presets.py и
#                               перекрывают её (--interval 5, --port 8766, --db …)
#   ./run.sh index [...]      — индекс правил (день 21): PDF из rag_sources/ →
#                               куски → эмбеддинги → src/data/rag/index.sqlite3;
#                               --probe — пробные вопросы по готовому индексу
#                               (с дня 23 скачивает и реранкер, ≈1,5 ГБ),
#                               --dump <каталог>, --db <путь>, --sources <каталог>
#   ./run.sh rag-eval [...]   — контрольные (К1-К10), уточняющие (У1-У3) вопросы и
#                               вопросы проверки ответа (Н1, Н2, Ф1) в режимах RAG,
#                               по умолчанию режим `полный`, проверка источников и
#                               цитат (день 24); отчёт Markdown (нужны ключ
#                               DeepSeek, индекс и реранкер в кэше): --modes
#                               простой,полный, --only К1,У1,Ф1, --threshold,
#                               --candidates, --top-k, --preset <имя>, --db <путь>,
#                               --out <файл.md>
#   ./run.sh rag-dialog [...] — два длинных сценария (А — 14 сообщений, Б — 13) на
#                               «Скользящем окне» в режимах «с памятью задачи» и
#                               «без памяти задачи» (день 25); отчёт Markdown в
#                               src/data/rag/dialog/ (нужны ключ DeepSeek, индекс и
#                               реранкер в кэше): --scenarios А,Б, --modes память,без,
#                               --strategy <имя>, --preset <имя>, --db <путь>,
#                               --out <файл.md>
#   ./run.sh local [...]      — три запроса к локальной модели (день 26): Л1 —
#                               простой, Л2 — вопрос К5 из общих знаний, Л3 — он же
#                               с полным RAG; отвечает модель, загруженная в LM
#                               Studio (lms load <id>, lms server start), ключ не
#                               нужен; отчёт Markdown в src/data/local/ (для Л3 —
#                               индекс и реранкер в кэше): --only Л1,Л3, --url
#                               <адрес>, --db <путь>, --out <файл.md>
#   ./run.sh rag-compare [...] — сравнение локальной и облачной модели в RAG (день
#                               28): 16 вопросов rag-eval в режимах `фильтр` и
#                               `полный`, каждый по N повторов; отчёт Markdown и
#                               JSON записей в src/data/rag/compare/ (индекс и
#                               реранкер в кэше; ключ DeepSeek — только колонке
#                               «Базовый»; локальную колонку запускайте сразу
#                               после lms load): --presets, --modes, --repeat,
#                               --only, --temperature, --db, --out, --merge
#                               a.json,b.json (общий отчёт без вызовов модели)
#   ./run.sh service [...]    — локальная модель как приватный сервис (день 30):
#                               OpenAI-совместимый прокси перед LM Studio и
#                               страница чата на http://<адрес Mac>:8800; ключи
#                               клиентов — TOOMANYRULES_SERVICE_KEYS в src/.env;
#                               --host, --port, --upstream <адрес LM Studio>,
#                               --new-key (напечатать новый ключ и выйти)
#   ./run.sh service-load [...] — проверка стабильности сервиса (день 30): ключ,
#                               окно модели, параллельные запросы, лимит; отчёт
#                               Markdown в src/data/service/: --url, --scenarios,
#                               --requests N, --clients a,b, --out <файл.md>
APP="app.py"
case "${1:-}" in
    "")        APP="app.py" ;;
    week1)     APP="app_week1.py" ;;
    faq-watch) APP="" ;;
    index)     APP="rag_index.py" ;;
    rag-eval)  APP="rag_eval.py" ;;
    rag-dialog) APP="rag_dialog.py" ;;
    local)     APP="local_llm.py" ;;
    rag-compare) APP="rag_compare.py" ;;
    service)   APP="llm_service.py" ;;
    service-load) APP="service_load.py" ;;
    *)
        echo "Неизвестный аргумент: $1. Допустимо: ./run.sh, ./run.sh week1, ./run.sh faq-watch [аргументы сервера], ./run.sh index [аргументы индекса], ./run.sh rag-eval [аргументы прогона], ./run.sh rag-dialog [аргументы прогона], ./run.sh local [аргументы прогона], ./run.sh rag-compare [аргументы прогона], ./run.sh service [аргументы сервиса] или ./run.sh service-load [аргументы прогона]" >&2
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

if [ "$APP" = "rag_index.py" ]; then
    # Индексу ключ DeepSeek не нужен: LLM он не вызывает, эмбеддинги считает
    # локальная модель. Поэтому src/.env здесь не проверяется.
    exec python rag_index.py "${@:2}"
fi

if [ "$APP" = "local_llm.py" ]; then
    # Локальной модели ключ не нужен (api_key_env=None): LLM отвечает LM Studio
    # на этой машине. Поэтому src/.env здесь не проверяется, как у index.
    exec python local_llm.py "${@:2}"
fi

if [ "$APP" = "rag_compare.py" ]; then
    # Ключ нужен только колонке DeepSeek («Базовый»), и проверяет его сама
    # программа (отказ до вызова модели); локальной колонке и --merge ключ не
    # нужен. Поэтому src/.env здесь не проверяется, как у local.
    exec python rag_compare.py "${@:2}"
fi

if [ "$APP" = "llm_service.py" ] || [ "$APP" = "service_load.py" ]; then
    # Сервису и программе нагрузки ключ DeepSeek не нужен: модель — LM Studio
    # на этой машине. Свои ключи клиентов (TOOMANYRULES_SERVICE_KEYS) они
    # проверяют сами и отказывают с подсказкой, поэтому src/.env здесь не
    # проверяется, как у local.
    exec python "$APP" "${@:2}"
fi

if [ ! -f ".env" ]; then
    echo "Внимание: src/.env не найден. Создайте его с DEEPSEEK_API_KEY=... перед запуском (см. README, шаг 2)." >&2
fi

if [ "$APP" = "rag_eval.py" ]; then
    # Программе сравнения ключ нужен (она ходит к модели через агента), поэтому
    # src/.env проверяется выше, как у приложения.
    exec python rag_eval.py "${@:2}"
fi

if [ "$APP" = "rag_dialog.py" ]; then
    # Программе длинных сценариев ключ тоже нужен (ходит к модели через агента);
    # src/.env проверяется выше, как у приложения.
    exec python rag_dialog.py "${@:2}"
fi

python "$APP"
