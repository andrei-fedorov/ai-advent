#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")"
if ! command -v uv >/dev/null 2>&1; then
    echo 'Нужен uv — установите его перед запуском недели 7.' >&2
    exit 2
fi
case "${1:-}" in
    index) program=docs_index ;;
    help) program=devhelp ;;
    review) program=pr_review ;;
    support) program=support_desk ;;
    crm-server) program=crm_server ;;
    git-server) program=git_server ;;
    *) echo 'Команды: index, help, review, support, git-server, crm-server. Аргументы: --help.' >&2; exit 2 ;;
esac
shift
uv sync --quiet
exec uv run --no-sync python "src/$program.py" "$@"
