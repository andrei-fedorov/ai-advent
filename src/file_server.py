# TooManyRules — сервер файлов: свой MCP-сервер выгрузки в Markdown (день 20,
# неделя 4).
#
# **Отдельная программа, а не модуль приложения** (спецификация дня 20, §5.1),
# как `faq_server.py`: её запускает `mcp_client` процессом по stdio на каждый
# вызов, приложение её не импортирует, и она не импортирует ничего из проекта.
# В графе зависимостей приложения её нет. Серверную часть SDK `mcp`
# импортируют программы-серверы — `faq_server.py`, `faq_watch.py`,
# `wiki_server.py` и эта; клиентскую — только `mcp_client.py`.
#
# Сервер знает один каталог — аргумент `--out` — и ничего про игру, FAQ или
# вики: сохраняет то, что передала модель, **по значению** (§2.8). Ссылка
# `s…` дня 19 понятна только серверу FAQ, который её выдал, поэтому гарантии
# «в файле ровно памятка s3» между разными серверами нет — это записано
# прямо. Проверяет сервер свою часть: запись атомарная (временный файл +
# `os.replace`), файлы не перезаписываются (совпадающее имя получает суффикс
# `-2`, `-3`), путь остаётся внутри `--out`, после записи файл перечитывается и
# сверяется байт в байт — не совпал, файл удаляется. Код перенесён из
# `Watch.save_cheatsheet()` дня 19 (`faq_watch.py`), где он больше не нужен.
#
# Сервер — «файлы», а не «памятки»: рядом потом встанут другие форматы
# (`save_pdf`); сегодня инструмент один — `save_markdown`.
#
# stdout процесса — канал JSON-RPC, поэтому логи идут только в stderr, с
# префиксом `[Файлы]`, а он — в терминал приложения как есть.

import argparse
import hashlib
import logging
import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

SERVER_NAME = "toomanyrules-files"
SERVER_VERSION = "1.0.0"

# --- Файл (§5.2) -----------------------------------------------------------
# Имя: только эти символы, до 40 знаков; пусто после очистки — `notes`.
NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")
NAME_MAX = 40
NAME_EMPTY = "notes"
NAME_PARAM_MAX = 60
CONTENT_MAX_CHARS = 20_000
DATE_FORMAT = "%Y-%m-%d"
TIME_FORMAT = "%d.%m.%Y %H:%M"
# Хеш в строке происхождения и ответе — первые 12 знаков sha256.
HASH_SHOWN = 12

# --- Описания (§5.3) -------------------------------------------------------
# Описание инструмента и есть промпт (правило дня 17). Черновик спецификации;
# каждое изменение по сбою живого прогона — строкой в комментарии у константы.
SAVE_DESCRIPTION = (
    "Сохраняет Markdown в файл — последним шагом, когда игрок просит сохранить или выгрузить "
    "результат (памятку, выжимку, ответ). content — текст целиком: для памятки из "
    "faq_summarize — её Markdown как есть; для памятки из нескольких источников — твой текст "
    "со ссылками на источники. Возвращает путь к файлу."
)
NAME_DESCRIPTION = "Имя файла латиницей: «tink», «poison». Дата добавится сама."
CONTENT_DESCRIPTION = "Markdown целиком, до 20 000 символов."

# Аннотации честные (§5.2): пишет файл, но не разрушает (не перезаписывает),
# повтор создаёт второй файл, мир закрытый — свой каталог.
SAVE_ANNOTATIONS = ToolAnnotations(
    read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False,
)

logger = logging.getLogger("toomanyrules.file_server")

# Корень репозитория — только для показа путей в ответе и логах: структура
# папок автора не должна попадать в видео.
_REPO_ROOT = Path(__file__).resolve().parent.parent


def _plural(count: int, one: str, few: str, many: str) -> str:
    if 11 <= count % 100 <= 14:
        return many
    match count % 10:
        case 1:
            return one
        case 2 | 3 | 4:
            return few
        case _:
            return many


def _number(count: int) -> str:
    """«1 842» — разряды через неразрывный пробел, как в примерах §5.2."""
    return f"{count:,}".replace(",", " ")


def display_path(path: Path) -> str:
    """Путь для ответа и логов: внутри репозитория — относительный, в
    домашнем каталоге — через `~` (перенесено из `faq_watch.py` дня 19)."""
    resolved = path.resolve()
    for base, prefix in ((_REPO_ROOT, ""), (Path.home(), "~/")):
        try:
            return prefix + str(resolved.relative_to(base))
        except ValueError:
            continue
    return str(resolved)


def clean_name(raw: str) -> str:
    """Имя файла из параметра `name` (§5.2): только `[a-z0-9-]`, до 40 знаков;
    пусто после очистки — `notes`."""
    cleaned = "".join(ch for ch in raw.strip().lower() if ch in NAME_CHARS)
    cleaned = cleaned.strip("-")[:NAME_MAX].strip("-")
    return cleaned or NAME_EMPTY


def _sha(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


class Files:
    """Каталог файлов — единственное состояние сервера, и то из аргумента."""

    def __init__(self, out_dir: Path) -> None:
        self.out_dir = out_dir

    def _unique_path(self, stem: str) -> Path:
        candidate = self.out_dir / f"{stem}.md"
        number = 2
        while candidate.exists():
            candidate = self.out_dir / f"{stem}-{number}.md"
            number += 1
        return candidate

    def save_markdown(self, name: str, content: str) -> tuple[str, str]:
        """Ответ `save_markdown` и строка для лога (§5.2): файл = `content`
        как есть + пустая строка + строка происхождения; запись атомарная,
        без перезаписи, внутри `--out`; файл перечитывается и сверяется байт
        в байт."""
        if not content.strip():
            raise ToolError("content пустой — сохранять нечего")
        now = datetime.now().astimezone()
        digest = _sha(content)
        provenance = (
            f"Сохранено {now.strftime(TIME_FORMAT)} · сервер файлов TooManyRules · передано "
            f"агентом: {_number(len(content))} "
            f"{_plural(len(content), 'символ', 'символа', 'символов')}, "
            f"sha {digest[:HASH_SHOWN]}…"
        )
        # `content` как есть: сверка байт в байт идёт по нему, поэтому
        # перевод строки в конце не добавляется и не срезается.
        expected = f"{content}\n\n{provenance}\n"

        try:
            self.out_dir.mkdir(parents=True, exist_ok=True)
        except OSError as exc:
            raise ToolError(f"каталог файлов не создаётся: {type(exc).__name__}: {exc}") from exc
        stem = f"{now.strftime(DATE_FORMAT)}-{clean_name(name)}"
        path = self._unique_path(stem)
        # Проверка ещё раз, что путь остался внутри `--out`: очистка имени уже
        # не оставляет в нём «/» и «..», это защита сверх неё.
        if self.out_dir.resolve() not in path.resolve().parents:
            raise ToolError("путь файла вышел за пределы каталога файлов — сохранение отменено")

        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        try:
            tmp.write_text(expected, encoding="utf-8", newline="")
            os.replace(tmp, path)
        except OSError as exc:
            tmp.unlink(missing_ok=True)
            raise ToolError(f"файл не записался: {type(exc).__name__}: {exc}") from exc

        try:
            written = path.read_bytes()
        except OSError as exc:
            raise ToolError(f"файл записан, но не перечитывается: {type(exc).__name__}: {exc}") from exc
        if written != expected.encode("utf-8"):
            path.unlink(missing_ok=True)
            raise ToolError("файл записан, но текст в нём не совпадает — файл удалён")

        shown = display_path(path)
        size = len(written)
        text = "\n".join([
            f"файл: {shown}",
            f"Сохранено: {_number(size)} {_plural(size, 'байт', 'байта', 'байт')}. Проверка: "
            f"файл перечитан, записан ровно переданный текст (sha {digest[:HASH_SHOWN]}… "
            f"совпадает).",
        ])
        return text, f"{shown}, {size} байт, sha совпадает"


def build_server(files: Files) -> MCPServer:
    # `log_level="WARNING"`: строки INFO SDK — шум в терминале приложения.
    server = MCPServer(name=SERVER_NAME, version=SERVER_VERSION, log_level="WARNING")

    @server.tool(
        title="Сохранение Markdown в файл",
        description=SAVE_DESCRIPTION,
        annotations=SAVE_ANNOTATIONS,
        structured_output=False,
    )
    async def save_markdown(
        name: Annotated[
            str, Field(min_length=1, max_length=NAME_PARAM_MAX, description=NAME_DESCRIPTION)
        ],
        content: Annotated[
            str, Field(min_length=1, max_length=CONTENT_MAX_CHARS, description=CONTENT_DESCRIPTION)
        ],
    ) -> str:
        return _logged(
            f'save_markdown("{name}", {len(content)} символов)',
            lambda: files.save_markdown(name, content),
        )

    return server


def _logged(call: str, work) -> str:
    """Строка лога на вызов: итог или отказ. `ToolError` уходит клиенту
    текстом (SDK сделает из него `isError: true`); любое другое исключение —
    ошибка в коде, трассировку пишет SDK."""
    started = time.perf_counter()
    try:
        text, summary = work()
    except ToolError as exc:
        logger.info("%s: отказ — %s, %.2f с", call, exc, time.perf_counter() - started)
        raise
    logger.info("%s: %s, %.2f с", call, summary, time.perf_counter() - started)
    return text


def _args(argv: list[str] | None = None) -> Path:
    parser = argparse.ArgumentParser(
        description="MCP-сервер (stdio): сохранение Markdown в файлы одного каталога.",
    )
    parser.add_argument("--out", required=True, help="каталог файлов; создаётся, если его нет")
    return Path(parser.parse_args(argv).out).expanduser()


def main() -> None:
    out_dir = _args()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[Файлы] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    build_server(Files(out_dir)).run("stdio")


if __name__ == "__main__":
    main()
