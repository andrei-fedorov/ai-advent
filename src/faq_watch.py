# TooManyRules — сторож FAQ: свой долгоживущий MCP-сервер по Streamable HTTP
# (день 18, неделя 4; день 19 — пайплайн памятки к столу: поиск, сжатие
# моделью клиента через MCP sampling, сохранение в файл).
#
# **Отдельная программа, а не модуль приложения** (спецификация дня 18, §3.1),
# как `faq_server.py`, но запускает её не `mcp_client`, а человек:
# `./run.sh faq-watch` (командная строка — `presets.FAQ_WATCH_ARGV`).
# Приложение её не импортирует, в графе зависимостей приложения её нет. Сама
# она импортирует `faq_server.py` — чтение сайта и разбор страниц; это ребро
# между двумя серверами, вне графа приложения. Серверную часть SDK `mcp`
# импортируют две программы — `faq_server.py` и эта; клиентскую — только
# `mcp_client.py`.
#
# Что делает (§2.1):
# 1. по расписанию (по умолчанию раз в сутки) снимает FAQ целиком — список
#    вопросов и тексты всех статей — и сохраняет снимок в SQLite;
# 2. сравнивает снимок с прошлым удачным и после каждого снимка сам пишет
#    сводку в свой терминал (stderr) и события в базу;
# 3. по вызову `faq_changes` отдаёт агрегированную сводку за период — из базы,
#    без запросов к сайту;
# 4. по вызову `faq_watch_schedule` меняет расписание: ответ сразу, снимок по
#    новому расписанию — потом, в фоне;
# 5. (день 19) `faq_search` находит статьи в последнем снимке по английским
#    ключевым словам, `faq_summarize` сжимает найденное в памятку по-русски
#    (модель клиента, через MCP sampling), `cheatsheet_save` сохраняет памятку
#    в Markdown-файл. Данные между тремя шагами передаются по ссылке — id
#    `q…`/`s…`, а не текстом: каждый шаг сам достаёт из базы то, что записал
#    предыдущий, и проверяет, что переданный id — того вида и с той же базы
#    (спецификация дня 19, §2.3).
#
# Сервер знает портал Freshdesk, но не конкретную игру: адрес портала, id
# категории, её имя и путь к базе приходят аргументами командной строки.
# День 19 добавляет `--out` — каталог для файлов памяток.
#
# Процесс живёт долго и держит своё состояние — базу и расписание; подключения
# клиента при этом короткие (одно на вызов), поэтому `stateless_http=True`:
# сессий Streamable HTTP сервер не держит. Планировщик — одна фоновая задача в
# `lifespan` сервера: в SDK v2 для Streamable HTTP lifespan входит один раз на
# процесс, а не на подключение (§2.9).
#
# Единственное место работы с базой сторожа (`sqlite3` из стандартной
# библиотеки). Правило «диск трогает только `storage.py`» — про приложение, а
# это отдельная программа со своими данными; база — данные сервера, приложение
# её не открывает. День 19 добавляет второй вид файлов — памятки в `--out`:
# запись атомарная (временный файл + `os.replace`, как у хранилищ
# приложения), файлы не перезаписываются.
#
# Логи — только stderr, с префиксом `[FAQ-сторож]`. Строк на каждый запрос к
# сайту нет: логгер `faq_server` здесь не настроен, и его INFO не выходит.
#
# `faq_summarize` (день 19, §2.4) — первое место в проекте, где сервер MCP
# просит модель у клиента (`sampling/createMessage`), а не наоборот: сервер
# отвечает `InputRequiredResult`, клиент выполняет просьбу и повторяет
# `tools/call` с ответом (эра протокола 2026-07-28, §2.9). У сервера при этом
# нет ни ключа DeepSeek, ни SDK `openai` — правило «LLM API — только в
# agent.py» этим не нарушается: модель вызывает клиент. Реализовано через
# `Annotated[CreateMessageResult, Resolve(fn)]` SDK v2: параметр с `Resolve` в
# схему инструмента не попадает, модель его не видит.

import argparse
import asyncio
import difflib
import hashlib
import itertools
import json
import logging
import os
import re
import socket
import sqlite3
import sys
import time
from collections.abc import Iterator
from contextlib import asynccontextmanager, closing, contextmanager
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Annotated

import anyio
from mcp.server.mcpserver import Context, MCPServer, Resolve, Sample
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import CreateMessageResult, SamplingMessage, TextContent, ToolAnnotations
from pydantic import Field

from faq_server import ARTICLE_PATH, Article, Config, Site, read_article, read_sections

SERVER_NAME = "toomanyrules-faq-watch"
SERVER_VERSION = "1.1.0"

# --- Сеть ------------------------------------------------------------------
# Адрес — только 127.0.0.1, аргумента хоста нет (§15): наружу сервер не
# открывается. Путь `/mcp` — по умолчанию SDK.
HOST = "127.0.0.1"
DEFAULT_PORT = 8765

# --- Расписание (§2.4) -----------------------------------------------------
# Нижняя граница — вежливость к чужому сайту: полный снимок — 89 запросов.
DEFAULT_INTERVAL_MIN = 1440
MIN_INTERVAL_MIN = 5
MAX_INTERVAL_MIN = 10080
# После неудачной попытки — повтор через `min(интервал, RETRY_MIN)`.
RETRY_MIN = 15
# Планировщик спит не дольше этого и каждый раз заново сверяет стенные часы
# со сроком: на ноутбуке, который засыпал, монотонные часы во сне стоят.
SCHEDULER_TICK_S = 60

# --- Сводка (§3.6) ---------------------------------------------------------
DAYS_DEFAULT = 7
DAYS_MAX = 365
DIFF_MAX_LINES = 12          # строк разницы текста на статью
DIFF_LINE_MAX_CHARS = 200    # строка разницы — абзац статьи; длиннее — с «…»
CHANGES_MAX_ITEMS = 40       # статей в каждой части сводки
# Весь ответ `faq_changes`, символов. 40 статей на часть этого не держат:
# изменённая статья с разницей — до ≈2,7 тыс. символов, а у клиента дня 17
# потолок результата — 12 000, дальше он обрезает хвост, где стоит вся часть
# «по датам сайта». Бюджет — с запасом ниже: та часть резервируется первой,
# события — пока есть место (правка по ревью дня 18).
CHANGES_MAX_CHARS = 10_000
CHANGES_TAIL_RESERVE = 60    # место под строку «… и ещё N — не поместились в ответ»

TIME_FORMAT = "%d.%m.%Y %H:%M"
DATE_FORMAT = "%d.%m.%Y"

# Дата изменения на странице статьи (§2.9): «Modified on: Tue, 13 Nov, 2018 at
# 8:35 AM» — пробелы уже схлопнуты `faq_server`. Часовой пояс портала на
# странице не указан — разобранное время хранится без пояса.
MODIFIED_PREFIX = "Modified on:"
MODIFIED_FORMAT = "%a, %d %b, %Y at %I:%M %p"

# Хеш текста ответа — первые 16 шестнадцатеричных знаков sha256 (§3.5).
HASH_CHARS = 16

# Виды событий сравнения (§2.5).
KIND_NEW = "новая"
KIND_CHANGED = "изменена"
KIND_MOVED = "перенесена"
KIND_DELETED = "удалена"
KINDS = (KIND_NEW, KIND_CHANGED, KIND_MOVED, KIND_DELETED)

# Откуда взялся интервал — для строки старта.
ORIGIN_DB = "из базы"
ORIGIN_DEFAULT = "по умолчанию"
ORIGIN_ARG = "из --interval"

# --- База (§3.9, день 19 §3.7) ----------------------------------------------
# Версия схемы — `PRAGMA user_version`. 0 у пустой базы: схема создаётся
# целиком; 1 — база дня 18: добавляются только три новые таблицы; своя версия
# (2) — работаем; любая другая — чужая база, сервер не стартует. Миграция
# 1 → 2 — единственная, добавочная: таблицы дня 18 не меняются.
SCHEMA_VERSION = 2
SCHEMA = (
    """CREATE TABLE runs (
        id INTEGER PRIMARY KEY, started_at TEXT NOT NULL, finished_at TEXT NOT NULL,
        ok INTEGER NOT NULL, error TEXT NOT NULL DEFAULT '', articles INTEGER NOT NULL DEFAULT 0,
        requests INTEGER NOT NULL DEFAULT 0, elapsed_s REAL NOT NULL)""",
    "CREATE TABLE texts (hash TEXT PRIMARY KEY, text TEXT NOT NULL)",
    """CREATE TABLE articles (
        run_id INTEGER NOT NULL REFERENCES runs(id), article_id TEXT NOT NULL,
        section TEXT NOT NULL, question TEXT NOT NULL, modified_raw TEXT NOT NULL,
        modified_at TEXT, text_hash TEXT NOT NULL REFERENCES texts(hash), chars INTEGER NOT NULL,
        PRIMARY KEY (run_id, article_id))""",
    """CREATE TABLE changes (
        run_id INTEGER NOT NULL REFERENCES runs(id), article_id TEXT NOT NULL,
        kind TEXT NOT NULL, detail TEXT NOT NULL DEFAULT '')""",
    "CREATE TABLE settings (key TEXT PRIMARY KEY, value TEXT NOT NULL)",
)
# Таблицы дня 19 (§3.7) — пайплайн памятки: `searches` хранит, что и где
# нашлось (`article_ids` — JSON-список в порядке поиска, ссылка на снимок —
# `run_id`), `summaries` — сжатую памятку и её хеш (для проверки на шаге
# сохранения), `saves` — куда и с каким хешем файла она легла.
SCHEMA_V19 = (
    """CREATE TABLE searches (
        id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, run_id INTEGER NOT NULL REFERENCES runs(id),
        query TEXT NOT NULL, limit_n INTEGER NOT NULL, article_ids TEXT NOT NULL)""",
    """CREATE TABLE summaries (
        id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, search_id INTEGER NOT NULL REFERENCES searches(id),
        topic TEXT NOT NULL, text TEXT NOT NULL, text_hash TEXT NOT NULL, model TEXT NOT NULL,
        items INTEGER NOT NULL, foreign_refs TEXT NOT NULL DEFAULT '')""",
    """CREATE TABLE saves (
        id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, summary_id INTEGER NOT NULL REFERENCES summaries(id),
        path TEXT NOT NULL, file_hash TEXT NOT NULL, bytes INTEGER NOT NULL)""",
)
SETTING_INTERVAL = "interval_minutes"

# --- Описания инструментов (§3.7) ------------------------------------------
# Описание инструмента и есть промпт (правило дня 17). Черновик спецификации;
# каждое изменение по сбою живого прогона — строкой в комментарии у константы.
CHANGES_DESCRIPTION = (
    "Сводка изменений официального FAQ издателя по игре {name} за последние дни: что сторож "
    "FAQ заметил, сравнивая снимки, которые он делает по расписанию (новые, изменённые, "
    "перенесённые и удалённые статьи), какие статьи по датам сайта изменены за период, и "
    "состояние самого сторожа. Вызывай, когда игрок спрашивает, что нового или что "
    "изменилось в FAQ, в разъяснениях или эррате издателя. Для вопросов о самих правилах — "
    "faq_questions и faq_article. Отвечая, перескажи по-русски и дай ссылки на статьи."
)
DAYS_DESCRIPTION = "За сколько последних дней: от 1 до 365. «За месяц» — 30, «за неделю» — 7."
SCHEDULE_DESCRIPTION = (
    "Меняет, как часто сторож проверяет FAQ по игре {name}: интервал в минутах, от 5 до "
    "10080 (неделя). Сохраняется и после перезапуска сервера. Вызывай только по прямой "
    "просьбе игрока изменить частоту проверок — не для того, чтобы узнать новости FAQ."
)
INTERVAL_DESCRIPTION = "Интервал между проверками в минутах: 60 — раз в час, 1440 — раз в сутки."

# Аннотации честные (§3.3): мир закрытый — инструменты работают с базой, а не
# с сайтом. Сводка только читает; расписание пишет, но не разрушает, и повтор
# с тем же значением ничего не меняет.
CHANGES_ANNOTATIONS = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)
SCHEDULE_ANNOTATIONS = ToolAnnotations(
    read_only_hint=False,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=False,
)

# --- Памятка к столу (день 19) ----------------------------------------------
# Ссылки между шагами пайплайна — id внутри базы сторожа, не тексты (§2.3):
# `q…` — результат `faq_search`, `s…` — результат `faq_summarize`. Модель
# передаёт id следующему шагу, шаг сам достаёт данные из базы.
SEARCH_PREFIX = "q"
SUMMARY_PREFIX = "s"

# Поиск по словам (§2.5): токены — `[a-z0-9']+` в нижнем регистре, от 3 букв,
# без короткого списка служебных английских слов (черновик; правки по
# прогону — строкой в комментарии). Слово запроса совпадает, если оно —
# начало слова статьи («poison» находит «poisoned»).
_SEARCH_WORD_RE = re.compile(r"[a-z0-9']+")
SEARCH_STOPWORDS = frozenset({
    "the", "and", "for", "with", "how", "does", "can", "that", "this",
    "from", "are", "was", "were", "has", "have", "not", "but", "you",
    "your", "when", "what", "who", "why", "his", "her", "its", "they",
    "them", "their", "any", "all", "did", "yes", "get", "one",
})
SEARCH_QUESTION_WEIGHT = 3
SEARCH_TEXT_WEIGHT = 1
SEARCH_LIMIT_DEFAULT = 5
SEARCH_LIMIT_MAX = 10
NO_SNAPSHOT_ERROR = "снимков FAQ ещё нет — сторож делает первый снимок; попробуй через минуту"
# Ответ модели клиента без пунктов, но по теме (§3.6) — единственный
# допустимый ответ без «- »: сервер про игру не знает и промпт запрашивает
# ровно эту строку.
NO_ANSWER_LINE = "В найденных статьях нет ответа на тему"

# Имя файла памятки (§2.6): только эти символы, до 40 знаков; пусто после
# очистки — по номеру памятки.
CHEATSHEET_NAME_CHARS = frozenset("abcdefghijklmnopqrstuvwxyz0123456789-")
CHEATSHEET_NAME_MAX = 40
CHEATSHEET_DATE_FORMAT = "%Y-%m-%d"

# Промпт сжатия — промпт сервера, а не агента (§3.6): сервер решает, о чём
# просить, клиент — какой моделью. Каждое правило закрывает возможный сбой
# памятки; правки по живому прогону — строкой в комментарии у константы.
SUMMARY_PROMPT = (
    "Ты составляешь памятку к игровому столу по официальному FAQ настольной игры {name}. "
    "Пиши по-русски, только по статьям ниже, ничего не добавляй от себя. Игровые термины "
    "оставляй по-английски (Poison, Baddie, Gearloc). Формат: от 3 до 8 пунктов, каждый "
    "начинается с «- », один пункт — одно правило, в конце пункта — номер статьи-источника "
    "в квадратных скобках, например [33000210161]. Не больше 200 слов. Без заголовка и "
    "вступления. Если статьи не отвечают на тему — одна строка: «" + NO_ANSWER_LINE + "»."
)
SUMMARY_MAX_TOKENS = 1000     # страховка; длину держит «не больше 200 слов» (урок дня 9)

# Описания — черновик; правки по прогону — строкой в комментарии у константы.
SEARCH_DESCRIPTION = (
    "Шаг 1 памятки к столу: поиск статей официального FAQ по игре {name} в последнем снимке "
    "сторожа — по английским ключевым словам (FAQ английский). Возвращает id поиска (q…) и "
    "найденные статьи. Для памятки передай id поиска в faq_summarize — сам статьи не "
    "пересказывай. Для ответа на обычный вопрос о правилах — faq_questions и faq_article."
)
QUERY_DESCRIPTION = (
    "Английские ключевые слова через пробел: «poison», «tink bots». Ищутся по началу слова "
    "в вопросе и тексте статьи."
)
LIMIT_DESCRIPTION = f"Сколько статей взять, 1-{SEARCH_LIMIT_MAX}."
SUMMARIZE_DESCRIPTION = (
    "Шаг 2 памятки: сжимает статьи из результата faq_search в памятку по-русски со ссылками на "
    "статьи. Принимает только id поиска (q…) — тексты передавать не нужно, сервер берёт их сам "
    "из того же снимка. Сжатие делает модель клиента по просьбе сервера. Возвращает id памятки "
    "(s…) и её текст."
)
SEARCH_ID_DESCRIPTION = "id поиска из первой строки ответа faq_search: q7."
TOPIC_DESCRIPTION = "Тема памятки по-русски — станет её заголовком: «Яд (Poison)»."
SAVE_DESCRIPTION = (
    "Шаг 3 памятки: сохраняет памятку (id s… из faq_summarize) в Markdown-файл с источниками и "
    "проверяет, что записан ровно её текст. Вызывай, когда игрок просит сохранить памятку."
)
SUMMARY_ID_DESCRIPTION = "id памятки из первой строки ответа faq_summarize: s3."
NAME_DESCRIPTION = "Необязательное имя файла латиницей: «tink-bots». Пусто — по номеру памятки."

# Аннотации честные (§3.1): все три пишут в базу (или файл), не разрушают
# (не перезаписывают), мир закрытый — данные только из снимков сторожа.
SEARCH_ANNOTATIONS = ToolAnnotations(
    read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False,
)
SUMMARIZE_ANNOTATIONS = ToolAnnotations(
    read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False,
)
SAVE_ANNOTATIONS = ToolAnnotations(
    read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False,
)

logger = logging.getLogger("toomanyrules.faq_watch")

# Корень репозитория — только для показа путей в логах: структура папок
# автора не должна попадать в видео.
_REPO_ROOT = Path(__file__).resolve().parent.parent


# --- Мелочи ----------------------------------------------------------------

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


def _count(count: int, one: str, few: str, many: str) -> str:
    return f"{count} {_plural(count, one, few, many)}"


def _articles_word(count: int) -> str:
    return _count(count, "статья", "статьи", "статей")


def _bytes_word(count: int) -> str:
    return _count(count, "байт", "байта", "байт")


def _events_word(count: int) -> str:
    return _count(count, "событие", "события", "событий")


def _requests_word(count: int) -> str:
    return _count(count, "запрос", "запроса", "запросов")


def _now() -> datetime:
    """Стенные часы с часовым поясом машины, до секунд (§3.9)."""
    return datetime.now().astimezone().replace(microsecond=0)


def _iso(moment: datetime) -> str:
    return moment.isoformat(timespec="seconds")


def _fmt(moment: datetime) -> str:
    return moment.astimezone().strftime(TIME_FORMAT)


def display_path(path: Path) -> str:
    """Путь для логов: внутри репозитория — относительный, в домашнем
    каталоге — через `~`."""
    resolved = path.resolve()
    for base, prefix in ((_REPO_ROOT, ""), (Path.home(), "~/")):
        try:
            return prefix + str(resolved.relative_to(base))
        except ValueError:
            continue
    return str(resolved)


def parse_modified(raw: str) -> datetime | None:
    """«Modified on: Tue, 13 Nov, 2018 at 8:35 AM» → время без пояса; не
    разобралось — `None` (это не сбой снимка, а предупреждение в сводке)."""
    text = " ".join(raw.split())
    if text.startswith(MODIFIED_PREFIX):
        text = text[len(MODIFIED_PREFIX):].strip()
    try:
        return datetime.strptime(text, MODIFIED_FORMAT)
    except ValueError:
        return None


def _size(lines: list[str]) -> int:
    """Длина строк, склеенных через перевод строки, — в символах ответа."""
    return sum(len(line) + 1 for line in lines)


def text_hash(text: str) -> str:
    return hashlib.sha256(text.encode("utf-8")).hexdigest()[:HASH_CHARS]


def next_due(
    last_ok: datetime | None, failed_after: datetime | None, interval_min: int, now: datetime
) -> datetime:
    """Срок следующего снимка — чистая функция над историей попыток (§2.4).

    `last_ok` — начало последнего удачного снимка; `failed_after` — начало
    последней неудачной попытки, если она была после него (или удачных нет
    вовсе). Срок уже прошёл — вызывающий снимает сразу. Неудача без удачных
    снимков тоже ждёт повтора, а не идёт сразу: иначе недоступный сайт
    опрашивался бы без передышки.
    """
    if failed_after is not None:
        return failed_after + timedelta(minutes=min(interval_min, RETRY_MIN))
    if last_ok is None:
        return now
    return last_ok + timedelta(minutes=interval_min)


# --- База ------------------------------------------------------------------

class DbError(Exception):
    """База чужая или битая — сервер не стартует и ничего в неё не пишет."""


def check_db(path: Path) -> int:
    """Версия схемы существующей базы; 0 — нового пути или пустого файла.
    Только чтение (`mode=ro`): отказ не должен оставить следов в файле.
    Версия 1 (день 18) принимается — `init_db()` домигрирует её до 2."""
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        uri = path.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            tables = conn.execute("SELECT count(*) FROM sqlite_master").fetchone()[0]
    except sqlite3.Error as exc:
        raise DbError(f"файл не открывается как база SQLite: {exc}") from exc
    if version == 0 and tables:
        raise DbError("версия схемы 0, но в базе уже есть таблицы — это не база сторожа")
    if version not in (0, 1, SCHEMA_VERSION):
        raise DbError(f"версия схемы {version}, сервер знает только {SCHEMA_VERSION}")
    return version


def _connect(path: Path) -> sqlite3.Connection:
    # Автокоммит на уровне драйвера: транзакции открываются явно
    # (`_transaction`), чтобы запись снимка была ровно одной транзакцией.
    conn = sqlite3.connect(path, isolation_level=None, timeout=5)
    conn.row_factory = sqlite3.Row
    return conn


@contextmanager
def _transaction(conn: sqlite3.Connection) -> Iterator[None]:
    conn.execute("BEGIN IMMEDIATE")
    try:
        yield
    except BaseException:
        conn.execute("ROLLBACK")
        raise
    conn.execute("COMMIT")


def init_db(path: Path, version: int) -> None:
    """Схема для новой базы (одной транзакцией, вместе с версией) и режим WAL:
    инструменты читают, пока снимок пишет. Версия 1 (база дня 18) —
    единственная миграция, добавочная: дописывает только три новые таблицы
    (§3.7), таблицы дня 18 не трогает."""
    if version == SCHEMA_VERSION:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(path)) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        with _transaction(conn):
            if version == 0:
                for statement in SCHEMA:
                    conn.execute(statement)
            for statement in SCHEMA_V19:
                conn.execute(statement)
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")


@dataclass(frozen=True)
class Run:
    """Попытка снимка — строка `runs`."""
    id: int
    started_at: datetime
    ok: bool
    error: str
    articles: int
    requests: int
    elapsed_s: float


def _run(row: sqlite3.Row) -> Run:
    return Run(
        id=row["id"],
        started_at=datetime.fromisoformat(row["started_at"]),
        ok=bool(row["ok"]),
        error=row["error"],
        articles=row["articles"],
        requests=row["requests"],
        elapsed_s=row["elapsed_s"],
    )


@dataclass(frozen=True)
class Snapshot:
    """Статья снимка — то, что нужно для сравнения (§2.5)."""
    article_id: str
    section: str
    question: str
    modified_raw: str
    modified_at: datetime | None
    hash: str
    chars: int


def _snapshot_rows(conn: sqlite3.Connection, run_id: int) -> dict[str, Snapshot]:
    rows = conn.execute(
        "SELECT article_id, section, question, modified_raw, modified_at, text_hash, chars "
        "FROM articles WHERE run_id = ? ORDER BY rowid",
        (run_id,),
    )
    return {
        row["article_id"]: Snapshot(
            article_id=row["article_id"],
            section=row["section"],
            question=row["question"],
            modified_raw=row["modified_raw"],
            modified_at=datetime.fromisoformat(row["modified_at"]) if row["modified_at"] else None,
            hash=row["text_hash"],
            chars=row["chars"],
        )
        for row in rows
    }


# --- Сравнение (§2.5) ------------------------------------------------------

@dataclass(frozen=True)
class Event:
    article_id: str
    kind: str
    before: Snapshot | None
    after: Snapshot | None

    def detail(self) -> str:
        """JSON для `changes.detail`: вопрос, раздел, хеш и длина было/стало.
        Разницу строк сводка строит на лету по двум текстам из `texts`."""
        def side(snap: Snapshot | None) -> dict | None:
            if snap is None:
                return None
            return {
                "section": snap.section,
                "question": snap.question,
                "hash": snap.hash,
                "chars": snap.chars,
            }
        return json.dumps(
            {"before": side(self.before), "after": side(self.after)}, ensure_ascii=False
        )


def compare(previous: dict[str, Snapshot], current: dict[str, Snapshot]) -> list[Event]:
    """События нового снимка относительно прошлого удачного — в порядке
    нового снимка, удалённые — в конце, в порядке прошлого. Одна смена даты
    «Modified on» событием не считается: игроку она ничего не говорит."""
    events: list[Event] = []
    for article_id, after in current.items():
        before = previous.get(article_id)
        if before is None:
            events.append(Event(article_id, KIND_NEW, None, after))
            continue
        if before.section != after.section:
            events.append(Event(article_id, KIND_MOVED, before, after))
        if before.question != after.question or before.hash != after.hash:
            events.append(Event(article_id, KIND_CHANGED, before, after))
    for article_id, before in previous.items():
        if article_id not in current:
            events.append(Event(article_id, KIND_DELETED, before, None))
    return events


def _event_from_row(row: sqlite3.Row) -> Event:
    detail = json.loads(row["detail"] or "{}")

    def side(data: dict | None) -> Snapshot | None:
        if not data:
            return None
        return Snapshot(
            article_id=row["article_id"], section=data.get("section", ""),
            question=data.get("question", ""), modified_raw="", modified_at=None,
            hash=data.get("hash", ""), chars=data.get("chars", 0),
        )

    return Event(row["article_id"], row["kind"], side(detail.get("before")), side(detail.get("after")))


def _event_brief(event: Event) -> str:
    """Событие одной строкой — для сводки снимка в терминале."""
    snap = event.after or event.before
    head = f"{event.kind} · {event.article_id} · {snap.section}"
    if event.kind == KIND_MOVED:
        return f"{head} · раздел: было {event.before.section}"
    if event.kind == KIND_CHANGED:
        parts = []
        if event.before.question != event.after.question:
            parts.append("вопрос изменён")
        if event.before.hash != event.after.hash:
            parts.append(f"текст {event.before.chars} → {event.after.chars} симв.")
        return f"{head} · {' · '.join(parts)}"
    return f"{head} · {snap.question}"


def _clip(line: str) -> str:
    if len(line) <= DIFF_LINE_MAX_CHARS:
        return line
    return line[:DIFF_LINE_MAX_CHARS].rstrip() + " …"


def text_diff(old: str, new: str) -> list[str]:
    """Строки разницы без контекста, не больше `DIFF_MAX_LINES`.

    Заголовок `---`/`+++` — первые две строки вывода — отбрасывается по
    позиции: по началу строки выпала бы и строка статьи, начатая с «--».
    Пустые строки (промежутки между абзацами) — не разница, а шум, который
    съедал бы лимит строк (правка по ревью дня 18)."""
    lines = [
        line for line in itertools.islice(
            difflib.unified_diff(old.splitlines(), new.splitlines(), lineterm="", n=0), 2, None
        )
        if line[:1] in ("+", "-") and line[1:].strip()
    ]
    shown = [_clip(f"{line[0]} {line[1:].strip()}") for line in lines[:DIFF_MAX_LINES]]
    if len(lines) > DIFF_MAX_LINES:
        shown.append(f"… и ещё {len(lines) - DIFF_MAX_LINES} строк")
    return shown


# --- Памятка к столу: поиск, id, имя файла (день 19, §2.3, §2.5, §2.6) -----

def _search_words(text: str) -> list[str]:
    """Токены для поиска: `[a-z0-9']+` в нижнем регистре, от 3 букв, без
    короткого списка служебных слов (§2.5)."""
    return [
        word for word in _SEARCH_WORD_RE.findall(text.lower())
        if len(word) >= 3 and word not in SEARCH_STOPWORDS
    ]


def _score_article(
    query_words: list[str], question_words: list[str], text_words: list[str]
) -> tuple[int, dict[str, list[str]]]:
    """Оценка статьи по словам запроса (§2.5): слово запроса совпадает, если
    оно — начало слова статьи («poison» находит «poisoned»); 3 за совпадение
    в вопросе, 1 — в тексте. Возвращает оценку и то, где какое слово
    совпало — для строки ответа."""
    score = 0
    hits: dict[str, list[str]] = {}
    for word in query_words:
        in_question = any(article_word.startswith(word) for article_word in question_words)
        in_text = any(article_word.startswith(word) for article_word in text_words)
        if in_question:
            score += SEARCH_QUESTION_WEIGHT
            hits.setdefault(word, []).append("вопрос")
        if in_text:
            score += SEARCH_TEXT_WEIGHT
            hits.setdefault(word, []).append("текст")
    return score, hits


def _require_search_ref(raw: str) -> int:
    """Разбирает id поиска (`q…`) — отдельная проверка вида, а не JSON Schema
    (§3.1): модель лучше исправляется по понятному тексту, чем по ошибке
    валидации pydantic. Регистр и пробелы по краям не важны: «Q7» — тот же
    q7 (правка по ревью)."""
    raw = raw.strip().lower()
    tail = raw[len(SUMMARY_PREFIX):]
    if raw.startswith(SUMMARY_PREFIX) and tail.isdigit():
        raise ToolError(f"{raw} — id памятки, а нужен id поиска ({SEARCH_PREFIX}…) из faq_search")
    tail = raw[len(SEARCH_PREFIX):]
    if not (raw.startswith(SEARCH_PREFIX) and tail.isdigit()):
        raise ToolError(f"«{raw}» не похоже на id поиска ({SEARCH_PREFIX}…) из faq_search")
    return int(tail)


def _require_summary_ref(raw: str) -> int:
    """Разбирает id памятки (`s…`) — та же проверка вида, что у поиска."""
    raw = raw.strip().lower()
    tail = raw[len(SEARCH_PREFIX):]
    if raw.startswith(SEARCH_PREFIX) and tail.isdigit():
        raise ToolError(f"{raw} — id поиска, а нужен id памятки ({SUMMARY_PREFIX}…) из faq_summarize")
    tail = raw[len(SUMMARY_PREFIX):]
    if not (raw.startswith(SUMMARY_PREFIX) and tail.isdigit()):
        raise ToolError(f"«{raw}» не похоже на id памятки ({SUMMARY_PREFIX}…) из faq_summarize")
    return int(tail)


def _clean_cheatsheet_name(raw: str, summary_id: int) -> str:
    """Имя файла из параметра `name` (§2.6): только `[a-z0-9-]`, до 40 знаков;
    пусто после очистки — по номеру памятки."""
    cleaned = "".join(ch for ch in raw.strip().lower() if ch in CHEATSHEET_NAME_CHARS)
    cleaned = cleaned.strip("-")[:CHEATSHEET_NAME_MAX].strip("-")
    return cleaned or f"cheatsheet-{SUMMARY_PREFIX}{summary_id}"


def _summary_user_text(topic: str, articles: list[tuple[str, str, str, str]]) -> str:
    """Вход сжатия (§3.4): «Тема: …», затем по статье — заголовок и текст, в
    порядке поиска. `articles` — (номер, раздел, вопрос, текст)."""
    parts = [f"Тема: {topic}"]
    for article_id, section, question, text in articles:
        parts.append(f"### {article_id} · {section} · {question}\n{text}")
    return "\n\n".join(parts)


def _summary_items(text: str) -> list[str]:
    return [line for line in text.splitlines() if line.startswith("- ")]


def _is_no_answer(text: str) -> bool:
    """Ответ «нет ответа на тему» (§3.6) — без точки, кавычек и «ёлочек» по
    краям: промпт сам показывает строку в «ёлочках», и модель может их
    повторить. Отказ после оплаченного сэмплинга из-за знака препинания был
    бы ложным (правка по ревью)."""
    return text.strip().strip(" .«»\"'").strip() == NO_ANSWER_LINE


@dataclass(frozen=True)
class SearchRecord:
    """Строка `searches` (§3.7): что искали и на каком снимке."""
    id: int
    query: str
    limit: int
    run_id: int
    article_ids: list[str]


# --- Сторож ----------------------------------------------------------------

class SnapshotError(Exception):
    """Снимок неполный — попытка неудачная, со статьями не записывается."""


@dataclass
class Running:
    number: int
    started_at: datetime


class Watch:
    """Состояние процесса сторожа: где база, какой интервал, идёт ли снимок и
    как разбудить планировщик. Один на процесс; снимки делает только
    планировщик, инструменты читают базу и меняют интервал."""

    def __init__(self, config: Config, db: Path, out_dir: Path) -> None:
        self.config = config
        self.db = db
        # Каталог памяток (день 19, §2.6) — аргумент `--out`; создаётся, если
        # его нет, до первого вызова `cheatsheet_save()`.
        self.out_dir = out_dir
        self.interval = DEFAULT_INTERVAL_MIN
        self.interval_origin = ORIGIN_DEFAULT
        self.running: Running | None = None
        # Срок «не раньше» после попытки, которой нет в `runs` (§2.4, правка
        # по ревью): сайт прочитан, но база снимок не приняла, или после
        # чтения упал код. Срок из базы тогда остался в прошлом, и без этого
        # поля планировщик перечитывал бы сайт на каждом шаге проверки часов.
        # Только в памяти процесса: запись попытки его снимает, перезапуск —
        # тоже.
        self.not_before: datetime | None = None
        # Событие создаётся в цикле событий — при старте планировщика.
        self.wake: anyio.Event | None = None

    # --- расписание ---

    def load_interval(self, from_arg: int | None) -> None:
        """Кто задал интервал последним, тот и прав (§2.4): явный `--interval`
        перекрывает сохранённый и сохраняется сам; без него — из базы или
        умолчание."""
        if from_arg is not None:
            self.save_interval(from_arg)
            self.interval_origin = ORIGIN_ARG
            return
        with closing(_connect(self.db)) as conn:
            row = conn.execute(
                "SELECT value FROM settings WHERE key = ?", (SETTING_INTERVAL,)
            ).fetchone()
        if row is not None and str(row["value"]).isdigit():
            value = int(row["value"])
            if MIN_INTERVAL_MIN <= value <= MAX_INTERVAL_MIN:
                self.interval, self.interval_origin = value, ORIGIN_DB
                return
            logger.warning(
                "интервал в базе вне %d-%d мин: %s — действует умолчание",
                MIN_INTERVAL_MIN, MAX_INTERVAL_MIN, row["value"],
            )
        self.interval, self.interval_origin = DEFAULT_INTERVAL_MIN, ORIGIN_DEFAULT

    def save_interval(self, minutes: int) -> None:
        with closing(_connect(self.db)) as conn, _transaction(conn):
            conn.execute(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                (SETTING_INTERVAL, str(minutes)),
            )
        self.interval = minutes

    def attempts(self, conn: sqlite3.Connection) -> tuple[Run | None, Run | None]:
        """Последний удачный снимок и неудачная попытка после него (если
        последняя попытка — неудачная)."""
        last_ok = conn.execute("SELECT * FROM runs WHERE ok = 1 ORDER BY id DESC LIMIT 1").fetchone()
        last = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()
        ok_run = _run(last_ok) if last_ok is not None else None
        failed = _run(last) if last is not None and not last["ok"] else None
        return ok_run, failed

    def next_due(self, conn: sqlite3.Connection | None = None) -> datetime:
        if conn is None:
            with closing(_connect(self.db)) as own:
                return self.next_due(own)
        ok_run, failed = self.attempts(conn)
        due = next_due(
            ok_run.started_at if ok_run else None,
            failed.started_at if failed else None,
            self.interval,
            _now(),
        )
        if self.not_before is not None and self.not_before > due:
            return self.not_before
        return due

    def _hold_off(self) -> datetime:
        """Срок «не раньше» после попытки, которой нет в базе:
        `min(интервал, RETRY_MIN)` от сейчас — как повтор после неудачи."""
        self.not_before = _now() + timedelta(minutes=min(self.interval, RETRY_MIN))
        return self.not_before

    def _due_text(self, due: datetime) -> str:
        if self.running is not None:
            return f"идёт сейчас (снимок #{self.running.number}, начат {_fmt(self.running.started_at)})"
        if due <= _now():
            return "сейчас — срок уже прошёл"
        return _fmt(due)

    # --- планировщик (§3.4) ---

    async def scheduler(self) -> None:
        """Один цикл: снимки не перекрываются, срок — из базы по стенным
        часам, смена интервала будит цикл раньше конца сна."""
        self.wake = anyio.Event()
        while True:
            try:
                due = self.next_due()
                while _now() < due and not self.wake.is_set():
                    wait_s = min(SCHEDULER_TICK_S, max(0.0, (due - _now()).total_seconds()))
                    with anyio.move_on_after(wait_s):
                        await self.wake.wait()
                self.wake = anyio.Event()
                # Интервал мог смениться за время сна — пересчитать срок.
                due = self.next_due()
                if _now() >= due:
                    await self.snapshot(due)
            except Exception:
                # Сторож не падает от одной ошибки: трассировка — в stderr.
                # Ошибка могла случиться после чтения сайта, а попытки в базе
                # нет — поэтому снимок не раньше `_hold_off()`, а не на
                # следующем шаге проверки часов. Сон — на случай, если падает
                # само чтение срока из базы: без него цикл крутился бы вхолостую.
                logger.exception(
                    "планировщик: ошибка в коде сервера — снимок не раньше %s",
                    _fmt(self._hold_off()),
                )
                await anyio.sleep(SCHEDULER_TICK_S)

    # --- снимок (§3.5) ---

    async def _read_site(self, site: Site) -> tuple[dict[str, Snapshot], dict[str, str], list[str]]:
        """Все статьи категории: снимок, тексты по хешу и предупреждения."""
        sections, _ = await read_sections(site, self.config)
        incomplete = [s for s in sections if len(s.articles) != s.declared]
        if incomplete:
            raise SnapshotError("неполный снимок: " + "; ".join(
                f"раздел {s.name} — на сайте {s.declared}, прочитано {len(s.articles)}"
                for s in incomplete
            ))
        warnings: list[str] = []
        listed: dict[str, tuple[str, str]] = {}
        for section in sections:
            for article_id, question in section.articles:
                if article_id in listed:
                    warnings.append(
                        f"статья {article_id} в двух разделах: {listed[article_id][0]} и "
                        f"{section.name} — взят первый"
                    )
                    continue
                listed[article_id] = (section.name, question)

        found: dict[str, Article] = {}

        async def one(article_id: str) -> None:
            found[article_id] = await read_article(site, self.config, article_id)

        # Параллельно в пределах того же `Site`: потолок одновременных
        # запросов и счётчик — дня 17. Сбой одной статьи отменяет остальные.
        try:
            async with asyncio.TaskGroup() as group:
                for article_id in listed:
                    group.create_task(one(article_id))
        except ExceptionGroup as errors:
            first = errors.exceptions[0]
            while isinstance(first, ExceptionGroup):
                first = first.exceptions[0]
            raise first from None

        snapshot: dict[str, Snapshot] = {}
        texts: dict[str, str] = {}
        for article_id, (section, question) in listed.items():
            article = found[article_id]
            modified_at = parse_modified(article.modified)
            if modified_at is None:
                warnings.append(
                    f"дата изменения не разобрана: {article_id} — "
                    f"«{article.modified or 'блока даты на странице нет'}»"
                )
            digest = text_hash(article.text)
            texts[digest] = article.text
            # Номер, раздел и вопрос — из списка, дата и текст — со страницы.
            snapshot[article_id] = Snapshot(
                article_id=article_id,
                section=section,
                question=question,
                modified_raw=article.modified,
                modified_at=modified_at,
                hash=digest,
                chars=len(article.text),
            )
        return snapshot, texts, warnings

    def _next_number(self) -> int:
        with closing(_connect(self.db)) as conn:
            return (conn.execute("SELECT max(id) FROM runs").fetchone()[0] or 0) + 1

    async def snapshot(self, due: datetime) -> None:
        """Одна попытка снимка. Исключений наружу не выпускает, кроме отмены:
        неудача чтения сайта — строка `runs` с `ok=0` и причиной; база не
        приняла запись — строка лога и срок «не раньше» (`_hold_off()`)."""
        number = self._next_number()
        started_at = _now()
        started = time.perf_counter()
        self.running = Running(number, started_at)
        logger.info(
            "снимок #%d: начат (срок %s, раз в %d мин)", number, _fmt(due), self.interval
        )
        site = Site(self.config)
        reason = ""
        result = None
        try:
            async with site:
                result = await self._read_site(site)
        except anyio.get_cancelled_exc_class():
            logger.info(
                "снимок #%d: прерван остановкой сервера через %.1f с — в базу не попал",
                number, time.perf_counter() - started,
            )
            raise
        except (ToolError, SnapshotError) as exc:
            reason = str(exc)
        except Exception as exc:
            logger.exception("снимок #%d: ошибка в коде сервера", number)
            reason = f"ошибка в коде сервера: {type(exc).__name__}: {exc}"
        finally:
            self.running = None
        elapsed = time.perf_counter() - started
        finished_at = _now()

        # Запись попытки — отдельно от чтения сайта (правка по ревью): ошибка
        # базы здесь — не «ошибка в коде сервера», и попытки в `runs` после
        # неё нет, поэтому повтор — не раньше `_hold_off()`, а не через шаг
        # проверки часов с новым чтением всего сайта.
        try:
            with closing(_connect(self.db)) as conn:
                if reason:
                    self._record_failure(
                        conn, number, started_at, finished_at, reason, site.requests, elapsed
                    )
                else:
                    snapshot, texts, warnings = result
                    previous_run, events = self._record_snapshot(
                        conn, number, started_at, finished_at, snapshot, texts,
                        site.requests, elapsed,
                    )
        except sqlite3.Error as exc:
            logger.error(
                "снимок #%d: %s, но база его не записала — %s: %s; повтор не раньше %s",
                number, f"сбой ({reason})" if reason else f"прочитан за {elapsed:.1f} с",
                type(exc).__name__, exc, _fmt(self._hold_off()),
            )
            return
        self.not_before = None

        if reason:
            logger.warning(
                "снимок #%d: сбой через %.1f с — %s; повтор %s",
                number, elapsed, reason, _fmt(self.next_due()),
            )
            return
        head = (
            f"снимок #{number}: {_articles_word(len(snapshot))}, "
            f"{_requests_word(site.requests)}, {elapsed:.1f} с"
        )
        if previous_run is None:
            logger.info(
                "%s — первый снимок: %s, база для сравнения", head, _articles_word(len(snapshot))
            )
        else:
            since = f"с #{previous_run.id} ({_fmt(previous_run.started_at)})"
            if events:
                counts = {kind: sum(1 for e in events if e.kind == kind) for kind in KINDS}
                logger.info(
                    "%s — %s: новых %d · изменено %d · перенесено %d · удалено %d",
                    head, since, counts[KIND_NEW], counts[KIND_CHANGED],
                    counts[KIND_MOVED], counts[KIND_DELETED],
                )
                for event in events:
                    logger.info("  %s", _event_brief(event))
            else:
                logger.info("%s — %s: изменений нет", head, since)
        for warning in warnings:
            logger.warning("  ⚠️ %s", warning)
        logger.info("следующий снимок: %s", _fmt(self.next_due()))

    @staticmethod
    def _record_failure(
        conn: sqlite3.Connection, number: int, started_at: datetime, finished_at: datetime,
        reason: str, requests: int, elapsed: float,
    ) -> None:
        """Неудачная попытка — строка `runs` с `ok=0` и причиной, без статей."""
        with _transaction(conn):
            conn.execute(
                "INSERT INTO runs (id, started_at, finished_at, ok, error, requests, elapsed_s) "
                "VALUES (?, ?, ?, 0, ?, ?, ?)",
                (number, _iso(started_at), _iso(finished_at), reason, requests, elapsed),
            )

    def _record_snapshot(
        self, conn: sqlite3.Connection, number: int, started_at: datetime,
        finished_at: datetime, snapshot: dict[str, Snapshot], texts: dict[str, str],
        requests: int, elapsed: float,
    ) -> tuple[Run | None, list[Event]]:
        """Удачный снимок одной транзакцией (§3.5): попытка, статьи, новые
        тексты и события. До её конца в базе от снимка нет ничего. Возвращает
        прошлый удачный снимок и события сравнения с ним."""
        previous_run, _ = self.attempts(conn)
        with _transaction(conn):
            previous = _snapshot_rows(conn, previous_run.id) if previous_run else None
            events = compare(previous, snapshot) if previous is not None else []
            conn.execute(
                "INSERT INTO runs (id, started_at, finished_at, ok, articles, requests, elapsed_s) "
                "VALUES (?, ?, ?, 1, ?, ?, ?)",
                (number, _iso(started_at), _iso(finished_at), len(snapshot), requests, elapsed),
            )
            conn.executemany(
                "INSERT OR IGNORE INTO texts (hash, text) VALUES (?, ?)", texts.items()
            )
            conn.executemany(
                "INSERT INTO articles (run_id, article_id, section, question, modified_raw, "
                "modified_at, text_hash, chars) VALUES (?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        number, snap.article_id, snap.section, snap.question, snap.modified_raw,
                        snap.modified_at.isoformat() if snap.modified_at else None,
                        snap.hash, snap.chars,
                    )
                    for snap in snapshot.values()
                ],
            )
            conn.executemany(
                "INSERT INTO changes (run_id, article_id, kind, detail) VALUES (?, ?, ?, ?)",
                [(number, event.article_id, event.kind, event.detail()) for event in events],
            )
        return previous_run, events

    # --- инструменты ---

    def _article_url(self, article_id: str) -> str:
        return f"{self.config.base_url}{ARTICLE_PATH.format(id=article_id)}"

    def _event_lines(
        self, conn: sqlite3.Connection, event: Event, at: datetime, *, with_diff: bool = True,
    ) -> list[str]:
        snap = event.after or event.before
        lines = [
            f"- {event.kind} · {event.article_id} · {snap.section} · {snap.question} · "
            f"снимок {_fmt(at)}",
            f"  {self._article_url(event.article_id)}",
        ]
        if event.kind == KIND_MOVED:
            lines.append(f"  раздел: было {event.before.section} / стало {event.after.section}")
        elif event.kind == KIND_CHANGED:
            if event.before.question != event.after.question:
                lines.append(
                    f"  вопрос: было «{event.before.question}» / стало «{event.after.question}»"
                )
            if event.before.hash != event.after.hash:
                lines.append(f"  текст: {event.before.chars} → {event.after.chars} симв.")
                if not with_diff:
                    lines.append("  разница не показана — ответ длинный")
                    return lines
                old = conn.execute("SELECT text FROM texts WHERE hash = ?", (event.before.hash,)).fetchone()
                new = conn.execute("SELECT text FROM texts WHERE hash = ?", (event.after.hash,)).fetchone()
                if old is not None and new is not None:
                    lines.extend(f"  {line}" for line in text_diff(old["text"], new["text"]))
        return lines

    def changes(self, days: int) -> tuple[str, str]:
        """Ответ `faq_changes` и строка для лога (§3.6). Только база — ни
        одного запроса к сайту."""
        now = _now()
        since = now - timedelta(days=days)
        with closing(_connect(self.db)) as conn:
            runs = [_run(row) for row in conn.execute("SELECT * FROM runs ORDER BY id")]
            in_period = [run for run in runs if run.started_at >= since]
            ok_runs = [run for run in runs if run.ok]
            last_ok = ok_runs[-1] if ok_runs else None
            first_ok = ok_runs[0] if ok_runs else None
            due = self.next_due(conn)

            lines = [
                f"Сторож FAQ «{self.config.category_name}» ({self.config.category_url})",
            ]
            status = f"Расписание: раз в {self.interval} мин. Снимков за {days} дн.: {len(in_period)}"
            failures = [run for run in in_period if not run.ok]
            if failures:
                last_failure = failures[-1]
                status += (
                    f" (удачных {len(in_period) - len(failures)}, сбоев {len(failures)} — "
                    f"последний {_fmt(last_failure.started_at)}: {last_failure.error})."
                )
            elif in_period:
                status += " (все удачные)."
            else:
                status += "."
            lines.append(status)
            if last_ok is None:
                lines.append(
                    f"Удачных снимков ещё нет. Следующий: {self._due_text(due)}."
                )
            else:
                lines.append(
                    f"Последний удачный снимок: {_fmt(last_ok.started_at)}, "
                    f"{_articles_word(last_ok.articles)}. Следующий: {self._due_text(due)}. "
                    f"Наблюдение ведётся с {_fmt(first_ok.started_at)}."
                )

            # По датам сайта — последний удачный снимок, граница — до суток.
            # Собирается первой: часть короткая, и её место в бюджете
            # `CHANGES_MAX_CHARS` зарезервировано, хотя в ответе она последняя.
            site_title = (
                f"По датам сайта («Modified on», по последнему снимку; часовой пояс сайта "
                f"неизвестен — граница периода с точностью до суток) за {days} дн."
            )
            site_lines: list[str] = []
            dated: list[Snapshot] = []
            unparsed = 0
            if last_ok is None:
                site_lines.append(f"{site_title}: снимков ещё нет.")
            else:
                first_day: date = since.date()
                for snap in _snapshot_rows(conn, last_ok.id).values():
                    if snap.modified_at is None:
                        unparsed += 1
                    elif snap.modified_at.date() >= first_day:
                        dated.append(snap)
                dated.sort(key=lambda snap: snap.modified_at, reverse=True)
                site_lines.append(
                    f"{site_title}: {_articles_word(len(dated))}." if dated
                    else f"{site_title}: статей нет."
                )
                for snap in dated[:CHANGES_MAX_ITEMS]:
                    site_lines.append(
                        f"- {snap.modified_at.strftime(DATE_FORMAT)} · {snap.article_id} · "
                        f"{snap.section} · {snap.question}"
                    )
                    site_lines.append(f"  {self._article_url(snap.article_id)}")
                if len(dated) > CHANGES_MAX_ITEMS:
                    site_lines.append(f"… и ещё {len(dated) - CHANGES_MAX_ITEMS}")
                if unparsed:
                    site_lines.append(f"Дат не разобрано: {unparsed}.")

            # Замечено сторожем — события удачных снимков периода, от новых к старым.
            events: list[tuple[Event, datetime]] = []
            period_ok = {run.id: run.started_at for run in in_period if run.ok}
            if period_ok:
                rows = conn.execute(
                    "SELECT run_id, article_id, kind, detail FROM changes WHERE run_id >= ? "
                    "ORDER BY run_id DESC, rowid",
                    (min(period_ok),),
                )
                events = [
                    (_event_from_row(row), period_ok[row["run_id"]])
                    for row in rows if row["run_id"] in period_ok
                ]
            title = f"Замечено сторожем за {days} дн. (сравнение соседних снимков)"
            observer: list[str] = []
            if first_ok is None:
                observer.append(f"{title}: снимков для сравнения ещё нет.")
            else:
                observer.append(
                    f"{title}: {_events_word(len(events))}." if events
                    else f"{title}: изменений нет."
                )
                if first_ok.started_at > since:
                    observer.append(
                        f"Сторож наблюдает с {_fmt(first_ok.started_at)}, раньше этого ничего "
                        f"не замечено."
                    )
                # Бюджет событий — что осталось от `CHANGES_MAX_CHARS` после
                # остальных частей и запаса на строку «… и ещё N». Событие, не
                # поместившееся с разницей, идёт без неё; не поместившееся и
                # так — конец списка.
                budget = (
                    CHANGES_MAX_CHARS - _size(lines + ["", *observer, "", *site_lines])
                    - CHANGES_TAIL_RESERVE
                )
                shown = 0
                for event, at in events[:CHANGES_MAX_ITEMS]:
                    block = self._event_lines(conn, event, at)
                    if _size(block) > budget:
                        block = self._event_lines(conn, event, at, with_diff=False)
                    if _size(block) > budget:
                        break
                    observer.extend(block)
                    budget -= _size(block)
                    shown += 1
                if shown < len(events):
                    cut = shown < min(len(events), CHANGES_MAX_ITEMS)
                    observer.append(
                        f"… и ещё {len(events) - shown}"
                        + (" — не поместились в ответ" if cut else "")
                    )
        lines += ["", *observer, "", *site_lines]
        summary = f"{_events_word(len(events))}, {len(dated)} по датам сайта"
        return "\n".join(lines), summary

    def schedule(self, minutes: int) -> tuple[str, str]:
        """Ответ `faq_watch_schedule` и строка для лога (§3.7): интервал
        сохраняется в базе, планировщик просыпается и пересчитывает срок."""
        previous = self.interval
        self.save_interval(minutes)
        self.interval_origin = ORIGIN_DB
        if self.wake is not None:
            self.wake.set()
        with closing(_connect(self.db)) as conn:
            ok_run, _ = self.attempts(conn)
            due = self.next_due(conn)
        last = (
            f"Последний удачный снимок — {_fmt(ok_run.started_at)}"
            if ok_run else "Удачных снимков ещё нет"
        )
        if self.running is not None:
            upcoming = (
                f"снимок #{self.running.number} идёт сейчас; следующий — по новому "
                f"расписанию после него"
            )
        elif due <= _now():
            upcoming = "срок по новому расписанию уже прошёл — снимок начинается сейчас"
        else:
            upcoming = f"следующий — {_fmt(due)}"
        text = f"Расписание: раз в {minutes} мин (было {previous}). {last}, {upcoming}."
        return text, f"было {previous}; следующий снимок {self._due_text(due)}"

    # --- памятка к столу (день 19) ---

    def search(self, query: str, limit: int) -> tuple[str, str]:
        """Ответ `faq_search` и строка для лога (§2.5, §3.3): ищет в
        последнем удачном снимке — сеть не трогает, результат воспроизводим."""
        with closing(_connect(self.db)) as conn:
            ok_run, _ = self.attempts(conn)
            if ok_run is None:
                raise ToolError(NO_SNAPSHOT_ERROR)
            rows = conn.execute(
                "SELECT a.article_id, a.section, a.question, t.text FROM articles a "
                "JOIN texts t ON t.hash = a.text_hash WHERE a.run_id = ? ORDER BY a.rowid",
                (ok_run.id,),
            ).fetchall()
            total = len(rows)
            query_words = _search_words(query)
            scored: list[tuple[int, dict[str, list[str]], sqlite3.Row]] = []
            for row in rows:
                question_words = _search_words(row["question"])
                text_words = _search_words(row["text"])
                score, hits = _score_article(query_words, question_words, text_words)
                if score > 0:
                    scored.append((score, hits, row))
            # Стабильная сортировка: при равной оценке порядок остаётся тем,
            # в каком статьи лежат в снимке — «как на сайте» (§2.5).
            scored.sort(key=lambda item: -item[0])
            top = scored[:limit]
            article_ids = [row["article_id"] for _, _, row in top]
            with _transaction(conn):
                conn.execute(
                    "INSERT INTO searches (id, created_at, run_id, query, limit_n, article_ids) "
                    "VALUES (NULL, ?, ?, ?, ?, ?)",
                    (_iso(_now()), ok_run.id, query, limit, json.dumps(article_ids)),
                )
                search_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        ref = f"{SEARCH_PREFIX}{search_id}"
        lines = [f"id: {ref}"]
        if not top:
            lines.append(
                f"Поиск «{query}» в снимке #{ok_run.id} ({_fmt(ok_run.started_at)}): "
                f"ничего не найдено среди {total}; попробуй другие английские слова."
            )
            return "\n".join(lines), f"{ref} — 0 из {total}"
        lines.append(
            f"Поиск «{query}» в снимке #{ok_run.id} ({_fmt(ok_run.started_at)}): "
            f"{_articles_word(len(top))} из {total}."
        )
        lines.append("Строка: номер · раздел · вопрос · совпало.")
        for _, hits, row in top:
            matched = ", ".join(
                f"{word} ({', '.join(sorted(set(where)))})" for word, where in hits.items()
            )
            lines.append(f"{row['article_id']} · {row['section']} · {row['question']} · {matched}")
        lines.append(f"Для памятки передай id {ref} в faq_summarize.")
        return "\n".join(lines), f"{ref} — {len(top)} из {total} (снимок #{ok_run.id})"

    def _load_search(self, conn: sqlite3.Connection, search_id: int) -> SearchRecord:
        row = conn.execute("SELECT * FROM searches WHERE id = ?", (search_id,)).fetchone()
        if row is None:
            raise ToolError(f"поиска {SEARCH_PREFIX}{search_id} нет")
        return SearchRecord(
            id=row["id"], query=row["query"], limit=row["limit_n"], run_id=row["run_id"],
            article_ids=json.loads(row["article_ids"]),
        )

    def _search_articles(
        self, conn: sqlite3.Connection, record: SearchRecord,
    ) -> list[tuple[str, str, str, str]]:
        """Тексты статей поиска — из того же снимка, на котором искали
        (§2.3): снимка нет или в нём нет статьи — одна и та же ошибка,
        «повтори поиск» (снимки не удаляются, поэтому в обычном прогоне она
        не встречается)."""
        missing = ToolError(
            f"снимка #{record.run_id}, на котором искали {SEARCH_PREFIX}{record.id}, "
            f"в базе нет — повтори поиск"
        )
        run = conn.execute("SELECT id FROM runs WHERE id = ? AND ok = 1", (record.run_id,)).fetchone()
        if run is None:
            raise missing
        found: dict[str, tuple[str, str, str]] = {}
        for row in conn.execute(
            "SELECT a.article_id, a.section, a.question, t.text FROM articles a "
            "JOIN texts t ON t.hash = a.text_hash WHERE a.run_id = ?", (record.run_id,),
        ):
            found[row["article_id"]] = (row["section"], row["question"], row["text"])
        articles: list[tuple[str, str, str, str]] = []
        for article_id in record.article_ids:
            info = found.get(article_id)
            if info is None:
                raise missing
            section, question, text = info
            articles.append((article_id, section, question, text))
        return articles

    def summary_source(
        self, search_id_raw: str,
    ) -> tuple[SearchRecord, list[tuple[str, str, str, str]]]:
        """Данные для сжатия по id поиска (§3.4): проверки входа — здесь, до
        просьбы к модели клиента, чтобы неверный id не стоил вызова."""
        search_id = _require_search_ref(search_id_raw)
        with closing(_connect(self.db)) as conn:
            record = self._load_search(conn, search_id)
            articles = self._search_articles(conn, record)
        if not articles:
            raise ToolError(f"поиск {SEARCH_PREFIX}{record.id} ничего не нашёл — сжимать нечего")
        return record, articles

    def summarize(
        self, search_id_raw: str, topic: str, result: CreateMessageResult,
    ) -> tuple[str, str]:
        """Ответ `faq_summarize` (§3.3): разбирает ответ модели клиента,
        проверяет ссылки на статьи этого поиска, пишет памятку одной строкой.
        Оборванный по лимиту ответ не записывается — правило трекера дня 13,
        оборванный результат хуже отсутствия."""
        record, articles = self.summary_source(search_id_raw)
        if result.stop_reason == "maxTokens":
            raise ToolError("памятка оборвана по лимиту — сократи limit поиска")
        content = result.content
        text = (content.text if isinstance(content, TextContent) else "").strip()
        if not text:
            raise ToolError("модель клиента вернула не памятку: (пустой ответ)")
        items = _summary_items(text)
        if not items and not _is_no_answer(text):
            raise ToolError(f"модель клиента вернула не памятку: {text[:200]}")
        article_ids = {article_id for article_id, *_ in articles}
        foreign = sorted({m for m in re.findall(r"\[(\d+)\]", text) if m not in article_ids})
        model = result.model or "?"
        digest = text_hash(text)
        with closing(_connect(self.db)) as conn, _transaction(conn):
            conn.execute(
                "INSERT INTO summaries (id, created_at, search_id, topic, text, text_hash, "
                "model, items, foreign_refs) VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?)",
                (_iso(_now()), record.id, topic, text, digest, model, len(items), json.dumps(foreign)),
            )
            summary_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        ref = f"{SUMMARY_PREFIX}{summary_id}"
        search_ref = f"{SEARCH_PREFIX}{record.id}"
        lines = [
            f"id: {ref}",
            f"Памятка «{topic}» из поиска {search_ref} ({_articles_word(len(articles))}, "
            f"снимок #{record.run_id}), модель клиента {model}, {len(items)} "
            f"{_plural(len(items), 'пункт', 'пункта', 'пунктов')}.",
            text,
            f"Чтобы сохранить, передай id {ref} в cheatsheet_save.",
        ]
        if foreign:
            lines.append("⚠️ ссылки вне поиска: " + ", ".join(foreign))
        # Время в строке лога — только тело инструмента: раунды — разные
        # HTTP-запросы, и сервер без состояния не знает, когда начался первый;
        # время сэмплинга — в логах клиента и агента (§3.8).
        summary = (
            f"{ref} — {_count(len(items), 'пункт', 'пункта', 'пунктов')}, модель {model}, "
            f"без сэмплинга"
        )
        return "\n".join(lines), summary

    def _load_summary(self, conn: sqlite3.Connection, summary_id: int) -> sqlite3.Row:
        row = conn.execute("SELECT * FROM summaries WHERE id = ?", (summary_id,)).fetchone()
        if row is None:
            raise ToolError(f"памятки {SUMMARY_PREFIX}{summary_id} нет")
        return row

    def _unique_cheatsheet_path(self, stem: str) -> Path:
        candidate = self.out_dir / f"{stem}.md"
        number = 2
        while candidate.exists():
            candidate = self.out_dir / f"{stem}-{number}.md"
            number += 1
        return candidate

    def _cheatsheet_content(
        self, topic: str, text: str, record: SearchRecord, run: Run,
        articles: list[tuple[str, str, str, str]], summary_ref: str, model: str,
        summary_hash: str, search_ref: str,
    ) -> str:
        """Файл памятки (§2.6): заголовок, источники и строка происхождения
        пишет код, из базы — модель пишет только пункты (уже в `text`)."""
        lines = [f"# Памятка: {topic}", "", text, "", "---", ""]
        lines.append(
            f"Источники — официальный FAQ издателя «{self.config.category_name}», снимок "
            f"#{record.run_id} от {_fmt(run.started_at)}:"
        )
        for article_id, section, question, _ in articles:
            lines.append(f"- [{article_id}]({self._article_url(article_id)}) · {section} · {question}")
        lines.append("")
        lines.append(
            f"Собрано {_fmt(_now())} · поиск {search_ref} («{record.query}», "
            f"{_articles_word(len(articles))}) → памятка {summary_ref} (модель {model}, "
            f"sha {summary_hash[:12]}…) → этот файл"
        )
        return "\n".join(lines)

    def save_cheatsheet(self, summary_id_raw: str, name: str) -> tuple[str, str]:
        """Ответ `cheatsheet_save` (§3.3, §3.5): хеш памятки сверяется с
        записанным при сжатии, файл пишется атомарно внутри `--out`,
        перечитывается и сверяется байт в байт."""
        summary_id = _require_summary_ref(summary_id_raw)
        with closing(_connect(self.db)) as conn:
            summary_row = self._load_summary(conn, summary_id)
            text = summary_row["text"]
            if text_hash(text) != summary_row["text_hash"]:
                raise ToolError(
                    f"памятка {SUMMARY_PREFIX}{summary_id} изменена после сжатия — сохранять не буду"
                )
            record = self._load_search(conn, summary_row["search_id"])
            articles = self._search_articles(conn, record)
            run_row = conn.execute("SELECT * FROM runs WHERE id = ?", (record.run_id,)).fetchone()
            run = _run(run_row)

        topic = summary_row["topic"]
        model = summary_row["model"]
        items = summary_row["items"]
        summary_ref = f"{SUMMARY_PREFIX}{summary_id}"
        search_ref = f"{SEARCH_PREFIX}{record.id}"

        self.out_dir.mkdir(parents=True, exist_ok=True)
        stem = f"{_now().strftime(CHEATSHEET_DATE_FORMAT)}-{_clean_cheatsheet_name(name, summary_id)}"
        path = self._unique_cheatsheet_path(stem)
        # Проверка ещё раз, что путь остался внутри `--out` (§2.6): очистка
        # имени уже не оставляет в нём «/» и «..», это защита сверх неё.
        if self.out_dir.resolve() not in path.resolve().parents:
            raise ToolError("путь файла вышел за пределы каталога памяток — сохранение отменено")

        content = self._cheatsheet_content(
            topic, text, record, run, articles, summary_ref, model,
            summary_row["text_hash"], search_ref,
        )
        tmp = path.with_name(path.name + f".tmp{os.getpid()}")
        try:
            tmp.write_text(content, encoding="utf-8")
            os.replace(tmp, path)
        except OSError as exc:
            if tmp.exists():
                tmp.unlink()
            raise ToolError(f"файл не записался: {type(exc).__name__}: {exc}") from exc

        written = path.read_text(encoding="utf-8")
        if text not in written:
            path.unlink(missing_ok=True)
            raise ToolError("файл записан, но текст памятки в нём не совпадает — файл удалён")

        file_bytes = len(written.encode("utf-8"))
        file_digest = text_hash(written)
        with closing(_connect(self.db)) as conn, _transaction(conn):
            conn.execute(
                "INSERT INTO saves (id, created_at, summary_id, path, file_hash, bytes) "
                "VALUES (NULL, ?, ?, ?, ?, ?)",
                (_iso(_now()), summary_id, str(path), file_digest, file_bytes),
            )

        shown_path = display_path(path)
        lines = [
            f"файл: {shown_path}",
            f"Памятка {summary_ref} «{topic}» сохранена: {_bytes_word(file_bytes)}, {items} "
            f"{_plural(items, 'пункт', 'пункта', 'пунктов')}, {len(articles)} "
            f"{_plural(len(articles), 'источник', 'источника', 'источников')}.",
            f"Проверка: файл перечитан, текст памятки {summary_ref} записан без изменений "
            f"(sha {summary_row['text_hash'][:12]}… совпадает с записанным при сжатии).",
            f"Цепочка: поиск {search_ref} («{record.query}», снимок #{record.run_id}) → "
            f"памятка {summary_ref} → файл.",
        ]
        return "\n".join(lines), f"{shown_path}, {_bytes_word(file_bytes)}, sha совпадает"

    def start_line(self, port: int) -> str:
        with closing(_connect(self.db)) as conn:
            total = conn.execute("SELECT count(*) FROM runs").fetchone()[0]
            ok_run, _ = self.attempts(conn)
            due = self.next_due(conn)
        last = f", последний удачный {_fmt(ok_run.started_at)}" if ok_run else ", удачных нет"
        return (
            f"слушаю http://{HOST}:{port}/mcp · база {display_path(self.db)} "
            f"(снимков: {total}{last}) · раз в {self.interval} мин ({self.interval_origin}) · "
            f"следующий снимок {'сразу' if due <= _now() else _fmt(due)}"
        )


# --- Сервер ----------------------------------------------------------------

def build_server(watch: Watch) -> MCPServer:
    @asynccontextmanager
    async def scheduler_lifespan(_server: MCPServer):
        # Вход — task group с планировщиком, выход (Ctrl+C) — её отмена.
        # Снимок, прерванный отменой, в базу не попадает (§3.3).
        async with anyio.create_task_group() as group:
            group.start_soon(watch.scheduler)
            try:
                yield None
            finally:
                group.cancel_scope.cancel()

    # `log_level="WARNING"` — и SDK, и uvicorn: строки INFO каждого
    # HTTP-запроса — шум; свою строку «слушаю …» сервер пишет сам.
    server = MCPServer(
        name=SERVER_NAME, version=SERVER_VERSION, log_level="WARNING",
        lifespan=scheduler_lifespan,
    )
    game_name = watch.config.category_name

    @server.tool(
        title="Сводка изменений FAQ",
        description=CHANGES_DESCRIPTION.format(name=game_name),
        annotations=CHANGES_ANNOTATIONS,
        structured_output=False,
    )
    async def faq_changes(
        days: Annotated[int, Field(ge=1, le=DAYS_MAX, description=DAYS_DESCRIPTION)] = DAYS_DEFAULT,
    ) -> str:
        return _logged(f"faq_changes(days={days})", lambda: watch.changes(days))

    @server.tool(
        title="Расписание сторожа FAQ",
        description=SCHEDULE_DESCRIPTION.format(name=game_name),
        annotations=SCHEDULE_ANNOTATIONS,
        structured_output=False,
    )
    async def faq_watch_schedule(
        interval_minutes: Annotated[
            int, Field(ge=MIN_INTERVAL_MIN, le=MAX_INTERVAL_MIN, description=INTERVAL_DESCRIPTION)
        ],
    ) -> str:
        return _logged(
            f"faq_watch_schedule({interval_minutes})", lambda: watch.schedule(interval_minutes)
        )

    @server.tool(
        title="Поиск статей для памятки",
        description=SEARCH_DESCRIPTION.format(name=game_name),
        annotations=SEARCH_ANNOTATIONS,
        structured_output=False,
    )
    async def faq_search(
        query: Annotated[str, Field(min_length=1, max_length=200, description=QUERY_DESCRIPTION)],
        limit: Annotated[
            int, Field(ge=1, le=SEARCH_LIMIT_MAX, description=LIMIT_DESCRIPTION)
        ] = SEARCH_LIMIT_DEFAULT,
    ) -> str:
        return _logged(f'faq_search("{query}", {limit})', lambda: watch.search(query, limit))

    def summary_request(search_id: str, topic: str, ctx: Context) -> Sample:
        # Детерминированная функция (§3.4): SDK выполняет её на каждом раунде
        # просьбы, и просьба должна совпадать — всё берётся из базы по
        # search_id, без времени и случайности. Проверки входа — здесь, до
        # просьбы к модели клиента: неверный id не должен стоить вызова.
        # Отказ — строкой лога, как у тел инструментов в `_logged()`: это
        # проверка передачи данных, её должно быть видно в терминале сервера
        # (§3.8, правка по ревью). Отказ бывает только на первом раунде —
        # просьбы нет, второго раунда нет, строка не повторяется.
        try:
            record, articles = watch.summary_source(search_id)
        except ToolError as exc:
            logger.info("faq_summarize(%s): отказ — %s", search_id, exc)
            raise
        except sqlite3.Error as exc:
            logger.warning("faq_summarize(%s): отказ — база сторожа: %s", search_id, exc)
            raise ToolError(f"база сторожа не читается: {type(exc).__name__}: {exc}") from exc
        user_text = _summary_user_text(topic, articles)
        # Строка «просьба к модели клиента» — только на первом раунде
        # (`ctx.input_responses is None`): на повторе функция выполняется
        # снова, но тело инструмента — только после ответа (§3.4, §3.8).
        if ctx.input_responses is None:
            logger.info(
                "faq_summarize(%s, «%s»): просьба к модели клиента — %s, %d символов, "
                "max_tokens %d",
                f"{SEARCH_PREFIX}{record.id}", topic, _articles_word(len(articles)),
                len(user_text), SUMMARY_MAX_TOKENS,
            )
        return Sample(
            [SamplingMessage(role="user", content=TextContent(type="text", text=user_text))],
            max_tokens=SUMMARY_MAX_TOKENS,
            system_prompt=SUMMARY_PROMPT.format(name=watch.config.category_name),
        )

    @server.tool(
        title="Сжатие статей в памятку",
        description=SUMMARIZE_DESCRIPTION.format(name=game_name),
        annotations=SUMMARIZE_ANNOTATIONS,
        structured_output=False,
    )
    async def faq_summarize(
        search_id: Annotated[
            str, Field(min_length=1, max_length=20, description=SEARCH_ID_DESCRIPTION)
        ],
        topic: Annotated[str, Field(min_length=1, max_length=100, description=TOPIC_DESCRIPTION)],
        summary: Annotated[CreateMessageResult, Resolve(summary_request)],
    ) -> str:
        return _logged(
            f"faq_summarize({search_id})", lambda: watch.summarize(search_id, topic, summary)
        )

    @server.tool(
        title="Сохранение памятки в файл",
        description=SAVE_DESCRIPTION.format(name=game_name),
        annotations=SAVE_ANNOTATIONS,
        structured_output=False,
    )
    async def cheatsheet_save(
        summary_id: Annotated[
            str, Field(min_length=1, max_length=20, description=SUMMARY_ID_DESCRIPTION)
        ],
        name: Annotated[str, Field(max_length=60, description=NAME_DESCRIPTION)] = "",
    ) -> str:
        return _logged(
            f"cheatsheet_save({summary_id})", lambda: watch.save_cheatsheet(summary_id, name)
        )

    return server


def _logged(call: str, work) -> str:
    """Строка лога на вызов. `ToolError` — отказ инструмента (§3.3): текст
    уходит клиенту как есть, здесь — только строка лога. Ошибка базы —
    `ToolError` с текстом (клиент увидит `isError`); любое другое исключение
    — ошибка в коде, трассировку пишет SDK."""
    started = time.perf_counter()
    try:
        text, summary = work()
    except ToolError as exc:
        logger.info("%s: отказ — %s, %.2f с", call, exc, time.perf_counter() - started)
        raise
    except sqlite3.Error as exc:
        logger.warning("%s: отказ — база сторожа: %s", call, exc)
        raise ToolError(f"база сторожа не читается: {type(exc).__name__}: {exc}") from exc
    logger.info("%s: %s, %.2f с", call, summary, time.perf_counter() - started)
    return text


def _interval(value: str) -> int:
    if not value.isdigit() or not MIN_INTERVAL_MIN <= int(value) <= MAX_INTERVAL_MIN:
        raise argparse.ArgumentTypeError(
            f"ожидается целое число минут от {MIN_INTERVAL_MIN} до {MAX_INTERVAL_MIN}, "
            f"пришло «{value}»"
        )
    return int(value)


def _args(argv: list[str] | None = None) -> tuple[Config, Path, Path, int, int | None]:
    parser = argparse.ArgumentParser(
        description=(
            "Сторож FAQ: MCP-сервер (Streamable HTTP) с планировщиком снимков одной "
            "категории FAQ портала Freshdesk."
        ),
    )
    parser.add_argument("--base-url", required=True, help="адрес портала без завершающего /")
    parser.add_argument("--category", required=True, help="id категории — строка цифр")
    parser.add_argument(
        "--category-name", required=True,
        help="ожидаемое имя категории: сверяется с заголовком её страницы",
    )
    parser.add_argument("--db", required=True, help="путь к файлу базы; каталог создаётся")
    parser.add_argument(
        "--out", required=True, help="каталог для файлов памяток (день 19); создаётся, если его нет",
    )
    parser.add_argument(
        "--port", type=int, default=DEFAULT_PORT,
        help=f"порт на {HOST}, по умолчанию {DEFAULT_PORT}",
    )
    parser.add_argument(
        "--interval", type=_interval, default=None,
        help=(
            f"интервал снимков в минутах, {MIN_INTERVAL_MIN}-{MAX_INTERVAL_MIN}; задан — "
            f"перекрывает сохранённый в базе и сохраняется"
        ),
    )
    args = parser.parse_args(argv)
    if not args.category.isdigit():
        parser.error(f"--category: ожидается строка цифр, пришло «{args.category}»")
    if not 1 <= args.port <= 65535:
        parser.error(f"--port: ожидается 1-65535, пришло {args.port}")
    config = Config(
        base_url=args.base_url.rstrip("/"),
        category=args.category,
        category_name=args.category_name,
    )
    return (
        config, Path(args.db).expanduser(), Path(args.out).expanduser(), args.port, args.interval,
    )


def _port_free(port: int) -> str:
    """Пустая строка — порт свободен; иначе причина. Проверка до базы: занятый
    порт не должен оставить в ней следов (uvicorn сообщил бы о порте уже
    после входа в lifespan — после старта планировщика)."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as probe:
        # Как у uvicorn: сокет в TIME_WAIT после перезапуска — не занятость.
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((HOST, port))
        except OSError as exc:
            return str(exc)
    return ""


def main() -> None:
    config, db, out_dir, port, interval = _args()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[FAQ-сторож] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # Строки httpx «HTTP Request: GET …» — шум; строки `faq_server` на каждый
    # запрос к сайту сторожу тоже не нужны (89 на снимок).
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        version = check_db(db)
    except DbError as exc:
        logger.error("база %s: %s — сервер не запущен, файл не изменён", display_path(db), exc)
        sys.exit(2)
    busy = _port_free(port)
    if busy:
        logger.error("порт %s:%d занят (%s) — сервер не запущен", HOST, port, busy)
        sys.exit(1)
    init_db(db, version)
    out_dir.mkdir(parents=True, exist_ok=True)

    watch = Watch(config, db, out_dir)
    watch.load_interval(interval)
    logger.info("%s", watch.start_line(port))
    try:
        build_server(watch).run("streamable-http", host=HOST, port=port, stateless_http=True)
    except KeyboardInterrupt:
        pass
    logger.info("остановлен")


if __name__ == "__main__":
    main()
