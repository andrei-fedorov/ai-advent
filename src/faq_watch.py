# TooManyRules — сторож FAQ: свой долгоживущий MCP-сервер по Streamable HTTP
# (день 18, неделя 4; день 19 — пайплайн памятки к столу: поиск, сжатие
# моделью клиента через MCP sampling; день 20 — «FAQ издателя: копия и
# сайт»).
#
# **Отдельная программа, а не модуль приложения** (спецификация дня 18, §3.1),
# как `faq_server.py`, но живёт долго: запускает её человек (`./run.sh
# faq-watch`, командная строка — `presets.FAQ_WATCH_ARGV`) или, с дня 20,
# приложение, если порт молчит (`mcp_client.ensure_started()`). Приложение её
# не импортирует, в графе зависимостей приложения её нет. Сама она
# импортирует `faq_server.py` — чтение сайта и разбор страниц; это ребро
# между двумя серверами, вне графа приложения. Серверную часть SDK `mcp`
# импортируют программы-серверы (`faq_server.py`, эта, с дня 20 —
# `wiki_server.py` и `file_server.py`); клиентскую — только `mcp_client.py`.
#
# Что делает (§2.1; как изменил день 20 — ниже):
# 1. снимает FAQ целиком — список вопросов и тексты всех статей — и сохраняет
#    снимок в SQLite: на дне 18 по расписанию, с дня 20 — по запросу, когда
#    копия пуста или устарела, а по расписанию — только если его включили;
# 2. сравнивает снимок с прошлым удачным и после каждого снимка сам пишет
#    сводку в свой терминал (stderr) и события в базу;
# 3. по вызову `faq_changes` отдаёт агрегированную сводку за период — из базы,
#    без запросов к сайту;
# 4. по вызову `faq_watch_schedule` меняет расписание: ответ сразу, снимок по
#    новому расписанию — потом, в фоне;
# 5. (день 19) `faq_search` находит статьи в последнем снимке по английским
#    ключевым словам, `faq_summarize` сжимает найденное в памятку по-русски
#    (модель клиента, через MCP sampling); сохранение в файл с дня 20 — на
#    сервере файлов. Между поиском и сжатием данные передаются по ссылке — id
#    `q…`, а не текстом: сжатие само достаёт из базы то, что записал поиск, и
#    проверяет, что переданный id — того вида и с той же базы (спецификация
#    дня 19, §2.3).
#
# Сервер знает портал Freshdesk, но не конкретную игру: адрес портала, id
# категории, её имя и путь к базе приходят аргументами командной строки.
#
# **День 20 (спецификация дня 20, §3)** — «FAQ издателя: копия и сайт».
# Сервер — единственный источник FAQ издателя у агента, и FAQ доступен всегда:
# 1. **копия** — последний удачный снимок в SQLite. Хорошая (удачный снимок
#    не старше `--max-age-min`) — `faq_search` и новый `faq_article` отвечают
#    из неё; плохая (пустая или устаревшая) — **с сайта**, а копия
#    **докачивается в фоне**: инструмент ставит флаг «нужен снимок», будит
#    исполнитель снимков и снимка не ждёт. Выбор «копия или сайт» делает код,
#    а не модель: откуда данные, видно по строке `источник:` ответа;
# 2. планировщик дня 18 стал **исполнителем снимков** с двумя источниками
#    работы — запрос снимка и расписание; **расписание выключено по
#    умолчанию** (`schedule_enabled` в `settings`), включают его `--interval`
#    или `faq_watch_schedule(N)`, выключает `faq_watch_schedule(0)`;
# 3. сервер обычно поднимает приложение (`mcp_client.ensure_started()`),
#    если порт молчит, с `--parent-pid`: родитель сменился — сервер
#    останавливается сам, как по Ctrl+C. Ручной запуск (`./run.sh faq-watch`)
#    остаётся, аргумента `--parent-pid` у него нет;
# 4. `cheatsheet_save` и `--out` ушли на сервер файлов (`file_server.py`,
#    `save_markdown`): `faq_summarize` теперь отдаёт ещё и готовый Markdown
#    памятки — данные в сервер файлов идут по значению. Таблица `saves`
#    остаётся историей дня 19, новых строк в неё нет;
# 5. база — версия 3: поиск по сайту записывается вместе с прочитанными
#    текстами (`search_articles`), чтобы `faq_summarize` работал по ссылке `q…`
#    тем же путём, что по копии.
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
# её не открывает. Файлов, кроме базы, сервер с дня 20 не пишет.
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
import signal
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
SERVER_VERSION = "1.2.0"

# --- Сеть ------------------------------------------------------------------
# Адрес — только 127.0.0.1, аргумента хоста нет (§15): наружу сервер не
# открывается. Путь `/mcp` — по умолчанию SDK.
HOST = "127.0.0.1"
DEFAULT_PORT = 8765

# --- Расписание (§2.4; день 20 — §3.3) -------------------------------------
# Нижняя граница — вежливость к чужому сайту: полный снимок — 89 запросов.
# С дня 20 расписание выключено по умолчанию; интервал по умолчанию — только
# на случай, когда базу включили без явного интервала.
DEFAULT_INTERVAL_MIN = 1440
MIN_INTERVAL_MIN = 5
MAX_INTERVAL_MIN = 10080
# После неудачной попытки — повтор через `min(интервал, RETRY_MIN)`. При
# выключенном расписании интервалом в этой формуле считается `RETRY_MIN`
# (день 20, §2.5): иначе недоступный сайт опрашивался бы на каждом вопросе.
RETRY_MIN = 15
# Исполнитель снимков спит не дольше этого и каждый раз заново сверяет
# стенные часы со сроком: на ноутбуке, который засыпал, монотонные часы во
# сне стоят. С той же частотой сервер сверяет родителя (`--parent-pid`).
SCHEDULER_TICK_S = 60

# --- Копия FAQ (день 20, §3.2) ---------------------------------------------
# Копия хорошая, если последний удачный снимок начат не раньше этого числа
# минут назад. Минуты, а не дни: для видео нужна копия, устаревшая за пару
# минут (`--max-age-min 2`).
DEFAULT_MAX_AGE_MIN = 10080

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

# --- База (§3.9, день 19 §3.7, день 20 §3.7) ---------------------------------
# Версия схемы — `PRAGMA user_version`. 0 у пустой базы: схема создаётся
# целиком; 1 — база дня 18: добавляются таблицы дня 19 и дня 20; 2 — база дня
# 19: `searches` пересобирается (у поиска по сайту нет снимка — `run_id`
# допускает NULL, новая колонка `source`), добавляется `search_articles`;
# своя версия (3) — работаем; любая другая — чужая база, сервер не стартует.
# Миграция односторонняя: сервер дня 19 базу версии 3 не откроет (честный
# отказ дня 18).
SCHEMA_VERSION = 3
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
# Таблицы дня 19 (§3.7) — пайплайн памятки: `summaries` хранит сжатую памятку
# и её хеш, `saves` — куда и с каким хешем файла она легла (с дня 20 —
# история: сохраняет сервер файлов).
SCHEMA_V19 = (
    """CREATE TABLE summaries (
        id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, search_id INTEGER NOT NULL REFERENCES searches(id),
        topic TEXT NOT NULL, text TEXT NOT NULL, text_hash TEXT NOT NULL, model TEXT NOT NULL,
        items INTEGER NOT NULL, foreign_refs TEXT NOT NULL DEFAULT '')""",
    """CREATE TABLE saves (
        id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, summary_id INTEGER NOT NULL REFERENCES summaries(id),
        path TEXT NOT NULL, file_hash TEXT NOT NULL, bytes INTEGER NOT NULL)""",
)
# `searches` в редакции дня 20 (§3.7): что и где нашлось. `article_ids` —
# JSON-список в порядке поиска; у поиска по копии — ссылка на снимок
# (`run_id`), у поиска по сайту `run_id` — NULL, а прочитанные тексты лежат в
# `search_articles` (тексты — в `texts` по хешу, как у снимков).
SEARCHES_TABLE = """CREATE TABLE {name} (
    id INTEGER PRIMARY KEY, created_at TEXT NOT NULL, run_id INTEGER REFERENCES runs(id),
    source TEXT NOT NULL DEFAULT 'копия', query TEXT NOT NULL, limit_n INTEGER NOT NULL,
    article_ids TEXT NOT NULL)"""
SEARCH_ARTICLES_TABLE = """CREATE TABLE search_articles (
    search_id INTEGER NOT NULL REFERENCES searches(id), position INTEGER NOT NULL,
    article_id TEXT NOT NULL, section TEXT NOT NULL, question TEXT NOT NULL,
    url TEXT NOT NULL, text_hash TEXT NOT NULL REFERENCES texts(hash),
    PRIMARY KEY (search_id, position))"""
# Миграция 2 → 3: `ALTER TABLE` в SQLite `NOT NULL` не снимает, поэтому
# `searches` пересобирается — новая таблица, копия строк с `source='копия'`,
# старая удаляется, новая переименовывается.
MIGRATE_V2_V3 = (
    SEARCHES_TABLE.format(name="searches_v3"),
    "INSERT INTO searches_v3 (id, created_at, run_id, source, query, limit_n, article_ids) "
    "SELECT id, created_at, run_id, 'копия', query, limit_n, article_ids FROM searches",
    "DROP TABLE searches",
    "ALTER TABLE searches_v3 RENAME TO searches",
    SEARCH_ARTICLES_TABLE,
)
SETTING_INTERVAL = "interval_minutes"
# Включено ли расписание (день 20, §3.3): "1" / "0". Нет ключа — выключено,
# даже если интервал сохранён: иначе у автора после обновления продолжились
# бы снимки по старому расписанию дней 18-19.
SETTING_SCHEDULE = "schedule_enabled"

SOURCE_COPY = "копия"
SOURCE_SITE = "сайт"

# --- Описания инструментов (день 20, §3.6) ----------------------------------
# Описание инструмента и есть промпт (правило дня 17). Черновик спецификации
# дня 20; каждое изменение по сбою живого прогона — строкой в комментарии у
# константы. Упоминаний `faq_questions` дня 17 больше нет: этого инструмента у
# агента нет.
#
# Правка `SEARCH_DESCRIPTION` по живому прогону (26.09.2026, `deepseek-flash`,
# сценарий О1-О6 одним диалогом): после длинного О2 модель ответила на О4 и
# О5 без вызовов — по истории и из общих знаний (О5 — неверно), хотя на
# свежем агенте те же вопросы шли в FAQ. Добавлено «Вызывай перед каждым
# ответом … даже если ответ кажется известным или уже звучал в разговоре» —
# урок дня 17 (`QUESTIONS_DESCRIPTION` в `faq_server.py`).
SEARCH_DESCRIPTION = (
    "Официальный FAQ издателя по игре {name} — разъяснения и эррата от авторов игры, на "
    "английском. Первый шаг для любого вопроса о правилах, спорного случая или эрраты: поиск "
    "статей по английским ключевым словам (тему игрока переведи сам). Вызывай перед каждым "
    "ответом на вопрос о правилах — даже если ответ кажется известным или уже звучал в "
    "разговоре: FAQ разбирает как раз частые ошибки. Ищет в копии FAQ, а если "
    "копия не готова — на сайте; откуда ответ, сказано в строке «источник». Возвращает id "
    "поиска (q…) и статьи. Текст статьи — faq_article(номер); памятку по найденному — "
    "faq_summarize(id поиска). При расхождении с другими источниками прав FAQ издателя."
)
QUERY_DESCRIPTION = (
    "Английские ключевые слова через пробел: «poison», «tink bots». Ищутся по началу слова "
    "в вопросе и тексте статьи."
)
ARTICLE_DESCRIPTION = (
    "Текст одной статьи официального FAQ издателя по игре {name} по номеру из faq_search: "
    "вопрос, раздел, ответ издателя и ссылка. Отвечая игроку, перескажи по-русски и дай ссылку."
)
ARTICLE_ID_DESCRIPTION = "Номер статьи из ответа faq_search: 33000210161."
SUMMARIZE_DESCRIPTION = (
    "Памятка к столу по одному поиску FAQ: сжимает статьи из результата faq_search в пункты "
    "по-русски со ссылками. Принимает только id поиска (q…) — тексты сервер берёт сам. "
    "Сжатие делает модель клиента по просьбе сервера. Возвращает id памятки (s…), пункты и "
    "готовый Markdown — чтобы сохранить, передай его в save_markdown как есть."
)
SEARCH_ID_DESCRIPTION = "id поиска из первой строки ответа faq_search: q7."
TOPIC_DESCRIPTION = "Тема памятки по-русски — станет её заголовком: «Яд (Poison)»."
CHANGES_DESCRIPTION = (
    "Что изменилось в официальном FAQ издателя по игре {name}: что сервер заметил, сравнивая "
    "копии FAQ при обновлениях (новые, изменённые, перенесённые, удалённые статьи), какие "
    "статьи по датам сайта изменены за период, и состояние копии. Вызывай, когда игрок "
    "спрашивает, что нового в FAQ или эррате. Для вопросов о самих правилах — faq_search."
)
DAYS_DESCRIPTION = "За сколько последних дней: от 1 до 365. «За месяц» — 30, «за неделю» — 7."
SCHEDULE_DESCRIPTION = (
    "Включает или выключает регулярную проверку FAQ по игре {name}: интервал в минутах от 5 "
    "до 10080, 0 — выключить. Без расписания копия FAQ обновляется сама, когда нужна и "
    "устарела. Вызывай только по прямой просьбе игрока проверять FAQ регулярно или перестать."
)
INTERVAL_DESCRIPTION = "Минуты между проверками: 60 — раз в час, 1440 — раз в сутки, 0 — выключить."

# Аннотации честные (§3.1). Сводка только читает базу; расписание пишет, но не
# разрушает, и повтор с тем же значением ничего не меняет. Поиск и статья с
# дня 20 могут читать сайт — мир открытый; статья только читает (запрос
# снимка — побочный эффект сервера, а не результат инструмента), поиск и
# сжатие пишут записи в базу.
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
SEARCH_ANNOTATIONS = ToolAnnotations(
    read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=True,
)
ARTICLE_ANNOTATIONS = ToolAnnotations(
    read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True,
)
SUMMARIZE_ANNOTATIONS = ToolAnnotations(
    read_only_hint=False, destructive_hint=False, idempotent_hint=False, open_world_hint=False,
)

# --- Памятка к столу (день 19) ----------------------------------------------
# Ссылки между шагами пайплайна — id внутри базы сервера, не тексты (§2.3):
# `q…` — результат `faq_search`, `s…` — результат `faq_summarize`. Модель
# передаёт id следующему шагу, шаг сам достаёт данные из базы.
SEARCH_PREFIX = "q"
SUMMARY_PREFIX = "s"

# Поиск по словам (§2.5): токены — `[a-z0-9']+` в нижнем регистре, от 3 букв,
# без короткого списка служебных английских слов (черновик; правки по
# прогону — строкой в комментарии). Слово запроса совпадает, если оно —
# начало слова статьи («poison» находит «poisoned»). Поиск по сайту (день 20,
# §3.4) сравнивает слова только с вопросами — тексты статей ещё не прочитаны.
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
LIMIT_DESCRIPTION = f"Сколько статей взять, 1-{SEARCH_LIMIT_MAX}."
# Ответ модели клиента без пунктов, но по теме (§3.6) — единственный
# допустимый ответ без «- »: сервер про игру не знает и промпт запрашивает
# ровно эту строку.
NO_ANSWER_LINE = "В найденных статьях нет ответа на тему"
# Ограда Markdown памятки в ответе `faq_summarize` (день 20, §3.5): в тексте
# памятки не бывает ни `~~~`, ни тройных обратных кавычек. Её же разбирает
# панель приложения.
MARKDOWN_FENCE_OPEN = "~~~markdown"
MARKDOWN_FENCE_CLOSE = "~~~"

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


def _span(minutes: float) -> str:
    """Промежуток словами (день 20, §2.4): «3 мин», «5 ч», «7 дн.». Порог
    задаётся минутами и показывается ровно: 10080 — «7 дн.», 2 — «2 мин»."""
    minutes = max(0, int(minutes))
    if minutes >= 1440 and (minutes % 1440 == 0 or minutes >= 2880):
        return f"{minutes // 1440} дн."
    if minutes >= 60 and (minutes % 60 == 0 or minutes >= 120):
        return f"{minutes // 60} ч"
    return f"{minutes} мин"


def is_fresh(started_at: datetime, now: datetime, max_age_min: int) -> bool:
    """Чистая часть «копия хорошая» (день 20, §3.2): удачный снимок начат не
    раньше `max_age_min` минут назад."""
    return now - started_at <= timedelta(minutes=max_age_min)


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
    Версии 1 (день 18) и 2 (день 19) принимаются — `init_db()` домигрирует
    их до 3."""
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
    if version not in (0, 1, 2, SCHEMA_VERSION):
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
    """Схема для новой базы и миграции — одной транзакцией вместе с версией —
    и режим WAL: инструменты читают, пока снимок пишет. Версия 1 (база дня
    18) — добавляются таблицы дней 19-20 (`searches` сразу в редакции дня 20);
    версия 2 (база дня 19) — `searches` пересобирается, `search_articles`
    добавляется (день 20, §3.7). Таблицы дня 18 и строки `settings` не
    трогаются."""
    if version == SCHEMA_VERSION:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with closing(_connect(path)) as conn:
        conn.execute("PRAGMA journal_mode = WAL")
        with _transaction(conn):
            if version == 0:
                for statement in SCHEMA:
                    conn.execute(statement)
            if version in (0, 1):
                conn.execute(SEARCHES_TABLE.format(name="searches"))
                for statement in SCHEMA_V19:
                    conn.execute(statement)
                conn.execute(SEARCH_ARTICLES_TABLE)
            if version == 2:
                for statement in MIGRATE_V2_V3:
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


# --- Памятка к столу: поиск и id (день 19, §2.3, §2.5) ----------------------

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


def _site_down(exc: ToolError) -> bool:
    """Сбой чтения сайта FAQ — сеть, ответ не 200, разметка, чужое имя
    категории (день 20, §2.4): тогда ответ идёт из устаревшей копии, если она
    есть. 404 статьи и статья чужой категории — не сбой сайта, а ответ сайта:
    их текст уходит модели как есть."""
    text = str(exc)
    return not (text.startswith("статьи ") or text.startswith("статья "))


def _modified_shown(raw: str) -> str:
    """«Modified on: Tue, 13 Nov, 2018 at 8:35 AM» → «Tue, 13 Nov, 2018 at 8:35
    AM»: в ответе `faq_article` у строки своя подпись."""
    text = " ".join(raw.split())
    if text.startswith(MODIFIED_PREFIX):
        text = text[len(MODIFIED_PREFIX):].strip()
    return text or "—"


@dataclass(frozen=True)
class SearchRecord:
    """Строка `searches` (§3.7): что искали и где — на каком снимке
    (`source='копия'`) или на сайте (`source='сайт'`, `run_id` — `None`)."""
    id: int
    created_at: datetime
    query: str
    limit: int
    run_id: int | None
    source: str
    article_ids: list[str]


@dataclass(frozen=True)
class CopyState:
    """Состояние копии FAQ (день 20, §3.2): последний удачный снимок, его
    возраст и порог, идёт ли снимок, неудачная попытка после него и срок
    следующей попытки (`None` — снимков сейчас не ждём)."""
    last_ok: "Run | None"
    age: timedelta | None
    max_age_min: int
    fresh: bool
    running: "Running | None"
    failed: "Run | None"
    next_at: datetime | None

    @property
    def threshold(self) -> str:
        return _span(self.max_age_min)

    def stale_reason(self) -> str:
        """Почему копия плохая — для запроса снимка и строки лога."""
        if self.last_ok is None:
            return "копия пуста"
        return (
            f"копия устарела (возраст {_span(self.age.total_seconds() / 60)}, "
            f"порог {self.threshold})"
        )


# --- Сторож ----------------------------------------------------------------

class SnapshotError(Exception):
    """Снимок неполный — попытка неудачная, со статьями не записывается."""


@dataclass
class Running:
    number: int
    started_at: datetime


class Watch:
    """Состояние процесса сервера: где база, расписание, порог свежести копии,
    идёт ли снимок, нужен ли снимок и как разбудить исполнитель снимков. Один
    на процесс; снимки делает только исполнитель, инструменты читают базу
    (или сайт), просят снимок и меняют расписание."""

    def __init__(
        self, config: Config, db: Path, max_age_min: int = DEFAULT_MAX_AGE_MIN,
        parent_pid: int | None = None,
    ) -> None:
        self.config = config
        self.db = db
        # Порог свежести копии (день 20, §3.2) — аргумент `--max-age-min`.
        self.max_age_min = max_age_min
        # Родитель, за которым следит сервер (день 20, §3.3) — аргумент
        # `--parent-pid` автозапуска; у ручного запуска его нет.
        self.parent_pid = parent_pid
        self.interval = DEFAULT_INTERVAL_MIN
        self.interval_origin = ORIGIN_DEFAULT
        # Расписание (день 20, §3.3) — выключено по умолчанию.
        self.schedule_enabled = False
        self.running: Running | None = None
        # Запрос снимка (день 20, §2.5): причина — «копия пуста» / «копия
        # устарела (…)»; "" — снимок не нужен. Ставят `faq_search` и
        # `faq_article`, снимает начало снимка. Только в памяти процесса.
        self.refresh_reason = ""
        # Срок «не раньше» после попытки, которой нет в `runs` (§2.4, правка
        # по ревью): сайт прочитан, но база снимок не приняла, или после
        # чтения упал код. Срок из базы тогда остался в прошлом, и без этого
        # поля исполнитель перечитывал бы сайт на каждом шаге проверки часов.
        # Только в памяти процесса: запись попытки его снимает, перезапуск —
        # тоже.
        self.not_before: datetime | None = None
        # Событие создаётся в цикле событий — при старте исполнителя.
        self.wake: anyio.Event | None = None

    # --- расписание ---

    def load_schedule(self, from_arg: int | None) -> None:
        """Кто задал расписание последним, тот и прав (§2.4, день 20 §3.3):
        явный `--interval` включает расписание и сохраняется; без него —
        интервал и включение из базы. Нет ключа `schedule_enabled` (базы дней
        18-19) — выключено, даже если интервал сохранён."""
        if from_arg is not None:
            self.save_schedule(from_arg, True)
            self.interval_origin = ORIGIN_ARG
            return
        with closing(_connect(self.db)) as conn:
            rows = {
                row["key"]: str(row["value"])
                for row in conn.execute(
                    "SELECT key, value FROM settings WHERE key IN (?, ?)",
                    (SETTING_INTERVAL, SETTING_SCHEDULE),
                )
            }
        self.schedule_enabled = rows.get(SETTING_SCHEDULE) == "1"
        value = rows.get(SETTING_INTERVAL)
        if value is not None and value.isdigit():
            if MIN_INTERVAL_MIN <= int(value) <= MAX_INTERVAL_MIN:
                self.interval, self.interval_origin = int(value), ORIGIN_DB
                return
            logger.warning(
                "интервал в базе вне %d-%d мин: %s — действует умолчание",
                MIN_INTERVAL_MIN, MAX_INTERVAL_MIN, value,
            )
        self.interval, self.interval_origin = DEFAULT_INTERVAL_MIN, ORIGIN_DEFAULT

    def save_schedule(self, minutes: int | None, enabled: bool) -> None:
        """Сохранить включение расписания и, если задан, интервал — одной
        транзакцией."""
        values = [(SETTING_SCHEDULE, "1" if enabled else "0")]
        if minutes is not None:
            values.append((SETTING_INTERVAL, str(minutes)))
        with closing(_connect(self.db)) as conn, _transaction(conn):
            conn.executemany(
                "INSERT INTO settings (key, value) VALUES (?, ?) "
                "ON CONFLICT(key) DO UPDATE SET value = excluded.value",
                values,
            )
        self.schedule_enabled = enabled
        if minutes is not None:
            self.interval = minutes

    def schedule_text(self) -> str:
        """Расписание словами — для строки старта и ответов."""
        if not self.schedule_enabled:
            return "расписание выключено — снимки по запросу, когда копия пуста или старше порога"
        return f"раз в {self.interval} мин ({self.interval_origin})"

    def _retry_interval(self) -> int:
        """Интервал для формулы повтора после неудачи: при выключенном
        расписании — `RETRY_MIN` (день 20, §2.5)."""
        return self.interval if self.schedule_enabled else RETRY_MIN

    def attempts(self, conn: sqlite3.Connection) -> tuple[Run | None, Run | None]:
        """Последний удачный снимок и неудачная попытка после него (если
        последняя попытка — неудачная)."""
        last_ok = conn.execute("SELECT * FROM runs WHERE ok = 1 ORDER BY id DESC LIMIT 1").fetchone()
        last = conn.execute("SELECT * FROM runs ORDER BY id DESC LIMIT 1").fetchone()
        ok_run = _run(last_ok) if last_ok is not None else None
        failed = _run(last) if last is not None and not last["ok"] else None
        return ok_run, failed

    def plan(self, conn: sqlite3.Connection | None = None) -> tuple[datetime | None, str]:
        """Срок следующего снимка и его причина (день 20, §3.3) — из двух
        источников работы: запрос снимка и расписание, если оно включено.
        `(None, "")` — снимков сейчас не ждём.

        Запрос — сразу, но после неудачной попытки не раньше её времени +
        `min(интервал, RETRY_MIN)`: пауза — вежливость к сайту, ответ
        пользователю в это время всё равно идёт с сайта. Расписание — формула
        дня 18 (`next_due()`)."""
        if conn is None:
            with closing(_connect(self.db)) as own:
                return self.plan(own)
        ok_run, failed = self.attempts(conn)
        now = _now()
        candidates: list[tuple[datetime, str]] = []
        if self.refresh_reason:
            due = (
                failed.started_at + timedelta(minutes=min(self._retry_interval(), RETRY_MIN))
                if failed else now
            )
            candidates.append((due, f"по запросу: {self.refresh_reason}"))
        if self.schedule_enabled:
            due = next_due(
                ok_run.started_at if ok_run else None,
                failed.started_at if failed else None,
                self.interval,
                now,
            )
            candidates.append((due, "по расписанию"))
        if not candidates:
            return None, ""
        due, reason = min(candidates, key=lambda item: item[0])
        if self.not_before is not None and self.not_before > due:
            due = self.not_before
        return due, reason

    def _hold_off(self) -> datetime:
        """Срок «не раньше» после попытки, которой нет в базе:
        `min(интервал, RETRY_MIN)` от сейчас — как повтор после неудачи."""
        self.not_before = _now() + timedelta(minutes=min(self._retry_interval(), RETRY_MIN))
        return self.not_before

    def _due_text(self, due: datetime | None) -> str:
        if self.running is not None:
            return f"идёт сейчас (снимок #{self.running.number}, начат {_fmt(self.running.started_at)})"
        if due is None:
            return "по запросу, когда копия пуста или устарела (расписание выключено)"
        if due <= _now():
            return "сейчас — срок уже прошёл"
        return _fmt(due)

    # --- копия FAQ и запрос снимка (день 20, §2.4-§2.5) ---

    def copy_state(self, conn: sqlite3.Connection, now: datetime) -> CopyState:
        """Хорошая ли копия: последний удачный снимок, возраст, порог, идёт ли
        снимок, неудачная попытка после него и срок следующей попытки."""
        ok_run, failed = self.attempts(conn)
        age = now - ok_run.started_at if ok_run is not None else None
        due, _ = self.plan(conn)
        return CopyState(
            last_ok=ok_run,
            age=age,
            max_age_min=self.max_age_min,
            fresh=ok_run is not None and is_fresh(ok_run.started_at, now, self.max_age_min),
            running=self.running,
            failed=failed,
            next_at=due,
        )

    def request_refresh(self, reason: str) -> bool:
        """Запрос снимка без ожидания (§2.5): флаг и побудка исполнителя.
        Пока идёт снимок, запрос ничего не делает — копия и так станет
        свежей. Возвращает, поставлен ли флаг."""
        if self.running is not None:
            return False
        self.refresh_reason = reason
        if self.wake is not None:
            self.wake.set()
        return True

    def _refresh_tail(self) -> str:
        """Хвост строки `источник:` у ответа с сайта — что с докачкой."""
        if self.running is not None:
            return (
                f" (снимок #{self.running.number} начат "
                f"{self.running.started_at.astimezone().strftime('%H:%M')})"
            )
        due, _ = self.plan()
        if due is not None and due > _now():
            return f" (следующая попытка не раньше {due.astimezone().strftime('%H:%M')})"
        return ""

    async def watch_parent(self) -> None:
        """Родитель сменился (приложение убито без выхода) — остановиться, как
        по Ctrl+C (день 20, §3.3): SIGINT себе, uvicorn выходит штатно, и
        идущий снимок в базу не попадает."""
        if self.parent_pid is None:
            return
        while True:
            await anyio.sleep(SCHEDULER_TICK_S)
            if os.getppid() != self.parent_pid:
                logger.info("родитель %d завершился — останавливаюсь", self.parent_pid)
                os.kill(os.getpid(), signal.SIGINT)
                return

    # --- исполнитель снимков (§3.4, день 20 §3.3) ---

    async def scheduler(self) -> None:
        """Один цикл: снимки не перекрываются, срок — из базы по стенным
        часам, запрос снимка и смена расписания будят цикл раньше конца сна."""
        self.wake = anyio.Event()
        while True:
            try:
                due, _ = self.plan()
                while (due is None or _now() < due) and not self.wake.is_set():
                    wait_s = SCHEDULER_TICK_S if due is None else min(
                        SCHEDULER_TICK_S, max(0.0, (due - _now()).total_seconds())
                    )
                    with anyio.move_on_after(wait_s):
                        await self.wake.wait()
                    due, _ = self.plan()
                self.wake = anyio.Event()
                # Запрос или расписание могли смениться за время сна —
                # пересчитать срок.
                due, reason = self.plan()
                if due is not None and _now() >= due:
                    self.refresh_reason = ""
                    await self.snapshot(reason)
            except Exception:
                # Сервер не падает от одной ошибки: трассировка — в stderr.
                # Ошибка могла случиться после чтения сайта, а попытки в базе
                # нет — поэтому снимок не раньше `_hold_off()`, а не на
                # следующем шаге проверки часов. Сон — на случай, если падает
                # само чтение срока из базы: без него цикл крутился бы вхолостую.
                logger.exception(
                    "исполнитель снимков: ошибка в коде сервера — снимок не раньше %s",
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

    async def snapshot(self, why: str) -> None:
        """Одна попытка снимка. Исключений наружу не выпускает, кроме отмены:
        неудача чтения сайта — строка `runs` с `ok=0` и причиной; база не
        приняла запись — строка лога и срок «не раньше» (`_hold_off()`)."""
        number = self._next_number()
        started_at = _now()
        started = time.perf_counter()
        self.running = Running(number, started_at)
        logger.info("снимок #%d: начат (%s)", number, why)
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
            due, _ = self.plan()
            logger.warning(
                "снимок #%d: сбой через %.1f с — %s; повтор %s",
                number, elapsed, reason,
                _fmt(due) if due is not None
                else f"по следующему запросу, не раньше чем через {min(self._retry_interval(), RETRY_MIN)} мин",
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
        logger.info("следующий снимок: %s", self._due_text(self.plan()[0]))

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
            state = self.copy_state(conn, now)

            # Строка состояния (день 20, §3.5) — про копию и докачку, а не
            # про «наблюдение по расписанию».
            lines = [
                f"FAQ издателя «{self.config.category_name}» ({self.config.category_url})",
            ]
            schedule = (
                f"расписание: раз в {self.interval} мин" if self.schedule_enabled
                else "расписание выключено"
            )
            if last_ok is None:
                copy = (
                    "Копии ещё нет: снимков не было; первый снимок начнётся по первому вопросу к FAQ"
                    if not runs else "Копии ещё нет: удачных снимков не было"
                )
            else:
                copy = (
                    f"Копия: снимок #{last_ok.id} от {_fmt(last_ok.started_at)}, "
                    f"{_articles_word(last_ok.articles)}, возраст "
                    f"{_span(state.age.total_seconds() / 60)}, порог {state.threshold}"
                    + ("" if state.fresh else " — устарела, обновится по следующему вопросу к FAQ")
                )
            if self.running is not None:
                schedule += (
                    f"; идёт снимок #{self.running.number} (начат {_fmt(self.running.started_at)})"
                )
            elif state.next_at is not None:
                schedule += f"; следующий снимок: {self._due_text(state.next_at)}"
            lines.append(f"{copy}; {schedule}.")
            status = f"Снимков за {days} дн.: {len(in_period)}"
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

    def _schedule_label(self) -> str:
        return f"раз в {self.interval} мин" if self.schedule_enabled else "выключено"

    def schedule(self, minutes: int) -> tuple[str, str]:
        """Ответ `faq_watch_schedule` и строка для лога (§3.7, день 20 §3.3):
        `0` выключает расписание, `N` от 5 включает его с этим интервалом; и
        то и другое сохраняется в базе, исполнитель просыпается и
        пересчитывает срок. 1-4 — отказ: схема пускает их (`ge=0`), потому
        что 0 — законное значение."""
        if 0 < minutes < MIN_INTERVAL_MIN:
            raise ToolError(
                f"интервал — от {MIN_INTERVAL_MIN} минут, или 0 — выключить; пришло {minutes}"
            )
        previous = self._schedule_label()
        if minutes == 0:
            self.save_schedule(None, False)
        else:
            self.save_schedule(minutes, True)
            self.interval_origin = ORIGIN_DB
        if self.wake is not None:
            self.wake.set()
        with closing(_connect(self.db)) as conn:
            ok_run, _ = self.attempts(conn)
            due, _ = self.plan(conn)
        last = (
            f"Последний удачный снимок — {_fmt(ok_run.started_at)}"
            if ok_run else "Удачных снимков ещё нет"
        )
        if minutes == 0:
            text = (
                f"Расписание выключено (было: {previous}). Копия FAQ обновляется сама, когда "
                f"нужна и устарела. {last}."
            )
            return text, f"было: {previous}; выключено"
        if self.running is not None:
            upcoming = (
                f"снимок #{self.running.number} идёт сейчас; следующий — по новому "
                f"расписанию после него"
            )
        elif due is not None and due <= _now():
            upcoming = "срок по новому расписанию уже прошёл — снимок начинается сейчас"
        else:
            upcoming = f"следующий — {self._due_text(due)}"
        text = f"Расписание: раз в {minutes} мин (было: {previous}). {last}, {upcoming}."
        return text, f"было: {previous}; следующий снимок {self._due_text(due)}"

    # --- источник ответа: копия или сайт (день 20, §2.4, §3.5) ---

    def _copy_source(self, state: CopyState) -> str:
        run = state.last_ok
        return (
            f"источник: {SOURCE_COPY} — снимок #{run.id} от {_fmt(run.started_at)} (возраст "
            f"{_span(state.age.total_seconds() / 60)}, порог {state.threshold})"
        )

    def _site_source(self, state: CopyState) -> str:
        """Строка `источник:` ответа с сайта — считается после чтения сайта,
        когда фоновый снимок уже начат."""
        if state.last_ok is None:
            head = "копия пуста"
        else:
            head = (
                f"копия устарела (снимок #{state.last_ok.id} от "
                f"{state.last_ok.started_at.astimezone().strftime(DATE_FORMAT)}, порог "
                f"{state.threshold})"
            )
        return f"источник: {SOURCE_SITE} — {head}, докачивается в фоне{self._refresh_tail()}"

    @staticmethod
    def _site_error(exc: ToolError) -> str:
        text = str(exc)
        prefix = "сайт FAQ недоступен: "
        return text[len(prefix):] if text.startswith(prefix) else text

    def _stale_source(self, state: CopyState, exc: ToolError) -> str:
        run = state.last_ok
        return (
            f"источник: {SOURCE_COPY} — снимок #{run.id} от "
            f"{run.started_at.astimezone().strftime(DATE_FORMAT)} (устарела); сайт недоступен: "
            f"{self._site_error(exc)}"
        )

    def _unavailable(self, exc: ToolError, state: CopyState, missing: str = "") -> ToolError:
        """Сайт не отвечает, а в копии ответа нет (§2.4)."""
        if state.last_ok is None:
            return ToolError(f"FAQ недоступен: сайт не отвечает ({self._site_error(exc)}), копии ещё нет")
        return ToolError(
            f"FAQ недоступен: сайт не отвечает ({self._site_error(exc)}), а в копии (снимок "
            f"#{state.last_ok.id}) {missing}"
        )

    # --- поиск (день 19; день 20 — копия или сайт) ---

    async def search(self, query: str, limit: int) -> tuple[str, str]:
        """Ответ `faq_search` и строка для лога (§2.4, §3.4): хорошая копия —
        поиск в ней, как на дне 19; плохая — запрос снимка и поиск по вопросам
        на сайте; сайт не ответил — устаревшая копия с пометкой, копии нет —
        отказ."""
        with closing(_connect(self.db)) as conn:
            state = self.copy_state(conn, _now())
        if state.fresh:
            return self._search_copy(query, limit, state.last_ok, self._copy_source(state))
        requested = self.request_refresh(state.stale_reason())
        need = " · нужен снимок" if requested else ""
        try:
            found = await self._read_site_search(query, limit)
        except ToolError as exc:
            if state.last_ok is None:
                raise self._unavailable(exc, state) from exc
            text, summary = self._search_copy(
                query, limit, state.last_ok, self._stale_source(state, exc)
            )
            return text, f"{summary}, сайт недоступен{need}"
        text, summary = self._record_site_search(query, limit, state, *found)
        return text, f"{summary}{need}"

    def _search_copy(self, query: str, limit: int, run: Run, source: str) -> tuple[str, str]:
        """Поиск в снимке `run` (день 19, §2.5) — сеть не трогает, результат
        воспроизводим."""
        with closing(_connect(self.db)) as conn:
            rows = conn.execute(
                "SELECT a.article_id, a.section, a.question, t.text FROM articles a "
                "JOIN texts t ON t.hash = a.text_hash WHERE a.run_id = ? ORDER BY a.rowid",
                (run.id,),
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
                    "INSERT INTO searches (id, created_at, run_id, source, query, limit_n, "
                    "article_ids) VALUES (NULL, ?, ?, ?, ?, ?, ?)",
                    (_iso(_now()), run.id, SOURCE_COPY, query, limit, json.dumps(article_ids)),
                )
                search_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        ref = f"{SEARCH_PREFIX}{search_id}"
        lines = [f"id: {ref}", source]
        where = f"в снимке #{run.id} ({_fmt(run.started_at)})"
        if not top:
            lines.append(
                f"Поиск «{query}» {where}: ничего не найдено среди {total}; попробуй другие "
                f"английские слова."
            )
            return "\n".join(lines), f"{ref} — 0 из {total}, из копии #{run.id}"
        lines.append(f"Поиск «{query}» {where}: {_articles_word(len(top))} из {total}.")
        lines.append("Строка: номер · раздел · вопрос · совпало.")
        for _, hits, row in top:
            lines.append(
                f"{row['article_id']} · {row['section']} · {row['question']} · {_hits_text(hits)}"
            )
        lines.append(
            f"Текст статьи — faq_article(номер). Для памятки передай id {ref} в faq_summarize."
        )
        return "\n".join(lines), f"{ref} — {len(top)} из {total}, из копии #{run.id}"

    async def _read_site_search(
        self, query: str, limit: int,
    ) -> tuple[int, list[tuple[dict[str, list[str]], str, str, str]], dict[str, Article], int]:
        """Поиск по сайту (§3.4): вопросы всех разделов (`read_sections()`,
        7 запросов), слова запроса сравниваются только с вопросами, тексты
        первых `limit` найденных статей читаются параллельно (потолок `Site` —
        4). Возвращает число статей, найденные (совпало, номер, раздел,
        вопрос), прочитанные статьи и число запросов. Сбой сайта — `ToolError`."""
        async with Site(self.config) as site:
            sections, _ = await read_sections(site, self.config)
            listed: dict[str, tuple[str, str]] = {}
            for section in sections:
                for article_id, question in section.articles:
                    listed.setdefault(article_id, (section.name, question))
            query_words = _search_words(query)
            scored: list[tuple[int, dict[str, list[str]], str, str, str]] = []
            for article_id, (section, question) in listed.items():
                score, hits = _score_article(query_words, _search_words(question), [])
                if score > 0:
                    scored.append((score, hits, article_id, section, question))
            scored.sort(key=lambda item: -item[0])
            top = [(hits, article_id, section, question) for _, hits, article_id, section, question in scored[:limit]]

            found: dict[str, Article] = {}

            async def one(article_id: str) -> None:
                found[article_id] = await read_article(site, self.config, article_id)

            try:
                async with asyncio.TaskGroup() as group:
                    for _, article_id, _, _ in top:
                        group.create_task(one(article_id))
            except ExceptionGroup as errors:
                first = errors.exceptions[0]
                while isinstance(first, ExceptionGroup):
                    first = first.exceptions[0]
                raise first from None
            return len(listed), top, found, site.requests

    def _record_site_search(
        self, query: str, limit: int, state: CopyState, total: int,
        top: list[tuple[dict[str, list[str]], str, str, str]], found: dict[str, Article],
        requests: int,
    ) -> tuple[str, str]:
        """Поиск по сайту — в базу одной транзакцией вместе с прочитанными
        текстами (§3.7): `faq_summarize` потом работает по ссылке `q…` тем же
        путём, что по копии. Номер, раздел и вопрос — из списка, текст — со
        страницы, как у снимка."""
        article_ids = [article_id for _, article_id, _, _ in top]
        with closing(_connect(self.db)) as conn, _transaction(conn):
            conn.execute(
                "INSERT INTO searches (id, created_at, run_id, source, query, limit_n, "
                "article_ids) VALUES (NULL, ?, NULL, ?, ?, ?, ?)",
                (_iso(_now()), SOURCE_SITE, query, limit, json.dumps(article_ids)),
            )
            search_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]
            for position, (_, article_id, section, question) in enumerate(top):
                article = found[article_id]
                digest = text_hash(article.text)
                conn.execute(
                    "INSERT OR IGNORE INTO texts (hash, text) VALUES (?, ?)", (digest, article.text),
                )
                conn.execute(
                    "INSERT INTO search_articles (search_id, position, article_id, section, "
                    "question, url, text_hash) VALUES (?, ?, ?, ?, ?, ?, ?)",
                    (search_id, position, article_id, section, question, article.url, digest),
                )

        ref = f"{SEARCH_PREFIX}{search_id}"
        lines = [f"id: {ref}", self._site_source(state)]
        where = f"Поиск «{query}» по вопросам FAQ на сайте"
        stale = "копия пуста" if state.last_ok is None else "копия устарела"
        summary_tail = f"с сайта ({stale}), {_requests_word(requests)}"
        if not top:
            lines.append(
                f"{where}: по вопросам ничего не найдено среди {total}; копия докачивается — "
                f"через минуту поиск пойдёт и по текстам статей."
            )
            return "\n".join(lines), f"{ref} — 0 из {total}, {summary_tail}"
        lines.append(f"{where}: {_articles_word(len(top))} из {total}.")
        lines.append("Строка: номер · раздел · вопрос · совпало.")
        for hits, article_id, section, question in top:
            lines.append(f"{article_id} · {section} · {question} · {_hits_text(hits)}")
        lines.append(
            f"Текст статьи — faq_article(номер). Для памятки передай id {ref} в faq_summarize."
        )
        return "\n".join(lines), f"{ref} — {len(top)} из {total}, {summary_tail}"

    # --- статья (день 20, §3.1, §3.4) ---

    async def article(self, article_id: str) -> tuple[str, str]:
        """Ответ `faq_article` и строка для лога: хорошая копия со статьёй —
        из копии; статьи в хорошей копии нет — с сайта (новая статья); копия
        плохая — запрос снимка и сайт; сайт не ответил — устаревшая копия с
        пометкой или отказ. 404 и статья чужой категории — отказ сайта как
        есть. В базу статья с сайта не пишется: копию обновит снимок."""
        with closing(_connect(self.db)) as conn:
            state = self.copy_state(conn, _now())
            row = None
            if state.last_ok is not None:
                row = conn.execute(
                    "SELECT a.section, a.question, a.modified_raw, t.text FROM articles a "
                    "JOIN texts t ON t.hash = a.text_hash WHERE a.run_id = ? AND a.article_id = ?",
                    (state.last_ok.id, article_id),
                ).fetchone()
        if state.fresh and row is not None:
            text = self._article_text(
                self._copy_source(state), row["question"], row["section"],
                self._article_url(article_id), row["modified_raw"], row["text"],
            )
            return text, f"из копии #{state.last_ok.id}"

        need = ""
        if state.fresh:
            run = state.last_ok
            source = (
                f"источник: {SOURCE_SITE} — в копии нет (снимок #{run.id} от "
                f"{_fmt(run.started_at)}): статья прочитана с сайта"
            )
            label = "с сайта (в копии нет)"
        else:
            if self.request_refresh(state.stale_reason()):
                need = " · нужен снимок"
            source = ""
            label = "с сайта (" + ("копия пуста" if state.last_ok is None else "копия устарела") + ")"
        try:
            async with Site(self.config) as site:
                found = await read_article(site, self.config, article_id)
                requests = site.requests
        except ToolError as exc:
            if not _site_down(exc) or state.fresh:
                raise
            if row is None:
                raise self._unavailable(exc, state, f"статьи {article_id} нет") from exc
            text = self._article_text(
                self._stale_source(state, exc), row["question"], row["section"],
                self._article_url(article_id), row["modified_raw"], row["text"],
            )
            return text, f"из устаревшей копии #{state.last_ok.id}, сайт недоступен{need}"
        if not source:
            source = self._site_source(state)
        if self.running is not None:
            label = label[:-1] + f"; снимок #{self.running.number} идёт)"
        text = self._article_text(
            source, found.question, found.section, found.url, found.modified, found.text,
        )
        return text, f"{label}, {_requests_word(requests)}{need}"

    @staticmethod
    def _article_text(
        source: str, question: str, section: str, url: str, modified: str, text: str,
    ) -> str:
        """Формат `faq_article` дня 17 плюс строка `источник:` и дата
        изменения (день 20, §3.5)."""
        return "\n".join([
            source,
            f"Вопрос: {question}",
            f"Раздел: {section or '—'}",
            f"Ссылка: {url}",
            f"Изменена на сайте: {_modified_shown(modified)}",
            "Ответ издателя:",
            text or "(текст статьи пуст)",
        ])

    # --- памятка к столу (день 19; день 20 — поиск по сайту и Markdown) ---

    def _load_search(self, conn: sqlite3.Connection, search_id: int) -> SearchRecord:
        row = conn.execute("SELECT * FROM searches WHERE id = ?", (search_id,)).fetchone()
        if row is None:
            raise ToolError(f"поиска {SEARCH_PREFIX}{search_id} нет")
        return SearchRecord(
            id=row["id"], created_at=datetime.fromisoformat(row["created_at"]),
            query=row["query"], limit=row["limit_n"], run_id=row["run_id"],
            source=row["source"], article_ids=json.loads(row["article_ids"]),
        )

    def _search_articles(
        self, conn: sqlite3.Connection, record: SearchRecord,
    ) -> list[tuple[str, str, str, str]]:
        """Тексты статей поиска. Поиск по копии — из того же снимка, на
        котором искали (§2.3): снимка нет или в нём нет статьи — одна и та же
        ошибка, «повтори поиск» (снимки не удаляются, поэтому в обычном
        прогоне она не встречается). Поиск по сайту (день 20, §3.7) — из
        `search_articles`, то, что поиск прочитал; не совпало с найденным — та
        же ошибка."""
        if record.source == SOURCE_SITE:
            rows = conn.execute(
                "SELECT sa.article_id, sa.section, sa.question, t.text FROM search_articles sa "
                "JOIN texts t ON t.hash = sa.text_hash WHERE sa.search_id = ? ORDER BY sa.position",
                (record.id,),
            ).fetchall()
            if [row["article_id"] for row in rows] != record.article_ids:
                raise ToolError(
                    f"тексты поиска {SEARCH_PREFIX}{record.id} в базе не совпадают с найденным — "
                    f"повтори поиск"
                )
            return [
                (row["article_id"], row["section"], row["question"], row["text"]) for row in rows
            ]
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
        просьбы к модели клиента, чтобы неверный id не стоил вызова.
        Детерминирована: `Resolve` выполняется на каждом раунде просьбы, и
        сайт здесь не читается — поиск по сайту уже записал тексты."""
        search_id = _require_search_ref(search_id_raw)
        with closing(_connect(self.db)) as conn:
            record = self._load_search(conn, search_id)
            articles = self._search_articles(conn, record)
        if not articles:
            raise ToolError(f"поиск {SEARCH_PREFIX}{record.id} ничего не нашёл — сжимать нечего")
        return record, articles

    def _origin(self, record: SearchRecord, run: Run | None) -> tuple[str, str]:
        """Откуда тексты памятки — строка `источник:` и строка источников
        Markdown (день 20, §3.5)."""
        search_ref = f"{SEARCH_PREFIX}{record.id}"
        if record.source == SOURCE_SITE or run is None:
            read_at = _fmt(record.created_at)
            return (
                f"источник: {SOURCE_SITE} — прочитано {read_at} (поиск {search_ref})",
                f"прочитано с сайта {read_at} (поиск {search_ref})",
            )
        return (
            f"источник: {SOURCE_COPY} — снимок #{run.id} от {_fmt(run.started_at)}",
            f"снимок #{run.id} от {_fmt(run.started_at)}",
        )

    def _cheatsheet_markdown(
        self, topic: str, text: str, record: SearchRecord, sources: str,
        articles: list[tuple[str, str, str, str]], summary_ref: str, model: str,
        summary_hash: str,
    ) -> str:
        """Markdown памятки для сохранения (день 20, §3.5) — формат файла дня
        19 без «→ этот файл»: заголовок, источники и строку происхождения
        пишет код из базы, модель — только пункты (уже в `text`)."""
        lines = [f"# Памятка: {topic}", "", text, "", "---", ""]
        lines.append(
            f"Источники — официальный FAQ издателя «{self.config.category_name}», {sources}:"
        )
        for article_id, section, question, _ in articles:
            lines.append(f"- [{article_id}]({self._article_url(article_id)}) · {section} · {question}")
        lines.append("")
        lines.append(
            f"Собрано {_fmt(_now())} · поиск {SEARCH_PREFIX}{record.id} («{record.query}», "
            f"{_articles_word(len(articles))}) → памятка {summary_ref} (модель {model}, "
            f"sha {summary_hash[:12]}…)"
        )
        return "\n".join(lines)

    def summarize(
        self, search_id_raw: str, topic: str, result: CreateMessageResult,
    ) -> tuple[str, str]:
        """Ответ `faq_summarize` (§3.3): разбирает ответ модели клиента,
        проверяет ссылки на статьи этого поиска, пишет памятку одной строкой,
        и (день 20, §3.5) отдаёт готовый Markdown для `save_markdown` сервера
        файлов. Оборванный по лимиту ответ не записывается — правило трекера
        дня 13, оборванный результат хуже отсутствия."""
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
        with closing(_connect(self.db)) as conn:
            run_row = (
                conn.execute("SELECT * FROM runs WHERE id = ?", (record.run_id,)).fetchone()
                if record.run_id is not None else None
            )
            with _transaction(conn):
                conn.execute(
                    "INSERT INTO summaries (id, created_at, search_id, topic, text, text_hash, "
                    "model, items, foreign_refs) VALUES (NULL, ?, ?, ?, ?, ?, ?, ?, ?)",
                    (_iso(_now()), record.id, topic, text, digest, model, len(items),
                     json.dumps(foreign)),
                )
                summary_id = conn.execute("SELECT last_insert_rowid()").fetchone()[0]

        ref = f"{SUMMARY_PREFIX}{summary_id}"
        search_ref = f"{SEARCH_PREFIX}{record.id}"
        source_line, sources = self._origin(record, _run(run_row) if run_row is not None else None)
        markdown = self._cheatsheet_markdown(
            topic, text, record, sources, articles, ref, model, digest,
        )
        lines = [
            f"id: {ref}",
            source_line,
            f"Памятка «{topic}» из поиска {search_ref} ({_articles_word(len(articles))}), "
            f"модель клиента {model}, {len(items)} "
            f"{_plural(len(items), 'пункт', 'пункта', 'пунктов')}.",
            text,
        ]
        if foreign:
            lines.append("⚠️ ссылки вне поиска: " + ", ".join(foreign))
        lines += [
            "Чтобы сохранить в файл, передай Markdown между ~~~ в save_markdown как есть:",
            MARKDOWN_FENCE_OPEN,
            markdown,
            MARKDOWN_FENCE_CLOSE,
        ]
        # Время в строке лога — только тело инструмента: раунды — разные
        # HTTP-запросы, и сервер без состояния не знает, когда начался первый;
        # время сэмплинга — в логах клиента и агента (§3.8).
        summary = (
            f"{ref} — {_count(len(items), 'пункт', 'пункта', 'пунктов')}, модель {model}, "
            f"Markdown {len(markdown)} символов"
        )
        return "\n".join(lines), summary

    def start_line(self, port: int) -> str:
        """Строка старта (день 20, §3.8): адрес, база, копия, порог свежести,
        расписание и родитель."""
        now = _now()
        with closing(_connect(self.db)) as conn:
            total = conn.execute("SELECT count(*) FROM runs").fetchone()[0]
            state = self.copy_state(conn, now)
        if state.last_ok is None:
            copy = f"копия: пусто (попыток снимка: {total})"
        else:
            copy = (
                f"копия: снимок #{state.last_ok.id} от {_fmt(state.last_ok.started_at)}, возраст "
                f"{_span(state.age.total_seconds() / 60)}" + ("" if state.fresh else " — устарела")
            )
        parent = f" · родитель {self.parent_pid}" if self.parent_pid is not None else ""
        return (
            f"слушаю http://{HOST}:{port}/mcp · база {display_path(self.db)} · {copy} · копия "
            f"считается свежей {state.threshold} (--max-age-min {self.max_age_min}) · "
            f"{self.schedule_text()}{parent}"
        )


def _hits_text(hits: dict[str, list[str]]) -> str:
    """«poison (вопрос, текст)» — где совпало каждое слово запроса."""
    return ", ".join(f"{word} ({', '.join(sorted(set(where)))})" for word, where in hits.items())


# --- Сервер ----------------------------------------------------------------

def build_server(watch: Watch) -> MCPServer:
    @asynccontextmanager
    async def scheduler_lifespan(_server: MCPServer):
        # Вход — task group с исполнителем снимков и (день 20, §3.3)
        # сторожем родителя, выход (Ctrl+C) — её отмена. Снимок, прерванный
        # отменой, в базу не попадает (§3.3).
        async with anyio.create_task_group() as group:
            group.start_soon(watch.scheduler)
            group.start_soon(watch.watch_parent)
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
            int, Field(ge=0, le=MAX_INTERVAL_MIN, description=INTERVAL_DESCRIPTION)
        ],
    ) -> str:
        return _logged(
            f"faq_watch_schedule({interval_minutes})", lambda: watch.schedule(interval_minutes)
        )

    @server.tool(
        title="Поиск в FAQ издателя",
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
        return await _logged_async(
            f'faq_search("{query}", {limit})', watch.search(query, limit),
        )

    @server.tool(
        title="Статья FAQ издателя",
        description=ARTICLE_DESCRIPTION.format(name=game_name),
        annotations=ARTICLE_ANNOTATIONS,
        structured_output=False,
    )
    async def faq_article(
        # Шаблон в схеме здесь уместен (§3.1): чужого вида ссылок, как у
        # `q…`/`s…`, у номера статьи нет, а ошибка валидации pydantic модели
        # понятна.
        article_id: Annotated[
            str, Field(pattern=r"^\d{5,20}$", description=ARTICLE_ID_DESCRIPTION)
        ],
    ) -> str:
        return await _logged_async(f"faq_article({article_id})", watch.article(article_id))

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


async def _logged_async(call: str, work) -> str:
    """То же, что `_logged()`, для инструментов, которые читают сайт (день
    20): поиск и статья."""
    started = time.perf_counter()
    try:
        text, summary = await work
    except ToolError as exc:
        logger.info("%s: отказ — %s, %.2f с", call, exc, time.perf_counter() - started)
        raise
    except sqlite3.Error as exc:
        logger.warning("%s: отказ — база сервера: %s", call, exc)
        raise ToolError(f"база сервера не читается: {type(exc).__name__}: {exc}") from exc
    logger.info("%s: %s, %.2f с", call, summary, time.perf_counter() - started)
    return text


def _interval(value: str) -> int:
    if not value.isdigit() or not MIN_INTERVAL_MIN <= int(value) <= MAX_INTERVAL_MIN:
        raise argparse.ArgumentTypeError(
            f"ожидается целое число минут от {MIN_INTERVAL_MIN} до {MAX_INTERVAL_MIN}, "
            f"пришло «{value}»"
        )
    return int(value)


def _positive(value: str) -> int:
    if not value.isdigit() or int(value) < 1:
        raise argparse.ArgumentTypeError(f"ожидается целое число ≥ 1, пришло «{value}»")
    return int(value)


@dataclass(frozen=True)
class Args:
    config: Config
    db: Path
    port: int
    interval: int | None
    max_age_min: int
    parent_pid: int | None


def _args(argv: list[str] | None = None) -> Args:
    parser = argparse.ArgumentParser(
        description=(
            "FAQ издателя: MCP-сервер (Streamable HTTP) одной категории FAQ портала Freshdesk — "
            "копия в SQLite и сайт, фоновые снимки по запросу или по расписанию."
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
        "--port", type=int, default=DEFAULT_PORT,
        help=f"порт на {HOST}, по умолчанию {DEFAULT_PORT}",
    )
    parser.add_argument(
        "--interval", type=_interval, default=None,
        help=(
            f"интервал снимков в минутах, {MIN_INTERVAL_MIN}-{MAX_INTERVAL_MIN}; задан — "
            f"включает расписание и сохраняется в базе (по умолчанию расписание выключено)"
        ),
    )
    parser.add_argument(
        "--max-age-min", type=_positive, default=DEFAULT_MAX_AGE_MIN,
        help=(
            f"копия свежая, пока последний удачный снимок не старше стольких минут; по "
            f"умолчанию {DEFAULT_MAX_AGE_MIN} (неделя)"
        ),
    )
    parser.add_argument(
        "--parent-pid", type=_positive, default=None,
        help="pid процесса, который запустил сервер: сменился родитель — сервер останавливается",
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
    return Args(
        config=config, db=Path(args.db).expanduser(), port=args.port, interval=args.interval,
        max_age_min=args.max_age_min, parent_pid=args.parent_pid,
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
    args = _args()
    db, port = args.db, args.port
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
    if version not in (0, SCHEMA_VERSION):
        logger.info("база %s: схема %d → %d", display_path(db), version, SCHEMA_VERSION)

    # SIGTERM — как Ctrl+C (день 20, §6.1): им сервер останавливает
    # приложение, которое его запустило. Без обработчика процесс умирал бы
    # сразу, без строки «остановлен» и без штатного выхода uvicorn; снимок в
    # базу и так попадает только целиком.
    signal.signal(signal.SIGTERM, signal.default_int_handler)

    watch = Watch(args.config, db, args.max_age_min, args.parent_pid)
    watch.load_schedule(args.interval)
    logger.info("%s", watch.start_line(port))
    try:
        build_server(watch).run("streamable-http", host=HOST, port=port, stateless_http=True)
    except KeyboardInterrupt:
        pass
    logger.info("остановлен")


if __name__ == "__main__":
    main()
