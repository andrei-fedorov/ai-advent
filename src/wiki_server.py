# TooManyRules — свой MCP-сервер вокруг фанатской вики на boardgamehub (день
# 20, неделя 4).
#
# **Отдельная программа, а не модуль приложения** (спецификация дня 20, §4.1),
# как `faq_server.py`: её запускает `mcp_client` процессом по stdio на каждый
# вызов, приложение её не импортирует. В графе зависимостей приложения её нет.
# Сама она импортирует из `faq_server.py` только разбор HTML — дерево `Node`,
# `parse_html()` и `body_text()`: второго разбора в проекте не пишем. Это
# второе ребро между серверами (как `faq_watch.py` → `faq_server.py`), вне
# графа приложения; Freshdesk-часть `faq_server.py` вики не использует.
#
# Сервер знает движок вики boardgamehub, но не Too Many Bones: адрес сайта,
# slug вики и её имя для описаний приходят аргументами командной строки (их
# собирает `presets.WIKI_MCP`). Разметка страниц — константы ниже, чтобы смена
# вёрстки правилась в одном месте (проверено по живым страницам 26.09.2026).
#
# Вики — **только удалённо** (§2.7): ни копии, ни кэша, ни поиска по тексту.
# Навигации хватает: корень вики со списком разделов, индекс раздела —
# таблица «имя · slug · колонки», страница записи — поля и текст. Один вызов —
# обычно один запрос к сайту; `wiki_index` по русскому имени раздела — два
# (корень вики, чтобы найти slug раздела, и сам раздел): slug по русскому
# имени не угадывается («Тираны» — `tiran`, а не `tirany`).
#
# Ожидаемый сбой — `ToolError` (день 17): сайт недоступен, 404, разметка не
# разобрана, неверный раздел или slug. Логи — только stderr, префикс
# `[Вики-сервер]`: stdout процесса — канал JSON-RPC.

import argparse
import logging
import re
import sys
import time
from dataclasses import dataclass
from typing import Annotated
from urllib.parse import urlsplit

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

from faq_server import Node, body_text, parse_html

SERVER_NAME = "toomanyrules-wiki"
SERVER_VERSION = "1.0.0"

# --- Сайт ------------------------------------------------------------------
# Вежливость к чужому сайту (§4.1): один запрос на вызов, таймаут, повторов
# нет — сбой становится ошибкой инструмента.
HTTP_TIMEOUT_S = 10
# Заголовок HTTP кириллицу не примет — только ASCII.
USER_AGENT = "TooManyRules-Wiki-MCP/1.0 (AI Advent Challenge study project)"

WIKI_PATH = "/wiki/{wiki}"
TYPE_PATH = "/wiki/{wiki}/type/{type}"
PAGE_PATH = "/wiki/{wiki}/{slug}"
SLUG_RE = re.compile(r"[a-z0-9-]+")

# --- Потолки (§4.2) --------------------------------------------------------
# Строк индекса в ответе: плохишей 190 — без фильтра видны первые 80.
WIKI_INDEX_MAX_ROWS = 80
# Символов текста страницы: ниже потолка результата клиента (12 000), чтобы
# пометку об обрезке писал сервер, а не клиент (страница Тинка — 12,6 тыс.).
WIKI_PAGE_MAX_CHARS = 9_000

# --- Маркеры разметки ------------------------------------------------------
# (тег, класс) — как у `faq_server.py`. Проверено по страницам 26.09.2026.
ARTICLE = ("article", "wiki-article")                  # статья: страница записи и корень
ARTICLE_TYPE = ("a", "wiki-article__type")             # раздел записи — ссылка над заголовком
ARTICLE_TITLE = ("h1", "article-title")                # заголовок записи и индекса
ARTICLE_BODY = ("div", "wiki-article__body")           # тело статьи
INFOBOX = ("div", "wiki-infobox")                      # карточка полей справа
INFOBOX_GRID = ("div", "grid")                         # сетка «подпись — значение» (характеристики)
TYPE_TABLE_WRAP = ("div", "wiki-type-table-wrap")      # таблица индекса раздела
FILTER_ROW = ("tr", "wiki-type-filters")               # строка фильтров в thead — не данные
# Что из тела статьи не уходит в текст: карточка полей (её поля идут
# строками «ключ: значение»), картинки с подписями и кнопки просмотра.
SKIP_IN_TEXT = frozenset({"figure", "img", "button", "svg"})

# --- Описания инструментов (§4.3) ------------------------------------------
# Описание инструмента и есть промпт (правило дня 17). Черновик спецификации;
# каждое изменение по сбою живого прогона — строкой в комментарии у константы.
#
# Правка `INDEX_DESCRIPTION` по живому прогону (26.09.2026, `deepseek-flash`,
# сценарий О1-О6 одним диалогом): после длинного О2 модель ответила на О3
# («характеристики Бумера») без вызовов и неверно (ОЗ 6, Атака 3 вместо 3 / 1),
# хотя на свежем агенте тот же вопрос шёл в вики. Добавлено «Вызывай, когда
# игрок спрашивает … — не отвечай по памяти».
INDEX_DESCRIPTION = (
    "Фанатская вики по игре {name} на русском: справочник по гирлокам (героям), плохишам, "
    "тиранам, навыкам, коробкам и дополнениям, волнам, совместимости контента, плюс FAQ из "
    "чата сообщества. Вызывай, когда игрок спрашивает про характеристики, роль, навыки или "
    "стиль героя, про врагов, коробки и дополнения — не отвечай по памяти. "
    "Неофициальная: при расхождении с официальным FAQ издателя прав FAQ "
    "издателя. Без type — разделы и отдельные страницы вики; с type — строки раздела (имя, "
    "slug, колонки). Гирлоки названы по-русски (Тинк, Бумер), slug у них английский "
    "(geroi-tink); плохиши — по-английски. Текст записи — wiki_page(slug)."
)
TYPE_DESCRIPTION = "Раздел вики: slug или русское имя — «гирлоки», «plohishi». Пусто — список разделов."
CONTAINS_DESCRIPTION = "Необязательный фильтр строк раздела по подстроке: «Undertow», «Звери»."
PAGE_DESCRIPTION = (
    "Страница фанатской вики по игре {name} по slug из wiki_index: поля (характеристики, "
    "роль, коробка) и текст. Отвечая игроку, скажи, что это фанатская вики, и дай ссылку."
)
SLUG_DESCRIPTION = "slug страницы из wiki_index: geroi-tink, sovmestimost, faq."

# Аннотации честные (§4.2): только чтение, повтор безопасен, мир открытый.
READ_ONLY = ToolAnnotations(
    read_only_hint=True, destructive_hint=False, idempotent_hint=True, open_world_hint=True,
)

logger = logging.getLogger("toomanyrules.wiki_server")


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


def _rows_word(count: int) -> str:
    return f"{count} {_plural(count, 'строка', 'строки', 'строк')}"


def _requests_word(count: int) -> str:
    return f"{count} {_plural(count, 'запрос', 'запроса', 'запросов')}"


def _number(count: int) -> str:
    return f"{count:,}".replace(",", " ")


def _without(node: Node, tags: frozenset[str], markers: tuple[tuple[str, str], ...]) -> Node:
    """Копия поддерева без узлов с этими тегами и маркерами — для текста
    страницы: разбор `faq_server` общий, фильтр — свой."""
    kept: list[Node | str] = []
    for child in node.children:
        if isinstance(child, str):
            kept.append(child)
        elif child.tag in tags or any(child.matches(marker) for marker in markers):
            continue
        else:
            kept.append(_without(child, tags, markers))
    return Node(node.tag, node.attrs, kept)


@dataclass
class Config:
    base_url: str
    wiki: str
    wiki_name: str

    @property
    def wiki_url(self) -> str:
        return self.base_url + WIKI_PATH.format(wiki=self.wiki)

    def page_url(self, slug: str) -> str:
        return self.base_url + PAGE_PATH.format(wiki=self.wiki, slug=slug)

    def type_url(self, type_slug: str) -> str:
        return self.base_url + TYPE_PATH.format(wiki=self.wiki, type=type_slug)


class NotFound(ToolError):
    """Сайт ответил 404 — у раздела это повод показать список разделов."""


class Site:
    """Чтение сайта в пределах одного вызова: свой клиент httpx и счётчик
    запросов для строки лога."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.requests = 0
        self._client = httpx.AsyncClient(
            base_url=config.base_url,
            timeout=HTTP_TIMEOUT_S,
            follow_redirects=True,
            headers={"User-Agent": USER_AGENT},
        )

    async def __aenter__(self) -> "Site":
        return self

    async def __aexit__(self, *exc) -> None:
        await self._client.aclose()

    async def get(self, path: str, *, not_found: str) -> Node:
        """Страница разобранным деревом. Сбой сети, ответ не 200 и редирект за
        пределы запрошенного пути — `ToolError`."""
        self.requests += 1
        try:
            response = await self._client.get(path)
        except httpx.HTTPError as exc:
            raise ToolError(f"сайт вики недоступен: {type(exc).__name__} при запросе {path}") from exc
        if response.status_code == 404:
            raise NotFound(not_found)
        if response.status_code != 200:
            raise ToolError(f"сайт вики ответил {response.status_code} на {path}")
        # Редирект на другой путь (§4.2): страница не из этой вики, или
        # сайт подменил её другой — это не ответ на вопрос.
        landed = response.url.path.rstrip("/")
        if landed != path.rstrip("/"):
            raise ToolError(f"сайт перенаправил {path} на {landed} — это не страница этой вики")
        return parse_html(response.text)


def _not_parsed(marker: tuple[str, str], path: str) -> ToolError:
    tag, selector = marker
    return ToolError(f"разметка страницы не разобрана: не найден {tag}.{selector} на {path}")


# --- Разделы: корень вики ----------------------------------------------------

@dataclass(frozen=True)
class WikiType:
    slug: str
    name: str


def _link_slug(href: str, config: Config) -> tuple[str, str] | None:
    """(«type» | «page», slug) для ссылки внутрь этой вики; иначе `None`.
    Ссылки на сайте абсолютные, поэтому сравнивается путь."""
    path = urlsplit(href).path.rstrip("/")
    prefix = WIKI_PATH.format(wiki=config.wiki) + "/"
    if not path.startswith(prefix):
        return None
    rest = path[len(prefix):]
    if rest.startswith("type/"):
        slug = rest[len("type/"):]
        return ("type", slug) if SLUG_RE.fullmatch(slug) else None
    return ("page", rest) if SLUG_RE.fullmatch(rest) else None


def _nav_types(page: Node, config: Config) -> list[WikiType]:
    """Разделы вики — ссылки `/type/<slug>` в навигации, вне статьи: в теле
    корня те же адреса стоят у ссылок «Весь справочник». Имя — первый `span`
    ссылки (второй — число записей): число в ответ не идёт (§4.2)."""
    types: list[WikiType] = []
    seen: set[str] = set()
    article = page.find(ARTICLE)
    inside = {id(node) for node in article.iter()} if article is not None else set()
    for a in page.find_all(("a", "")):
        if id(a) in inside:
            continue
        target = _link_slug(a.attrs.get("href", ""), config)
        if target is None or target[0] != "type" or target[1] in seen:
            continue
        spans = [child for child in a.children if isinstance(child, Node) and child.tag == "span"]
        if spans:
            seen.add(target[1])
            types.append(WikiType(target[1], spans[0].text()))
    return types


def _root_pages(page: Node, config: Config) -> list[tuple[str, str]]:
    """Отдельные страницы корня (§4.2): ссылки на записи из списков тела
    корня — slug и строка пункта («FAQ — частые вопросы из чата»); затем
    остальные ссылки навигации вне статьи, кроме разделов и истории."""
    pages: list[tuple[str, str]] = []
    seen: set[str] = set()
    article = page.find(ARTICLE)
    body = article.find(ARTICLE_BODY) if article is not None else None
    if body is not None:
        for item in body.find_all(("li", "")):
            for a in item.find_all(("a", "")):
                target = _link_slug(a.attrs.get("href", ""), config)
                if target and target[0] == "page" and target[1] not in seen:
                    seen.add(target[1])
                    pages.append((target[1], item.text().rstrip(".")))
    inside = {id(node) for node in article.iter()} if article is not None else set()
    for a in page.find_all(("a", "")):
        if id(a) in inside:
            continue
        target = _link_slug(a.attrs.get("href", ""), config)
        if target and target[0] == "page" and target[1] not in seen and target[1] != "history":
            seen.add(target[1])
            pages.append((target[1], a.text()))
    return pages


def _types_line(types: list[WikiType]) -> str:
    return "; ".join(f"{t.slug} · {t.name}" for t in types)


async def _root(site: Site) -> Node:
    path = WIKI_PATH.format(wiki=site.config.wiki)
    return await site.get(path, not_found=f"вики {site.config.wiki} на сайте нет (сайт ответил 404)")


async def read_root(config: Config, contains: str) -> tuple[str, str]:
    """Ответ `wiki_index()` без раздела: разделы и отдельные страницы корня."""
    async with Site(config) as site:
        page = await _root(site)
        requests = site.requests
    types = _nav_types(page, config)
    if not types:
        raise ToolError(f"разметка корня вики не разобрана: нет ссылок на разделы на {config.wiki_url}")
    pages = _root_pages(page, config)
    wanted = contains.strip().lower()
    if wanted:
        types = [t for t in types if wanted in f"{t.slug} {t.name}".lower()]
        pages = [p for p in pages if wanted in f"{p[0]} {p[1]}".lower()]
    lines = [
        f"Вики «{config.wiki_name}» ({config.wiki_url}), фанатская, на русском.",
        f"Разделы — wiki_index(type): {_types_line(types) or '—'}.",
        "Отдельные страницы — wiki_page(slug): "
        + ("; ".join(f"{slug} · {label}" for slug, label in pages) or "—") + ".",
    ]
    return "\n".join(lines), f"{len(types)} разделов, {len(pages)} страниц, {_requests_word(requests)}"


# --- Индекс раздела ------------------------------------------------------------

def _cells(row: Node, tag: str) -> list[Node]:
    return [child for child in row.children if isinstance(child, Node) and child.tag == tag]


def _index_rows(page: Node, config: Config, path: str) -> tuple[str, list[str], list[list[str]]]:
    """Заголовок, шапка колонок и строки индекса: `[имя, slug, колонки…]`.
    Шапка — первая строка `thead` (строка фильтров — не данные); строка
    данных — ячейка со ссылкой на запись и остальные ячейки текстом."""
    title_node = page.find(ARTICLE_TITLE)
    wrap = page.find(TYPE_TABLE_WRAP)
    table = wrap.find(("table", "")) if wrap is not None else None
    if table is None:
        raise _not_parsed(TYPE_TABLE_WRAP, path)
    thead = table.find(("thead", ""))
    header_row = next(
        (tr for tr in (thead.find_all(("tr", "")) if thead else []) if not tr.matches(FILTER_ROW)),
        None,
    )
    headers = [th.text() for th in _cells(header_row, "th")] if header_row is not None else []
    tbody = table.find(("tbody", ""))
    rows: list[list[str]] = []
    name_column = -1
    for tr in tbody.find_all(("tr", "")) if tbody is not None else []:
        cells = _cells(tr, "td")
        for index, cell in enumerate(cells):
            link = next(
                (a for a in cell.find_all(("a", ""))
                 if (_link_slug(a.attrs.get("href", ""), config) or ("", ""))[0] == "page"),
                None,
            )
            if link is not None:
                name_column = index
                slug = _link_slug(link.attrs["href"], config)[1]
                rest = [c.text() or "—" for c in cells[index + 1:]]
                rows.append([link.text(), slug, *rest])
                break
    columns = headers[name_column + 1:] if name_column >= 0 else []
    title = title_node.text() if title_node is not None else path
    return title, columns, rows


def _find_type(types: list[WikiType], wanted: str) -> WikiType | None:
    """Раздел по slug или русскому имени, без учёта регистра; обратная черта
    в имени («Символы\\иконки») не обязательна."""
    def norm(text: str) -> str:
        return text.strip().lower().replace("\\", "").replace("/", "").replace(" ", "")
    key = norm(wanted)
    return next((t for t in types if key in (t.slug, norm(t.name))), None)


async def read_index(config: Config, type_raw: str, contains: str) -> tuple[str, str]:
    """Ответ `wiki_index(type)`: строки раздела с фильтром `contains`."""
    wanted = type_raw.strip().lower()
    async with Site(config) as site:
        if SLUG_RE.fullmatch(wanted):
            type_slug = wanted
        else:
            # Русское имя — slug раздела только из навигации корня.
            types = _nav_types(await _root(site), config)
            found = _find_type(types, wanted)
            if found is None:
                raise ToolError(
                    f"раздела «{type_raw.strip()}» в вике нет. Разделы: {_types_line(types)}"
                )
            type_slug = found.slug
        path = TYPE_PATH.format(wiki=config.wiki, type=type_slug)
        try:
            page = await site.get(path, not_found="")
        except NotFound:
            # 404 — неизвестный slug раздела: ответ со списком разделов (второй
            # запрос — только на этом пути ошибки).
            raise ToolError(
                f"раздела «{type_raw.strip()}» в вике нет. Разделы: "
                f"{_types_line(_nav_types(await _root(site), config))}"
            ) from None
        requests = site.requests
    title, columns, rows = _index_rows(page, config, path)
    total = len(rows)
    needle = contains.strip().lower()
    if needle:
        rows = [row for row in rows if any(needle in cell.lower() for cell in row)]

    url = config.type_url(type_slug)
    if needle:
        head = f"{title} — {_rows_word(len(rows))} из {total} с «{contains.strip()}» ({url})."
    else:
        head = f"{title} — {_rows_word(total)} ({url})."
    lines = [head]
    if not rows:
        lines.append("Ни одной строки не подошло — попробуй другую подстроку или без contains.")
        return "\n".join(lines), f"0 из {total}, {_requests_word(requests)}"
    lines.append("Строка: " + " · ".join(["имя", "slug", *columns]) + ".")
    lines.extend(" · ".join(row) for row in rows[:WIKI_INDEX_MAX_ROWS])
    if len(rows) > WIKI_INDEX_MAX_ROWS:
        lines.append(f"… ещё {len(rows) - WIKI_INDEX_MAX_ROWS} — уточни contains.")
    lines.append("Страница записи — wiki_page(slug).")
    shown = f"{_rows_word(len(rows))}" + (f" из {total}" if needle else "")
    return "\n".join(lines), f"{shown}, {_requests_word(requests)}"


# --- Страница ------------------------------------------------------------------

def normalize_slug(raw: str, config: Config) -> str:
    """slug из того, что передала модель (§4.2): полный адрес, путь с
    `/wiki/<вики>/` или просто slug — префикс срезается. Проверка вида —
    здесь, а не в схеме: отказ понятным текстом (правило дня 19)."""
    text = raw.strip()
    if "://" in text:
        text = urlsplit(text).path
    text = text.split("?", 1)[0].split("#", 1)[0].strip("/").lower()
    for prefix in (f"wiki/{config.wiki}/", f"{config.wiki}/"):
        if text.startswith(prefix):
            text = text[len(prefix):]
    if text in ("", "wiki", f"wiki/{config.wiki}", config.wiki):
        raise ToolError("это корень вики — список разделов и страниц даёт wiki_index() без type")
    if text.startswith("type/"):
        raise ToolError(f"«{raw.strip()}» — раздел, а не страница: строки раздела — wiki_index(type)")
    if not SLUG_RE.fullmatch(text):
        raise ToolError(
            f"«{raw.strip()}» не похоже на slug страницы: латиница, цифры и дефис, например "
            f"geroi-tink — возьми его из wiki_index"
        )
    return text


def _infobox_fields(box: Node) -> list[tuple[str, str]]:
    """Поля карточки строками «ключ: значение» (§4.2): сетка характеристик —
    пары `span`; списки `dl` — `dt`/`dd`; секции — заголовок и пункты через
    «; »."""
    fields: list[tuple[str, str]] = []
    for node in box.iter():
        if node.matches(INFOBOX_GRID):
            for cell in node.children:
                if not isinstance(cell, Node):
                    continue
                spans = _cells(cell, "span")
                if len(spans) >= 2:
                    fields.append((spans[0].text(), " ".join(s.text() for s in spans[1:])))
        elif node.tag == "dl":
            key = ""
            for item in node.iter():
                if item.tag == "dt":
                    key = item.text()
                elif item.tag == "dd" and key:
                    fields.append((key, item.text()))
                    key = ""
        elif node.tag == "section":
            heading = node.find(("h2", "")) or node.find(("h3", ""))
            if heading is None:
                continue
            # Пункты списка — через «; »; секция без списка (описание навыка)
            # — её текст без заголовка.
            items = [li.text() for li in node.find_all(("li", "")) if li.text()]
            value = "; ".join(items) if items else _without(node, frozenset(), (("h2", ""), ("h3", ""))).text()
            fields.append((heading.text(), value))
    return [(key, value) for key, value in fields if key and value]


def _clip_text(text: str, limit: int) -> tuple[str, bool]:
    """Текст не длиннее `limit` — по границе абзаца (строки), если она есть
    во второй половине; иначе по границе слова."""
    if len(text) <= limit:
        return text, False
    head = text[:limit]
    cut = head.rfind("\n")
    if cut < limit // 2:
        cut = head.rfind(" ")
    return head[:cut if cut > 0 else limit].rstrip(), True


async def read_page(config: Config, slug_raw: str) -> tuple[str, str]:
    """Ответ `wiki_page(slug)`: заголовок, раздел, ссылка, поля и текст."""
    slug = normalize_slug(slug_raw, config)
    path = PAGE_PATH.format(wiki=config.wiki, slug=slug)
    async with Site(config) as site:
        page = await site.get(path, not_found=f"страницы {slug} в вике нет (сайт ответил 404)")
    article = page.find(ARTICLE)
    if article is None:
        raise _not_parsed(ARTICLE, path)
    title = article.find(ARTICLE_TITLE)
    body = article.find(ARTICLE_BODY)
    if title is None:
        raise _not_parsed(ARTICLE_TITLE, path)
    if body is None:
        raise _not_parsed(ARTICLE_BODY, path)
    type_link = article.find(ARTICLE_TYPE)
    box = body.find(INFOBOX)
    fields = _infobox_fields(box) if box is not None else []
    text = body_text(_without(body, SKIP_IN_TEXT, (INFOBOX,)))
    full_chars = len(text)
    text, clipped = _clip_text(text, WIKI_PAGE_MAX_CHARS)

    head = " · ".join(
        part for part in (title.text(), type_link.text() if type_link is not None else "",
                          config.page_url(slug)) if part
    )
    lines = [f"Страница: {head}"]
    lines.extend(f"{key}: {value}" for key, value in fields)
    lines.append("Текст:")
    lines.append(text or "(текста на странице нет)")
    if clipped:
        lines.append(
            f"✂️ Страница длиннее {_number(WIKI_PAGE_MAX_CHARS)} символов — продолжение по ссылке."
        )
    summary = (
        f"{_number(len(text))} символов (обрезано из {_number(full_chars)})" if clipped
        else f"{_number(len(text))} символов"
    )
    return "\n".join(lines), f"{summary}, {len(fields)} полей"


# --- Сервер --------------------------------------------------------------------

def _args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(
        description="MCP-сервер (stdio) вокруг одной вики движка boardgamehub.",
    )
    parser.add_argument("--base-url", required=True, help="адрес сайта без завершающего /")
    parser.add_argument("--wiki", required=True, help="slug вики: too-many-bones")
    parser.add_argument("--wiki-name", required=True, help="имя вики для описаний инструментов")
    args = parser.parse_args(argv)
    if not SLUG_RE.fullmatch(args.wiki):
        parser.error(f"--wiki: ожидается slug [a-z0-9-]+, пришло «{args.wiki}»")
    return Config(base_url=args.base_url.rstrip("/"), wiki=args.wiki, wiki_name=args.wiki_name)


def build_server(config: Config) -> MCPServer:
    # `log_level="WARNING"`: строки INFO SDK — шум в терминале приложения.
    server = MCPServer(name=SERVER_NAME, version=SERVER_VERSION, log_level="WARNING")

    @server.tool(
        title="Индекс вики",
        description=INDEX_DESCRIPTION.format(name=config.wiki_name),
        annotations=READ_ONLY,
        structured_output=False,
    )
    async def wiki_index(
        type: Annotated[str, Field(max_length=40, description=TYPE_DESCRIPTION)] = "",
        contains: Annotated[str, Field(max_length=60, description=CONTAINS_DESCRIPTION)] = "",
    ) -> str:
        call = f'wiki_index(type="{type}"' + (f', contains="{contains}"' if contains else "") + ")"
        if not type.strip():
            return await _logged(call, read_root(config, contains))
        return await _logged(call, read_index(config, type, contains))

    @server.tool(
        title="Страница вики",
        description=PAGE_DESCRIPTION.format(name=config.wiki_name),
        annotations=READ_ONLY,
        structured_output=False,
    )
    async def wiki_page(
        slug: Annotated[str, Field(min_length=1, max_length=80, description=SLUG_DESCRIPTION)],
    ) -> str:
        return await _logged(f'wiki_page("{slug}")', read_page(config, slug))

    return server


async def _logged(call: str, work) -> str:
    """Строка лога на вызов: итог или отказ. `ToolError` уходит клиенту
    текстом; любое другое исключение — ошибка в коде, трассировку пишет SDK."""
    started = time.perf_counter()
    try:
        text, summary = await work
    except ToolError as exc:
        logger.info("%s: отказ — %s, %.2f с", call, exc, time.perf_counter() - started)
        raise
    logger.info("%s: %s, %.2f с", call, summary, time.perf_counter() - started)
    return text


def main() -> None:
    config = _args()
    handler = logging.StreamHandler(sys.stderr)
    handler.setFormatter(logging.Formatter("[Вики-сервер] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # Строки httpx «HTTP Request: GET …» уровня INFO — шум.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    build_server(config).run("stdio")


if __name__ == "__main__":
    main()
