# TooManyRules — свой MCP-сервер вокруг FAQ портала поддержки на Freshdesk
# (день 17, неделя 4).
#
# **Отдельная программа, а не модуль приложения** (спецификация дня 17, §3.1):
# её запускает `mcp_client` процессом (`python faq_server.py …`), приложение её
# не импортирует, и она не импортирует ничего из проекта. В графе зависимостей
# её нет — как нет `app_week1.py`. Клиентскую часть SDK `mcp` в проекте
# импортирует только `mcp_client.py`, серверную — только этот файл.
#
# Сервер знает портал Freshdesk, но не конкретную игру: адрес портала, id
# категории и её ожидаемое имя приходят аргументами командной строки (их
# собирает `presets.FAQ_MCP`). Разметка портала — константы ниже, чтобы смена
# вёрстки правилась в одном месте.
#
# Два инструмента (§2.2): `faq_questions` — список вопросов категории,
# `faq_article` — текст одной статьи. Поиска нет: `/support/search` закрыт в
# robots.txt, а подходящий вопрос по списку выбирает модель. При старте сервер
# в сеть не ходит — `tools/list` отвечает сразу; сайт читается только на
# `tools/call`. Состояния между вызовами нет: процесс живёт одно соединение,
# кэша страниц нет.
#
# SDK — вторая мажорная версия: `MCPServer` (в v1 он назывался `FastMCP`;
# `from mcp.server.fastmcp import FastMCP` в v2 падает с сообщением о
# миграции). Примеры v1 сюда не переносить.
#
# stdout процесса — канал JSON-RPC, поэтому логи идут только в stderr, а он —
# в терминал приложения как есть (§3.6).

import argparse
import asyncio
import logging
import re
import sys
import time
from dataclasses import dataclass, field
from html.parser import HTMLParser
from typing import Annotated

import httpx
from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from mcp.types import ToolAnnotations
from pydantic import Field

SERVER_NAME = "toomanyrules-faq"
SERVER_VERSION = "1.0.0"

# --- Сайт ------------------------------------------------------------------
# Вежливость к чужому сайту (§3.4): таймаут на запрос, не больше четырёх
# запросов одновременно, повторов нет — сбой становится ошибкой инструмента.
HTTP_TIMEOUT_S = 10
HTTP_CONCURRENCY = 4
# Заголовок HTTP кириллицу не примет — только ASCII.
USER_AGENT = "TooManyRules-FAQ-MCP/1.0 (AI Advent Challenge study project)"
# Защита от бесконечной пагинации раздела: у разделов FAQ сейчас по 1-2
# страницы по 10 статей.
MAX_FOLDER_PAGES = 20

# Адреса портала. Только категории, разделы и статьи — `/support/search` не
# используется (robots.txt).
CATEGORY_PATH = "/support/solutions/{id}"
FOLDER_PATH = "/support/solutions/folders/{id}"
ARTICLE_PATH = "/support/solutions/articles/{id}"
CATEGORY_HREF = re.compile(r"^/support/solutions/(\d+)(?:[/?#]|$)")
FOLDER_HREF = re.compile(r"^/support/solutions/folders/(\d+)")
# Часть адреса после номера статьи необязательна: сайт открывает статью и без
# неё, а ссылка в ответе — каноническая, без хвоста (§3.4).
ARTICLE_HREF = re.compile(r"^/support/solutions/articles/(\d+)")

# --- Маркеры разметки ------------------------------------------------------
# (тег, класс) или (тег, "#id"). Проверено по страницам 22-23.09.2026
# (спецификация дня 17, §2.1).
CATEGORY_HEADING = ("h2", "solution-category-heading")   # имя категории
SECTION_BLOCK = ("section", "article-list")              # раздел на странице категории
SECTION_TITLE = ("div", "solution-list-title")           # в нём ссылка на раздел: title — имя
SECTION_COUNT = ("span", "item-count")                   # объявленное число статей раздела
NEXT_PAGE = ("li", "next")                               # пагинация раздела: ссылка «Next»
ARTICLE_TITLE = ("h2", "")                               # последний h2 перед телом статьи
ARTICLE_BODY = ("article", "#article-body")              # тело статьи
BREADCRUMBS = ("div", "breadcrumbs")                     # хлебные крошки статьи
BREADCRUMB_LINK = ("a", "breadcrumbs-btn")               # категория и раздел статьи

# Теги, у которых нет закрывающего: в дерево не проталкиваются.
VOID_TAGS = frozenset({
    "area", "base", "br", "col", "embed", "hr", "img", "input", "link", "meta",
    "param", "source", "track", "wbr",
})
# Блочные теги тела статьи дают переводы строк (§3.4).
BLOCK_TAGS = frozenset({
    "p", "div", "ul", "ol", "table", "tr", "blockquote", "pre",
    "h1", "h2", "h3", "h4", "h5", "h6", "section",
})
SKIP_TAGS = frozenset({"script", "style"})

# --- Описания инструментов -------------------------------------------------
# Описание инструмента и есть промпт: когда звать инструмент, модель узнаёт
# отсюда, а не из системного промпта (§3.3, §9). Каждое изменение по сбою
# живого прогона записывается строкой в комментарии у константы.
#
# Правки `QUESTIONS_DESCRIPTION` по живому прогону (23.09.2026, `deepseek-flash`):
# - «Вызывай перед ответом…» → «перед каждым ответом… даже если ответ кажется
#   известным: FAQ разбирает как раз частые ошибки»: с черновиком спецификации
#   модель на И2 (Poison) не позвала инструменты и ответила неверно («останется
#   Poison 2»), на И4 — тоже без вызова. После правки оба вопроса — с вызовом
#   (по 2 прогона из 2), И3 («что ты умеешь») — по-прежнему без вызова.
# - «Подходящей статьи нет — скажи игроку, что в FAQ этого нет…»: на И4 модель
#   отвечала из общих знаний без оговорки (§7.3 ждёт оговорку).
QUESTIONS_DESCRIPTION = (
    "Список вопросов официального FAQ издателя по игре {name}: номер статьи, раздел и вопрос "
    "(по-английски). Вызывай перед каждым ответом на вопрос о правилах — даже если ответ "
    "кажется известным: FAQ разбирает как раз частые ошибки (эффекты вроде Poison, порядок "
    "хода, герои, лут). Затем открой подходящую статью через faq_article. Если вопрос про "
    "конкретного героя — сузь список параметром section. Подходящей статьи нет — скажи "
    "игроку, что в FAQ этого нет, и отвечай из общих знаний."
)
# Правка `SECTION_DESCRIPTION` по живому прогону (23.09.2026): на И2 (Poison)
# и `deepseek-flash`, и `deepseek-v4-pro` с thinking передавали
# `section="Poison"` как поиск по теме, получали перечень разделов, открывали
# статьи раздела Battle и отвечали неверно (статья про Poison — в разделе
# Baddie Skills and Encounter Terms). Добавлено последнее предложение.
SECTION_DESCRIPTION = (
    "Необязательный фильтр по разделу: часть английского названия, без учёта регистра — "
    "имя героя (Patches, Tink, Picket), Battle, Loot, Tyrants. Пусто — все вопросы. "
    "Это фильтр по названию раздела, а не поиск по теме: если вопрос не про конкретного "
    "героя, вызывай без фильтра."
)
ARTICLE_DESCRIPTION = (
    "Текст статьи официального FAQ по игре {name} по её номеру из faq_questions: вопрос, "
    "раздел, ссылка и ответ издателя (по-английски). Отвечая игроку по статье, перескажи "
    "ответ по-русски и дай ссылку на статью."
)
ARTICLE_ID_DESCRIPTION = "Номер статьи — первое поле строки из faq_questions."

# Аннотации честные (§3.3): инструменты только читают, повтор безопасен,
# мир открытый — чужой сайт.
READ_ONLY = ToolAnnotations(
    read_only_hint=True,
    destructive_hint=False,
    idempotent_hint=True,
    open_world_hint=True,
)

logger = logging.getLogger("toomanyrules.faq_server")


# --- Разбор HTML -----------------------------------------------------------
# `html.parser` из стандартной библиотеки строит маленькое дерево — без
# новых зависимостей. Сущности HTML раскрывает сам парсер
# (`convert_charrefs=True`), в тексте и в атрибутах.

@dataclass
class Node:
    tag: str
    attrs: dict[str, str]
    children: list["Node | str"] = field(default_factory=list)

    def classes(self) -> set[str]:
        return set(self.attrs.get("class", "").split())

    def matches(self, marker: tuple[str, str]) -> bool:
        tag, selector = marker
        if self.tag != tag:
            return False
        if not selector:
            return True
        if selector.startswith("#"):
            return self.attrs.get("id") == selector[1:]
        return selector in self.classes()

    def iter(self):
        """Узлы поддерева в порядке документа, включая этот."""
        yield self
        for child in self.children:
            if isinstance(child, Node):
                yield from child.iter()

    def find_all(self, marker: tuple[str, str]) -> list["Node"]:
        return [node for node in self.iter() if node.matches(marker)]

    def find(self, marker: tuple[str, str]) -> "Node | None":
        return next((node for node in self.iter() if node.matches(marker)), None)

    def text(self) -> str:
        """Весь текст поддерева одной строкой, пробелы схлопнуты."""
        parts: list[str] = []

        def walk(node: Node) -> None:
            for child in node.children:
                if isinstance(child, str):
                    parts.append(child)
                elif child.tag not in SKIP_TAGS:
                    walk(child)

        walk(self)
        return " ".join("".join(parts).split())


class _TreeBuilder(HTMLParser):
    """Дерево из HTML. Незакрытые теги (`li`, `p`) закрываются, когда
    закрывается их предок; закрывающий тег без открывающего пропускается."""

    def __init__(self) -> None:
        super().__init__(convert_charrefs=True)
        self.root = Node("#root", {})
        self._stack: list[Node] = [self.root]

    def handle_starttag(self, tag, attrs):
        node = Node(tag, {name: value or "" for name, value in attrs})
        self._stack[-1].children.append(node)
        if tag not in VOID_TAGS:
            self._stack.append(node)

    def handle_startendtag(self, tag, attrs):
        self._stack[-1].children.append(Node(tag, {name: value or "" for name, value in attrs}))

    def handle_endtag(self, tag):
        for index in range(len(self._stack) - 1, 0, -1):
            if self._stack[index].tag == tag:
                del self._stack[index:]
                return

    def handle_data(self, data):
        self._stack[-1].children.append(data)


def parse_html(html: str) -> Node:
    builder = _TreeBuilder()
    builder.feed(html)
    builder.close()
    return builder.root


def _link_id(node: Node, pattern: re.Pattern) -> str | None:
    match = pattern.match(node.attrs.get("href", ""))
    return match.group(1) if match else None


def body_text(body: Node) -> str:
    """Текст статьи: абзацы и `br` дают переводы строк, пункты списков — «- »,
    картинки — «[изображение]», у ссылок остаётся текст; пустые строки подряд
    схлопываются (§3.4)."""
    parts: list[str] = []

    def walk(node: Node) -> None:
        for child in node.children:
            if isinstance(child, str):
                parts.append(child)
                continue
            if child.tag in SKIP_TAGS:
                continue
            if child.tag == "br":
                parts.append("\n")
            elif child.tag == "img":
                parts.append(" [изображение] ")
            elif child.tag == "li":
                parts.append("\n- ")
                walk(child)
                parts.append("\n")
            elif child.tag in BLOCK_TAGS:
                parts.append("\n")
                walk(child)
                parts.append("\n")
            else:
                walk(child)

    walk(body)
    lines = [" ".join(line.split()) for line in "".join(parts).split("\n")]
    text = "\n".join(lines)
    # «- » пункта списка после схлопывания становится «-»: возвращаем пробел.
    text = re.sub(r"^-(?=\S)", "- ", text, flags=re.MULTILINE)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


# --- Данные ----------------------------------------------------------------

@dataclass
class Section:
    id: str
    name: str
    declared: int                                  # число статей по span.item-count
    articles: list[tuple[str, str]] = field(default_factory=list)  # (номер, вопрос)


@dataclass
class Config:
    base_url: str
    category: str
    category_name: str

    @property
    def category_url(self) -> str:
        return self.base_url + CATEGORY_PATH.format(id=self.category)


class Site:
    """Чтение сайта в пределах одного вызова инструмента: свой клиент httpx,
    потолок одновременных запросов и счётчик запросов для строки лога."""

    def __init__(self, config: Config) -> None:
        self.config = config
        self.requests = 0
        self._semaphore = asyncio.Semaphore(HTTP_CONCURRENCY)
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

    async def get(self, path: str, *, not_found: str = "") -> Node:
        """Страница разобранным деревом. Сбой сети и ответ не 200 — `ToolError`;
        `not_found` — свой текст для 404."""
        async with self._semaphore:
            self.requests += 1
            started = time.perf_counter()
            try:
                response = await self._client.get(path)
            except httpx.HTTPError as exc:
                logger.info(
                    "GET %s → %s за %.2f с", path, type(exc).__name__,
                    time.perf_counter() - started,
                )
                raise ToolError(
                    f"сайт FAQ недоступен: {type(exc).__name__} при запросе {path}"
                ) from exc
            logger.info(
                "GET %s → %d за %.2f с", path, response.status_code,
                time.perf_counter() - started,
            )
        if response.status_code == 404 and not_found:
            raise ToolError(not_found)
        if response.status_code != 200:
            raise ToolError(f"сайт FAQ ответил {response.status_code} на {path}")
        return parse_html(response.text)


def _not_parsed(marker: tuple[str, str], path: str) -> ToolError:
    tag, selector = marker
    where = selector if selector.startswith("#") else (f".{selector}" if selector else "")
    return ToolError(f"разметка страницы не разобрана: не найден {tag}{where} на {path}")


# --- faq_questions ---------------------------------------------------------

def _category_sections(page: Node, config: Config, path: str) -> list[Section]:
    """Имя категории сверяется с ожидаемым; разделы — в порядке страницы, со
    статьями, которые страница категории показывает (не больше пяти)."""
    heading = page.find(CATEGORY_HEADING)
    if heading is None:
        raise _not_parsed(CATEGORY_HEADING, path)
    name = heading.text()
    if name != config.category_name:
        raise ToolError(
            f"категория {config.category} на сайте называется «{name}», ожидалась "
            f"«{config.category_name}» — проверьте настройки сервера в presets.py"
        )
    sections: list[Section] = []
    for block in page.find_all(SECTION_BLOCK):
        title = block.find(SECTION_TITLE)
        link = next(
            (a for a in (title.find_all(("a", "")) if title else []) if _link_id(a, FOLDER_HREF)),
            None,
        )
        if link is None:
            continue
        count = link.find(SECTION_COUNT)
        declared = count.text() if count is not None else ""
        section = Section(
            id=_link_id(link, FOLDER_HREF),
            name=" ".join(link.attrs.get("title", "").split()) or link.text(),
            declared=int(declared) if declared.isdigit() else 0,
        )
        seen: set[str] = set()
        for a in block.find_all(("a", "")):
            article_id = _link_id(a, ARTICLE_HREF)
            if article_id and article_id not in seen:
                seen.add(article_id)
                question = " ".join(a.attrs.get("title", "").split()) or a.text()
                section.articles.append((article_id, question))
        sections.append(section)
    if not sections:
        raise _not_parsed(SECTION_BLOCK, path)
    return sections


async def _read_folder(site: Site, section: Section) -> None:
    """Раздел целиком — со всех страниц пагинации, в порядке сайта."""
    articles: list[tuple[str, str]] = []
    seen: set[str] = set()
    path: str | None = FOLDER_PATH.format(id=section.id)
    visited: set[str] = set()
    while path and path not in visited and len(visited) < MAX_FOLDER_PAGES:
        visited.add(path)
        page = await site.get(path)
        for a in page.find_all(("a", "")):
            article_id = _link_id(a, ARTICLE_HREF)
            if article_id and article_id not in seen:
                seen.add(article_id)
                articles.append((article_id, a.text()))
        next_item = page.find(NEXT_PAGE)
        next_link = next_item.find(("a", "")) if next_item is not None else None
        path = next_link.attrs.get("href") if next_link is not None else None
    section.articles = articles


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


def _questions_word(count: int) -> str:
    return f"{count} {_plural(count, 'вопрос', 'вопроса', 'вопросов')}"


def _in_sections(count: int) -> str:
    return f"в {count} {'разделе' if count % 10 == 1 and count % 100 != 11 else 'разделах'}"


async def questions(config: Config, section_filter: str) -> tuple[str, str]:
    """Ответ `faq_questions` и строка для лога."""
    path = CATEGORY_PATH.format(id=config.category)
    async with Site(config) as site:
        sections = _category_sections(await site.get(path), config, path)
        wanted = section_filter.strip().lower()
        chosen = [s for s in sections if wanted in s.name.lower()] if wanted else sections
        # Дочитываются только разделы, прошедшие фильтр, и только те, где
        # страница категории показала меньше объявленного. Параллельно, с
        # потолком одновременных запросов; порядок разделов при этом не
        # меняется — список `chosen` тот же.
        await asyncio.gather(*(
            _read_folder(site, s) for s in chosen if len(s.articles) < s.declared
        ))
        requests = site.requests

    total = sum(len(s.articles) for s in chosen)
    header = f"FAQ «{config.category_name}» ({config.category_url})"
    if wanted:
        lines = [f"{header}."]
    else:
        lines = [f"{header}: {_questions_word(total)} {_in_sections(len(chosen))}."]
    if wanted and not chosen:
        lines.append(
            f"Разделов с «{section_filter.strip()}» нет. Разделы: "
            + ", ".join(s.name for s in sections) + "."
        )
        return "\n".join(lines), (
            f"разделов по фильтру нет, {requests} "
            f"{_plural(requests, 'запрос', 'запроса', 'запросов')}"
        )
    if wanted:
        lines.append(
            f"Разделы по фильтру «{section_filter.strip()}»: "
            + ", ".join(f"{s.name} — {_questions_word(len(s.articles))}" for s in chosen)
            + "."
        )
    lines.append("Строка: номер статьи · раздел · вопрос.")
    for s in chosen:
        lines.extend(f"{article_id} · {s.name} · {question}" for article_id, question in s.articles)
    # Проверка полноты (§3.4, п. 5): неполный список — не ошибка, он полезен и
    # таким, но модель должна знать, что он неполный.
    for s in chosen:
        if len(s.articles) != s.declared:
            lines.append(
                f"⚠️ Раздел {s.name}: на сайте {s.declared}, прочитано {len(s.articles)}"
            )
    summary = (
        f"{_questions_word(total)}, {len(chosen)} "
        f"{_plural(len(chosen), 'раздел', 'раздела', 'разделов')}, {requests} "
        f"{_plural(requests, 'запрос', 'запроса', 'запросов')}"
    )
    return "\n".join(lines), summary


# --- faq_article -----------------------------------------------------------

async def article(config: Config, article_id: int) -> tuple[str, str]:
    """Ответ `faq_article` и строка для лога."""
    path = ARTICLE_PATH.format(id=article_id)
    async with Site(config) as site:
        page = await site.get(
            path, not_found=f"статьи {article_id} в FAQ нет (сайт ответил 404)"
        )

    crumbs = page.find(BREADCRUMBS)
    links = crumbs.find_all(BREADCRUMB_LINK) if crumbs is not None else []
    if not links:
        raise _not_parsed(BREADCRUMBS, path)
    # Проверка категории (§3.5): статья другой категории — ошибка, а не текст.
    # Номер статьи из FAQ 20 Strong иначе выглядел бы официальным ответом по
    # нашей игре.
    category = next((a for a in links if _link_id(a, CATEGORY_HREF)), None)
    if category is None or _link_id(category, CATEGORY_HREF) != config.category:
        where = f"в категории «{category.text()}»" if category is not None else "без категории"
        raise ToolError(
            f"статья {article_id} — не из FAQ «{config.category_name}»: по хлебным "
            f"крошкам она {where}"
        )
    folder = next((a for a in reversed(links) if _link_id(a, FOLDER_HREF)), None)

    body = None
    title = None
    for node in page.iter():
        if node.matches(ARTICLE_BODY):
            body = node
            break
        if node.matches(ARTICLE_TITLE):
            title = node
    if body is None:
        raise _not_parsed(ARTICLE_BODY, path)
    if title is None:
        raise _not_parsed(ARTICLE_TITLE, path)

    text = "\n".join([
        f"Вопрос: {title.text()}",
        f"Раздел: {folder.text() if folder is not None else '—'}",
        f"Ссылка: {config.base_url}{path}",
        "Ответ издателя:",
        body_text(body) or "(текст статьи пуст)",
    ])
    return text, f"{len(text)} символов"


# --- Сервер ----------------------------------------------------------------

def _args(argv: list[str] | None = None) -> Config:
    parser = argparse.ArgumentParser(
        description="MCP-сервер (stdio) вокруг одной категории FAQ портала Freshdesk.",
    )
    parser.add_argument("--base-url", required=True, help="адрес портала без завершающего /")
    parser.add_argument("--category", required=True, help="id категории — строка цифр")
    parser.add_argument(
        "--category-name", required=True,
        help="ожидаемое имя категории: сверяется с заголовком её страницы",
    )
    args = parser.parse_args(argv)
    if not args.category.isdigit():
        parser.error(f"--category: ожидается строка цифр, пришло «{args.category}»")
    return Config(
        base_url=args.base_url.rstrip("/"),
        category=args.category,
        category_name=args.category_name,
    )


def build_server(config: Config) -> MCPServer:
    # `log_level="WARNING"`: строки INFO SDK с форматированием rich — шум в
    # терминале приложения (§3.6). Настраивает корневой логгер — поэтому свой
    # логгер сервера ниже в корневой не пишет.
    server = MCPServer(name=SERVER_NAME, version=SERVER_VERSION, log_level="WARNING")

    @server.tool(
        title="Вопросы FAQ",
        description=QUESTIONS_DESCRIPTION.format(name=config.category_name),
        annotations=READ_ONLY,
        # Результат — текст для модели: без флага SDK добавил бы
        # `outputSchema` и `structuredContent` с тем же текстом.
        structured_output=False,
    )
    async def faq_questions(
        section: Annotated[str, Field(description=SECTION_DESCRIPTION)] = "",
    ) -> str:
        return await _logged(f'faq_questions(section="{section}")', questions(config, section))

    @server.tool(
        title="Статья FAQ",
        description=ARTICLE_DESCRIPTION.format(name=config.category_name),
        annotations=READ_ONLY,
        structured_output=False,
    )
    async def faq_article(
        article_id: Annotated[int, Field(description=ARTICLE_ID_DESCRIPTION)],
    ) -> str:
        return await _logged(f"faq_article({article_id})", article(config, article_id))

    return server


async def _logged(call: str, work) -> str:
    """Строка лога на вызов: итог или отказ. `ToolError` уходит клиенту
    текстом (SDK сделает из него `isError: true`); любое другое исключение —
    ошибка в коде сервера, клиент увидит только «Error executing tool …», а
    трассировку SDK напишет в stderr."""
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
    handler.setFormatter(logging.Formatter("[FAQ-сервер] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    # Каждая строка httpx «HTTP Request: GET …» уровня INFO — шум: свою строку
    # на запрос пишет `Site.get()`.
    logging.getLogger("httpx").setLevel(logging.WARNING)
    build_server(config).run("stdio")


if __name__ == "__main__":
    main()
