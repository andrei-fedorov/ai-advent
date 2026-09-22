# TooManyRules — MCP-клиент (день 16, неделя 4).
#
# Единственное место в проекте, где импортируется SDK `mcp`, — как `agent.py`
# единственное место вызова LLM API, а `storage.py` — единственное место
# работы с диском. Модуль запускает stdio-сервер, согласует с ним протокол,
# забирает список инструментов по всем страницам, закрывает соединение и
# отдаёт результат одним значением (`ToolListing`). Каждый шаг пишет в лог.
#
# Лист графа: из проекта не импортирует ничего; ни Gradio, ни Too Many Bones,
# ни BoardGameGeek, ни LLM. Что запускать — приходит параметром (`McpServer`),
# описание конкретного сервера живёт в `presets.py`.
#
# Имя — `mcp_client.py`, а не `mcp.py`: `mcp` — пакет SDK, и файл `src/mcp.py`
# подменил бы его всему процессу (ловушка `profile.py` дня 12, только с
# установленным пакетом вместо модуля стандартной библиотеки). По той же
# причине не `mcp_types.py` — это модуль зависимости SDK.
#
# SDK — вторая мажорная версия (`mcp>=2.2,<3`, спецификация дня 16, §2.5):
# `async with Client(StdioServerParameters(...))` запускает процесс и
# согласует протокол, выход из блока закрывает. Примеры v1 (`ClientSession` +
# `stdio_client` + `session.initialize()`) сюда не переносить. SDK асинхронный,
# проект синхронный: публичная функция обычная, внутри — `anyio.run()`;
# Gradio зовёт синхронные обработчики в рабочем потоке без своего цикла
# событий, и `anyio.run()` там работает.
#
# Режим согласования (`mode`) не закрепляется: `Client` сам пробует
# `server/discover` (эра 2026-07-28) и на ошибку откатывается к рукопожатию
# `initialize` (эра до неё). С сервером старой эры это штатный путь, а не сбой
# (§2.4).
#
# Состояния между вызовами нет: ни кэша, ни соединения, ни глобальных
# объектов, кроме констант и логгера. Каждый вызов `list_tools()` — новый
# процесс и новое соединение; две вкладки — два независимых подключения.
#
# Процесс сервера не наследует окружение приложения: SDK передаёт ему только
# белый список (`INHERITED_ENV`) и то, что перечислено явно (`env_keys`
# описания сервера, если заданы). Значения переменных нигде не показываются и
# не логируются — только имена. Командная строка логируется как есть, поэтому
# ключей в ней не бывает.
#
# Ошибок на ожидаемых сбоях модуль не бросает: любой сбой — `ToolListing` со
# стадией и текстом (§4.6), чтобы интерфейсу не нужно было знать про типы
# ошибок SDK.

import importlib.metadata
import logging
import os
import shlex
import time
from collections.abc import Mapping
from dataclasses import dataclass, field
from datetime import datetime

import anyio
from dotenv import load_dotenv
from mcp import Client, ListToolsResult, MCPError, StdioServerParameters
from mcp.client.stdio import DEFAULT_INHERITED_ENV_VARS

# Модуль сам читает окружение (замену командной строки и переменные для
# сервера), поэтому сам и подхватывает `src/.env` — правило `agent.py`.
# Повторная загрузка безвредна: `load_dotenv()` не перезаписывает заданное.
load_dotenv()

logger = logging.getLogger("toomanyrules.mcp")

# --- Константы -------------------------------------------------------------

STAGE_LAUNCH = "запуск"                 # процесс не запустился (или нечего запускать)
STAGE_CONNECT = "соединение"            # процесс есть, согласование не завершилось
STAGE_LIST = "список инструментов"      # соединение есть, списка нет

HANDSHAKE_INITIALIZE = "initialize"     # рукопожатие эры до 2026-07-28
HANDSHAKE_DISCOVER = "server/discover"  # эра 2026-07-28

SDK_VERSION: str = importlib.metadata.version("mcp")
# Белый список окружения процесса сервера — из SDK, а не второй копией: если
# SDK его поменяет, панель и лог скажут правду.
INHERITED_ENV: tuple[str, ...] = tuple(DEFAULT_INHERITED_ENV_VARS)

ORIGIN_DEFAULT = "по умолчанию"

# Тексты причин — нейтральные: модуль не знает, какой сервер запускает.
ERROR_EMPTY_COMMAND = "командная строка пустая"
ERROR_BAD_COMMAND = "командная строка не разбирается: {reason}"
ERROR_TIMEOUT = "таймаут {seconds:g} с"
ERROR_NO_TOOLS = "сервер не объявил возможность tools"
ERROR_REPEATED_CURSOR = "курсор повторился: {cursor}"


# --- Типы ------------------------------------------------------------------

@dataclass(frozen=True)
class McpServer:
    """Как запустить stdio-сервер. Описание — из presets.py; модуль про конкретный сервер не знает."""
    name: str                       # короткое имя для лога: «bgg»
    title: str                      # для панели
    command: str                    # командная строка по умолчанию — как в терминале
    command_env: str = ""           # переменная окружения, целиком заменяющая command; "" — замены нет
    env_keys: tuple[str, ...] = ()  # переменные, которые передаются серверу, если заданы
    source: str = ""                # ссылка на сервер — для панели


@dataclass(frozen=True)
class Launch:
    """Чем будет запущен сервер при данном окружении. Считается чистой функцией (`resolve_launch`)."""
    command_line: str               # итоговая строка — для лога и панели
    origin: str                     # «по умолчанию» | «из <имя переменной command_env>»
    argv: tuple[str, ...]           # разобранная строка; () — если не разобралась
    env_set: tuple[str, ...]        # какие из env_keys заданы и будут переданы
    env_missing: tuple[str, ...]    # какие не заданы
    error: str = ""                 # пустая или неразбираемая строка


@dataclass(frozen=True)
class McpParam:
    """Параметр для колонки таблицы. Описания параметров таблица не показывает — они в JSON ответа."""
    name: str
    type: str                       # «string», «number», «array[number]», «string|null»; "" — у свойства нет type
    required: bool
    enum: tuple[str, ...] = ()


@dataclass(frozen=True)
class McpTool:
    name: str
    title: str                      # "" — сервер не прислал
    description: str
    params: tuple[McpParam, ...]


@dataclass(frozen=True)
class ToolListing:
    """Результат одного подключения — и удачного, и нет."""
    server: str                     # McpServer.name
    launch: Launch
    at: str                         # момент нажатия, ISO до секунд
    stage: str = ""                 # на какой стадии оборвалось; "" — дошли до конца
    error: str = ""                 # текст сбоя; "" — успех
    # что узнали при подключении (остаётся и при сбое стадии «список инструментов»):
    server_name: str = ""
    server_version: str = ""
    protocol_version: str = ""
    handshake: str = ""             # HANDSHAKE_INITIALIZE | HANDSHAKE_DISCOVER
    capabilities: tuple[str, ...] = ()   # имена объявленных возможностей
    instructions: str = ""
    # список:
    tools: tuple[McpTool, ...] = ()
    pages: tuple[dict, ...] = ()    # ответы tools/list как пришли: model_dump(mode="json", by_alias=True, exclude_unset=True)
    # время, секунды:
    connect_s: float = 0.0          # от начала до установленного соединения
    list_s: float = 0.0             # все страницы tools/list
    total_s: float = 0.0            # от начала до закрытия (и при сбое)

    @property
    def ok(self) -> bool:
        return not self.error


class _NoToolsError(Exception):
    """Сервер не объявил возможность `tools` — `tools/list` не зовём."""

    def __str__(self) -> str:
        return ERROR_NO_TOOLS


class _RepeatedCursorError(Exception):
    """Сервер вернул уже виденный курсор — защита от бесконечного листания."""

    def __init__(self, cursor: str):
        super().__init__(cursor)
        self.cursor = cursor

    def __str__(self) -> str:
        return ERROR_REPEATED_CURSOR.format(cursor=self.cursor)


# --- Командная строка ------------------------------------------------------

def resolve_launch(server: McpServer, environ: Mapping[str, str]) -> Launch:
    """Чем будет запущен сервер при данном окружении — чистая функция.

    Окружение приходит параметром: её зовут и подключение (с `os.environ`), и
    панель при построении, и строка лога при старте — и все видят одно и то же.
    Замена из `command_env` берётся целиком, если непустая; строка разбирается
    `shlex.split()`, тильда и переменные внутри неё не раскрываются.
    """
    override = environ.get(server.command_env, "") if server.command_env else ""
    # Непустая, но из одних пробелов — это замена, и она даёт сбой «пустая»:
    # молча откатиться к умолчанию значило бы запустить не то, что просили.
    if override:
        command_line, origin = override, f"из {server.command_env}"
    else:
        command_line, origin = server.command, ORIGIN_DEFAULT

    env_set = tuple(key for key in server.env_keys if environ.get(key, ""))
    env_missing = tuple(key for key in server.env_keys if not environ.get(key, ""))

    try:
        argv = tuple(shlex.split(command_line))
    except ValueError as exc:
        error = ERROR_BAD_COMMAND.format(reason=exc)
        argv = ()
    else:
        error = "" if argv else ERROR_EMPTY_COMMAND
    return Launch(
        command_line=command_line,
        origin=origin,
        argv=argv,
        env_set=env_set,
        env_missing=env_missing,
        error=error,
    )


# --- Параметры инструмента -------------------------------------------------

def _schema_type(schema: Mapping) -> str:
    kind = schema.get("type")
    if isinstance(kind, list):
        return "|".join(str(item) for item in kind)
    if not isinstance(kind, str):
        return ""
    items = schema.get("items")
    if kind == "array" and isinstance(items, Mapping) and isinstance(items.get("type"), str):
        return f"array[{items['type']}]"
    return kind


def tool_params(input_schema: Mapping | None) -> tuple[McpParam, ...]:
    """Параметры инструмента для колонки таблицы — чистая функция над JSON Schema.

    Свойства — в том порядке, в каком пришли; вложенные объекты не
    разворачиваются (тип `object`, подробности — в JSON ответа).
    """
    if not isinstance(input_schema, Mapping):
        return ()
    properties = input_schema.get("properties")
    if not isinstance(properties, Mapping):
        return ()
    required = input_schema.get("required")
    required_names = set(required) if isinstance(required, list) else set()
    params = []
    for name, schema in properties.items():
        schema = schema if isinstance(schema, Mapping) else {}
        enum = schema.get("enum")
        params.append(McpParam(
            name=str(name),
            type=_schema_type(schema),
            required=name in required_names,
            enum=tuple(str(item) for item in enum) if isinstance(enum, list) else (),
        ))
    return tuple(params)


# --- Подключение -----------------------------------------------------------

def _server_prefix(server: McpServer) -> str:
    return f"[MCP {server.name}]"


def _names(keys: tuple[str, ...]) -> str:
    return ", ".join(keys)


@dataclass
class _Progress:
    """Что узнали за одно подключение — объект вызова, а не модуля.

    Помнит текущую стадию и всё, что уже известно, чтобы при сбое на списке в
    результате остались факты соединения. Строки «соединение установлено» и
    «tools/list» пишет в момент события: при зависании на списке в терминале
    уже видно, что соединение есть.
    """
    prefix: str
    started: float
    stage: str = STAGE_CONNECT
    server_name: str = ""
    server_version: str = ""
    protocol_version: str = ""
    handshake: str = ""
    capabilities: tuple[str, ...] = ()
    instructions: str = ""
    tools: list[McpTool] = field(default_factory=list)
    pages: list[dict] = field(default_factory=list)
    connect_s: float = 0.0
    list_started: float = 0.0
    list_s: float = 0.0

    def connected(self, client: Client) -> None:
        self.connect_s = time.perf_counter() - self.started
        info = client.server_info
        self.server_name = info.name if info is not None else ""
        self.server_version = info.version if info is not None else ""
        self.protocol_version = client.protocol_version
        self.handshake = (
            HANDSHAKE_DISCOVER if client.session.discover_result is not None
            else HANDSHAKE_INITIALIZE
        )
        caps = client.server_capabilities
        # В порядке полей модели SDK, а не по алфавиту и не как прислал сервер.
        self.capabilities = tuple(
            name for name in type(caps).model_fields if getattr(caps, name) is not None
        )
        self.instructions = client.instructions or ""
        self.stage = STAGE_LIST
        logger.info(
            "%s соединение установлено за %.2f с: %s, протокол %s (%s), возможности: %s",
            self.prefix, self.connect_s, _server_label(self.server_name, self.server_version),
            self.protocol_version, handshake_text(self.handshake),
            _names(self.capabilities) or "—",
        )

    def page(self, page: ListToolsResult) -> None:
        self.pages.append(page.model_dump(mode="json", by_alias=True, exclude_unset=True))
        for tool in page.tools:
            self.tools.append(McpTool(
                name=tool.name,
                title=tool.title or "",
                description=tool.description or "",
                params=tool_params(tool.input_schema),
            ))

    def listed(self) -> None:
        self.list_s = time.perf_counter() - self.list_started
        self.stage = ""
        logger.info(
            "%s tools/list: %d инструментов, страниц: %d, %.2f с — %s",
            self.prefix, len(self.tools), len(self.pages), self.list_s,
            ", ".join(tool.name for tool in self.tools) or "—",
        )


def _server_label(name: str, version: str) -> str:
    """«BGG MCP 1.6.0»; сервер эры 2026-07-28 может не назваться вовсе."""
    return " ".join(part for part in (name, version) if part) or "сервер не назвался"


def handshake_text(handshake: str) -> str:
    """Как договорились — словами для лога и панели."""
    if handshake == HANDSHAKE_DISCOVER:
        return "server/discover, без рукопожатия"
    return "проба server/discover не принята — рукопожатие initialize"


async def _connect_and_list(
    params: StdioServerParameters, progress: _Progress, timeout_s: float,
) -> None:
    # Один потолок на весь путь: запуск, согласование, все страницы и штатное
    # закрытие. Если он истёк, SDK закрывает процесс под защитой от отмены —
    # это добавляет ещё до 2 с.
    with anyio.fail_after(timeout_s):
        # `mode`, `client_info` и `read_timeout_seconds` не задаются: режим —
        # по умолчанию SDK («auto»), общий потолок держит `fail_after`.
        async with Client(params) as client:
            progress.connected(client)
            if client.server_capabilities.tools is None:
                raise _NoToolsError()
            progress.list_started = time.perf_counter()
            cursor: str | None = None
            seen: set[str] = set()
            while True:
                page = await client.list_tools(cursor=cursor)
                progress.page(page)
                cursor = page.next_cursor
                if not cursor:
                    break
                if cursor in seen:
                    raise _RepeatedCursorError(cursor)
                seen.add(cursor)
            progress.listed()
        # Выход из `async with` — закрытие: stdin, до 2 с ожидания, затем SDK
        # завершает процесс сам.


def _first_leaf(exc: BaseException) -> BaseException:
    """Исключения SDK изнутри `async with` приходят обёрнутыми в группу, иногда вложенными."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


_EXPECTED_ERRORS = (MCPError, OSError, TimeoutError, _NoToolsError, _RepeatedCursorError)


def _error_text(exc: BaseException, timeout_s: float) -> str:
    if isinstance(exc, MCPError):
        return f"MCPError({exc.code}): {exc.message}"
    if isinstance(exc, TimeoutError):
        # Своего текста у таймаута `fail_after` нет.
        return ERROR_TIMEOUT.format(seconds=timeout_s)
    if isinstance(exc, (_NoToolsError, _RepeatedCursorError)):
        return str(exc)
    return f"{type(exc).__name__}: {exc}"


def list_tools(server: McpServer, *, timeout_s: float) -> ToolListing:
    """Одно полное подключение: запуск → согласование → tools/list → закрытие.

    Синхронная, исключений на ожидаемых сбоях не бросает — любой сбой
    возвращается `ToolListing` со стадией и текстом.
    """
    at = datetime.now().isoformat(timespec="seconds")
    prefix = _server_prefix(server)
    launch = resolve_launch(server, os.environ)

    if launch.error:
        logger.warning(
            "%s сбой на стадии «%s» через 0.00 с: %s (%s)",
            prefix, STAGE_LAUNCH, launch.error, launch.origin,
        )
        return ToolListing(
            server=server.name, launch=launch, at=at,
            stage=STAGE_LAUNCH, error=launch.error,
        )

    logger.info(
        "%s подключение: %s (%s); серверу передаются: %s%s",
        prefix, launch.command_line, launch.origin, _names(launch.env_set) or "—",
        f" (не заданы {_names(launch.env_missing)})" if launch.env_missing else "",
    )

    # Значения — только для процесса сервера; дальше этой строки они не идут.
    params = StdioServerParameters(
        command=launch.argv[0],
        args=list(launch.argv[1:]),
        env={key: os.environ[key] for key in launch.env_set},
    )
    started = time.perf_counter()
    progress = _Progress(prefix=prefix, started=started)
    stage, error = "", ""
    try:
        anyio.run(_connect_and_list, params, progress, timeout_s)
    except Exception as exc:  # KeyboardInterrupt должен останавливать приложение
        leaf = _first_leaf(exc)
        stage = progress.stage or STAGE_LIST
        # Нет файла, нет прав — процесс не запустился, хоть SDK и был уже
        # внутри подключения. `TimeoutError` — тоже `OSError`, но таймаут
        # значит, что процесс запущен и молчит.
        if (
            isinstance(leaf, OSError) and not isinstance(leaf, TimeoutError)
            and stage == STAGE_CONNECT
        ):
            stage = STAGE_LAUNCH
        error = _error_text(leaf, timeout_s)
        total_s = time.perf_counter() - started
        hint = (
            " — сообщение сервера выше, в его stderr"
            if stage == STAGE_CONNECT and isinstance(leaf, MCPError) else ""
        )
        # Неожиданный тип — скорее ошибка в коде, чем у сервера: с трассировкой.
        logger.warning(
            "%s сбой на стадии «%s» через %.2f с: %s%s",
            prefix, stage, total_s, error, hint,
            exc_info=None if isinstance(leaf, _EXPECTED_ERRORS) else exc,
        )
    else:
        total_s = time.perf_counter() - started
        logger.info("%s соединение закрыто, всего %.2f с", prefix, total_s)

    return ToolListing(
        server=server.name,
        launch=launch,
        at=at,
        stage=stage,
        error=error,
        server_name=progress.server_name,
        server_version=progress.server_version,
        protocol_version=progress.protocol_version,
        handshake=progress.handshake,
        capabilities=progress.capabilities,
        instructions=progress.instructions,
        tools=tuple(progress.tools) if not error else (),
        pages=tuple(progress.pages) if not error else (),
        connect_s=progress.connect_s,
        list_s=progress.list_s,
        total_s=total_s,
    )
