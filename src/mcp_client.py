# TooManyRules — MCP-клиент (день 16, неделя 4; день 17 — вызов инструмента
# и адаптер для агента; день 18 — сервер по адресу и группа серверов).
#
# Единственное место в проекте, где импортируется клиентская часть SDK `mcp`,
# — как `agent.py` единственное место вызова LLM API, а `storage.py` —
# единственное место работы с диском (серверную часть SDK импортируют только
# отдельные программы-серверы `faq_server.py` и, с дня 18, `faq_watch.py` — не
# модули приложения). Модуль
# запускает stdio-сервер, согласует с ним протокол, забирает список
# инструментов по всем страницам (`list_tools()`) или вызывает один
# инструмент (`call_tool()`, день 17), закрывает соединение и отдаёт результат
# одним значением (`ToolListing` / `ToolCall`). Каждый шаг пишет в лог.
#
# С дня 17 здесь же адаптер `McpToolBox`: через него агент получает каталог и
# вызывает инструменты. Агент модуль не импортирует — `McpToolBox` отвечает
# протоколу `agent.ToolBox` структурно, обмен идёт словарями (тот же приём,
# что у хранилищ `storage.py`).
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
# объектов, кроме констант и логгера. Каждый вызов `list_tools()` и
# `call_tool()` — новый процесс и новое соединение; две вкладки — два
# независимых подключения. У `McpToolBox` своего состояния тоже нет, кроме
# описания сервера и потолка времени.
#
# С дня 18 (спецификация дня 18, §5) — второй транспорт: **Streamable HTTP**.
# Сервер с `url` в описании запускает не приложение, а человек; процесс живёт
# своей жизнью, и клиент только подключается по адресу — `Client(url)` вместо
# `Client(StdioServerParameters(...))`. Подключение по-прежнему одно на вызов.
# Окружение такому серверу не передаётся, а сбой «сервер не отвечает» — это
# стадия «соединение», а не «запуск»: запускать модулю нечего. HTTP SDK ходит
# через пакет `httpx2` (не `httpx`), и его ошибки — не `OSError`, поэтому они
# перечислены в ожидаемых отдельно. Там же `McpToolBoxGroup` — несколько
# серверов для агента как один `ToolBox`.
#
# Процесс сервера не наследует окружение приложения: SDK передаёт ему только
# белый список (`INHERITED_ENV`) и то, что перечислено явно (`env_keys`
# описания сервера, если заданы). Значения переменных нигде не показываются и
# не логируются — только имена. Командная строка логируется как есть, поэтому
# ключей в ней не бывает.
#
# Ошибок на ожидаемых сбоях модуль не бросает: любой сбой — `ToolListing` или
# `ToolCall` со стадией и текстом (§4.6), чтобы интерфейсу и агенту не нужно
# было знать про типы ошибок SDK.
#
# С дня 19 (спецификация дня 19, §4) — MCP sampling: сервер посреди
# `tools/call` может попросить модель у клиента. `call_tool()` получает
# необязательный `sampler` — обычную синхронную функцию словарь → словарь
# (модуль по-прежнему не знает LLM, как и раньше не знает Too Many Bones);
# `Client` получает `sampling_callback`, только если `sampler` задан **и**
# сервер разрешает сэмплинг (`McpServer.sampling`) — иначе клиент не
# объявляет возможность `sampling`, и сервер, которому она нужна, откажет
# сам. Обёртка переводит `CreateMessageRequestParams` в словарь, зовёт
# `sampler` через `anyio.to_thread.run_sync()` (вызов модели синхронный и
# долгий, цикл событий подключения он не блокирует) и переводит ответ
# обратно в `CreateMessageResult`. Отказ без вызова модели (не текст в
# просьбе, просьба с `tools`, или `sampler` вернул `ok=False`) — `ErrorData`:
# SDK превращает его в `MCPError` на стороне клиента до второго раунда
# `tools/call`, поэтому такой сбой к серверу не доходит вовсе.

import importlib.metadata
import json
import logging
import os
import shlex
import time
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from datetime import datetime
from urllib.parse import urlsplit

import anyio
import anyio.to_thread
import httpx2
from dotenv import load_dotenv
from mcp import Client, ErrorData, ListToolsResult, MCPError, StdioServerParameters
from mcp.client.stdio import DEFAULT_INHERITED_ENV_VARS
from mcp.types import INTERNAL_ERROR, INVALID_REQUEST, CallToolResult, CreateMessageResult, TextContent

# Модуль сам читает окружение (замену командной строки и переменные для
# сервера), поэтому сам и подхватывает `src/.env` — правило `agent.py`.
# Повторная загрузка безвредна: `load_dotenv()` не перезаписывает заданное.
load_dotenv()

logger = logging.getLogger("toomanyrules.mcp")
# HTTP-клиент SDK (день 18) пишет строку INFO на каждый POST — их по три на
# подключение к HTTP-серверу. Шум: подключение, согласование и вызов модуль
# логирует сам.
logging.getLogger("httpx2").setLevel(logging.WARNING)

# --- Константы -------------------------------------------------------------

STAGE_LAUNCH = "запуск"                 # процесс не запустился (или нечего запускать)
STAGE_CONNECT = "соединение"            # процесс есть, согласование не завершилось
STAGE_LIST = "список инструментов"      # соединение есть, списка нет
STAGE_CALL = "вызов инструмента"        # соединение есть, результата нет (день 17)

HANDSHAKE_INITIALIZE = "initialize"     # рукопожатие эры до 2026-07-28
HANDSHAKE_DISCOVER = "server/discover"  # эра 2026-07-28

SDK_VERSION: str = importlib.metadata.version("mcp")
# Белый список окружения процесса сервера — из SDK, а не второй копией: если
# SDK его поменяет, панель и лог скажут правду.
INHERITED_ENV: tuple[str, ...] = tuple(DEFAULT_INHERITED_ENV_VARS)

ORIGIN_DEFAULT = "по умолчанию"

# Транспорт сервера (день 18, §5.1): stdio — процесс запускает модуль,
# Streamable HTTP — процесс уже работает, модуль подключается по адресу.
TRANSPORT_STDIO = "stdio"
TRANSPORT_HTTP = "streamable-http"
TRANSPORT_LABELS = {TRANSPORT_STDIO: "stdio", TRANSPORT_HTTP: "Streamable HTTP"}

# Тексты причин — нейтральные: модуль не знает, какой сервер запускает.
ERROR_EMPTY_COMMAND = "командная строка пустая"
ERROR_BAD_COMMAND = "командная строка не разбирается: {reason}"
ERROR_TIMEOUT = "таймаут {seconds:g} с"
ERROR_NO_TOOLS = "сервер не объявил возможность tools"
ERROR_REPEATED_CURSOR = "курсор повторился: {cursor}"
ERROR_BAD_URL = "адрес не годится: {reason}"
NO_ANSWER = "сервер не отвечает по адресу"
ERROR_NO_ANSWER = NO_ANSWER + ": {reason}"
ERROR_NOT_IN_GROUP = "инструмента нет в каталоге группы"

# Отказы сэмплинга без вызова модели (день 19, §4.1): SDK превращает их в
# `MCPError` на стороне клиента, до сервера они не доходят.
ERROR_SAMPLING_TOOLS = "инструменты в сэмплинге не поддерживаются"
ERROR_SAMPLING_TEXT_ONLY = "клиент принимает только текст"


# --- Типы ------------------------------------------------------------------

@dataclass(frozen=True)
class McpServer:
    """Как запустить stdio-сервер или где найти HTTP-сервер. Описание — из
    presets.py; модуль про конкретный сервер не знает.

    Сервер с `url` (день 18) — HTTP-сервер, запущенный не приложением:
    `command`, `command_env` и `env_keys` у него пустые, окружение ему не
    передаётся — процесс уже живёт своей жизнью."""
    name: str                       # короткое имя для лога: «bgg»
    title: str                      # для панели
    command: str                    # командная строка по умолчанию — как в терминале
    command_env: str = ""           # переменная окружения, целиком заменяющая command; "" — замены нет
    env_keys: tuple[str, ...] = ()  # переменные, которые передаются серверу, если заданы
    source: str = ""                # ссылка на сервер — для панели
    # Поля дня 18 — в конце, с умолчаниями (§5.1):
    url: str = ""                   # Streamable HTTP: адрес сервера; "" — stdio по command
    url_env: str = ""               # переменная окружения, целиком заменяющая url; "" — замены нет
    # Поле дня 19 — в конце, с умолчанием (§4.1): сэмплинг разрешается серверу
    # явно, а не по умолчанию — иначе чужой сервер, подключённый днём 20,
    # получил бы модель агента за наш счёт. `False` у всех, кроме сторожа.
    sampling: bool = False

    @property
    def transport(self) -> str:
        return TRANSPORT_HTTP if self.url else TRANSPORT_STDIO


@dataclass(frozen=True)
class Launch:
    """Чем будет запущен сервер при данном окружении. Считается чистой функцией (`resolve_launch`).

    У HTTP-сервера (день 18) `command_line` — итоговый адрес, `origin` —
    «по умолчанию» или «из <url_env>», `argv` пуст."""
    command_line: str               # итоговая строка — для лога и панели
    origin: str                     # «по умолчанию» | «из <имя переменной command_env>»
    argv: tuple[str, ...]           # разобранная строка; () — если не разобралась
    env_set: tuple[str, ...]        # какие из env_keys заданы и будут переданы
    env_missing: tuple[str, ...]    # какие не заданы
    error: str = ""                 # пустая или неразбираемая строка
    transport: str = TRANSPORT_STDIO  # день 18: TRANSPORT_STDIO | TRANSPORT_HTTP


@dataclass(frozen=True)
class McpParam:
    """Параметр для колонки таблицы. Описание таблица дня 16 не показывает (оно в
    JSON ответа), таблица своего сервера дня 17 — показывает: это «описание
    входных параметров» из задания."""
    name: str
    type: str                       # «string», «number», «array[number]», «string|null»; "" — у свойства нет type
    required: bool
    enum: tuple[str, ...] = ()
    description: str = ""           # `description` из JSON Schema свойства (день 17)


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


@dataclass(frozen=True)
class ToolCall:
    """Результат одного вызова инструмента — и удачного, и нет (день 17, §4.1).

    Две разные неудачи — два поля: `error` — до ответа не дошли (процесс,
    протокол, таймаут), `is_error` — сервер ответил, что инструмент не
    справился (`isError: true`, текст ошибки — в `text`).
    """
    server: str
    launch: Launch
    at: str
    name: str
    arguments: dict
    stage: str = ""             # "" — дошли до конца; иначе запуск / соединение / вызов инструмента
    error: str = ""             # сбой соединения или протокола; "" — ответ получен
    is_error: bool = False      # сервер ответил, но с isError: ошибка самого инструмента
    text: str = ""              # текстовые части content через перевод строки; не текст — «[<type>]»
    server_name: str = ""
    server_version: str = ""
    protocol_version: str = ""
    handshake: str = ""
    connect_s: float = 0.0
    call_s: float = 0.0
    total_s: float = 0.0
    # Поле дня 19 — в конце (§4.1): сколько просьб к модели клиента выполнил
    # этот вызов. 0 — сервер не разрешён на сэмплинг, `sampler` не передан,
    # или инструмент сэмплинг не запрашивал.
    samples: int = 0

    @property
    def ok(self) -> bool:
        """Ответ получен, и это не ошибка инструмента."""
        return not self.error and not self.is_error


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
    `shlex.split()`, тильда и переменные внутри неё не раскрываются. У
    HTTP-сервера (день 18) — адрес из `url` или замена из `url_env`.
    """
    if server.transport == TRANSPORT_HTTP:
        return _resolve_url(server, environ)
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


def _resolve_url(server: McpServer, environ: Mapping[str, str]) -> Launch:
    """Адрес HTTP-сервера (день 18, §5.1): `http://` или `https://` и хост.
    Пустая замена — нет замены; замена из пробелов — сбой (правило дня 16)."""
    override = environ.get(server.url_env, "") if server.url_env else ""
    if override:
        url, origin = override, f"из {server.url_env}"
    else:
        url, origin = server.url, ORIGIN_DEFAULT
    error = ""
    if not url.strip():
        error = ERROR_BAD_URL.format(reason="пустой")
    else:
        parts = urlsplit(url)
        if parts.scheme not in ("http", "https"):
            error = ERROR_BAD_URL.format(reason=f"нужен http:// или https:// — «{url}»")
        elif not parts.hostname:
            error = ERROR_BAD_URL.format(reason=f"нет хоста — «{url}»")
    return Launch(
        command_line=url,
        origin=origin,
        argv=(),
        env_set=(),
        env_missing=(),
        error=error,
        transport=TRANSPORT_HTTP,
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
            description=str(schema.get("description") or ""),
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

    Помнит текущую стадию и всё, что уже известно, чтобы при сбое на списке
    или на вызове в результате остались факты соединения. Строки «соединение
    установлено», «tools/list» и «tools/call» пишет в момент события: при
    зависании на списке или на вызове в терминале уже видно, что соединение
    есть. `after_connect` — стадия после соединения: список инструментов у
    `list_tools()`, вызов инструмента у `call_tool()` (день 17).
    """
    prefix: str
    started: float
    after_connect: str = STAGE_LIST
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
    # Вызов инструмента (день 17):
    call_started: float = 0.0
    call_s: float = 0.0
    text: str = ""
    is_error: bool = False
    # Сэмплинг (день 19, §4.1): сколько просьб к модели клиента выполнено.
    samples: int = 0

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
        self.stage = self.after_connect
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

    def called(self, name: str, arguments: dict, result: CallToolResult) -> None:
        self.call_s = time.perf_counter() - self.call_started
        # Текстовые части — через перевод строки; не текст (картинка, ресурс)
        # — пометкой с типом: модели отдаётся текст, а не байты.
        self.text = "\n".join(
            part.text if part.type == "text" else f"[{part.type}]"
            for part in result.content
        )
        self.is_error = bool(result.is_error)
        self.stage = ""
        call = f"{self.prefix} tools/call {name} {_arguments_str(arguments)}"
        # Ошибка инструмента — ответ сервера, а не сбой клиента: INFO здесь,
        # предупреждение пишет агент, для которого это событие хода.
        if self.is_error:
            logger.info(
                "%s: ошибка инструмента за %.2f с — %s", call, self.call_s, self.text,
            )
        else:
            logger.info(
                "%s: ответ за %.2f с, %d символов", call, self.call_s, len(self.text),
            )

    def sampled(self, request: dict, response: dict) -> None:
        """Строка лога на одну просьбу сэмплинга (день 19, §4.1) — до ответа
        `_run_sampling()` возвращает результат агента, здесь только факт
        обмена: сколько сообщений ушло, что пришло."""
        self.samples += 1
        messages = request.get("messages") or []
        chars = len(request.get("system") or "") + sum(
            len(str(item.get("text") or "")) for item in messages
        )
        if response.get("ok"):
            logger.info(
                "%s sampling/createMessage: сообщений %d, %d символов, max_tokens %s → "
                "%d символов (%s, %s)",
                self.prefix, len(messages), chars, request.get("max_tokens"),
                len(response.get("text") or ""), response.get("model") or "?",
                response.get("finish_reason") or "endTurn",
            )
        else:
            logger.warning(
                "%s sampling/createMessage: отказ — %s", self.prefix, response.get("error") or "",
            )


def _arguments_str(arguments: dict) -> str:
    """Аргументы для лога: это не секреты, а номер статьи и имя раздела."""
    return json.dumps(arguments, ensure_ascii=False)


def _server_label(name: str, version: str) -> str:
    """«BGG MCP 1.6.0»; сервер эры 2026-07-28 может не назваться вовсе."""
    return " ".join(part for part in (name, version) if part) or "сервер не назвался"


def handshake_text(handshake: str) -> str:
    """Как договорились — словами для лога и панели."""
    if handshake == HANDSHAKE_DISCOVER:
        return "server/discover, без рукопожатия"
    return "проба server/discover не принята — рукопожатие initialize"


async def _connect_and_list(
    params: StdioServerParameters | str, progress: _Progress, timeout_s: float,
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


def _build_sampling_callback(progress: _Progress, sampler: Callable[[dict], dict]):
    """Оборачивает `sampler` (день 19, §4.1) в `sampling_callback` SDK:
    `CreateMessageRequestParams` → словарь → `sampler` (в чужом потоке,
    `anyio.to_thread.run_sync()`) → `CreateMessageResult`/`ErrorData`.

    Отказ без вызова модели — `ErrorData`: SDK сам превращает его в
    `MCPError` на стороне клиента, второго раунда `tools/call` не будет, и до
    тела инструмента сбой не дойдёт (§2.9, §4.3). Поток не отменяется
    (`abandon_on_cancel` не задаётся, умолчание SDK — `False`): истёкший
    `fail_after` закроет подключение только после ответа модели, а не
    оборвёт вызов на середине.
    """

    async def sampling_callback(_context, params) -> CreateMessageResult | ErrorData:
        if params.tools or params.tool_choice:
            return ErrorData(code=INVALID_REQUEST, message=ERROR_SAMPLING_TOOLS)
        messages: list[dict] = []
        for message in params.messages:
            content = message.content
            if not isinstance(content, TextContent):
                return ErrorData(code=INVALID_REQUEST, message=ERROR_SAMPLING_TEXT_ONLY)
            messages.append({"role": message.role, "text": content.text})
        request = {
            "system": params.system_prompt or "",
            "messages": messages,
            "max_tokens": params.max_tokens,
        }
        response = await anyio.to_thread.run_sync(sampler, request)
        progress.sampled(request, response)
        if not response.get("ok"):
            return ErrorData(code=INTERNAL_ERROR, message=response.get("error") or "сбой сэмплинга")
        return CreateMessageResult(
            role="assistant",
            content=TextContent(type="text", text=response.get("text") or ""),
            model=response.get("model") or "",
            stop_reason=response.get("finish_reason") or "endTurn",
        )

    return sampling_callback


async def _connect_and_call(
    params: StdioServerParameters | str, progress: _Progress, timeout_s: float,
    name: str, arguments: dict, sampler: Callable[[dict], dict] | None,
) -> None:
    # Тот же путь, что `_connect_and_list()`, но вместо листания — один
    # `tools/call`. Потолок `fail_after` — на всё подключение, как у списка.
    # `sampling_callback` (день 19, §4.1) объявляет возможность sampling
    # клиенту только когда `sampler` задан — вызывающий уже решил, разрешён
    # ли сэмплинг серверу (`McpServer.sampling`).
    sampling_callback = _build_sampling_callback(progress, sampler) if sampler is not None else None
    with anyio.fail_after(timeout_s):
        async with Client(params, sampling_callback=sampling_callback) as client:
            progress.connected(client)
            if client.server_capabilities.tools is None:
                raise _NoToolsError()
            progress.call_started = time.perf_counter()
            result = await client.call_tool(name, arguments)
            progress.called(name, arguments, result)


def _first_leaf(exc: BaseException) -> BaseException:
    """Исключения SDK изнутри `async with` приходят обёрнутыми в группу, иногда вложенными."""
    while isinstance(exc, BaseExceptionGroup) and exc.exceptions:
        exc = exc.exceptions[0]
    return exc


# `httpx2.HTTPError` (день 18, §5.3) — HTTP-клиент SDK: сервер не запущен
# (`ConnectError`), оборван, ответил не тем статусом. Это не `OSError`, и без
# этой строки сбой шёл бы в лог с трассировкой как ошибка в коде.
_EXPECTED_ERRORS = (
    MCPError, OSError, TimeoutError, httpx2.HTTPError, _NoToolsError, _RepeatedCursorError,
)


def is_no_answer(error: str) -> bool:
    """Сбой — «сервер по адресу не отвечает» (не запущен или другой порт), а
    не таймаут и не чужой ответ (день 18): панели — чтобы подсказывать запуск
    сервера только тогда, когда он и правда не запущен."""
    return error.startswith(NO_ANSWER)


def _error_text(exc: BaseException, timeout_s: float) -> str:
    if isinstance(exc, MCPError):
        return f"MCPError({exc.code}): {exc.message}"
    if isinstance(exc, TimeoutError):
        # Своего текста у таймаута `fail_after` нет.
        return ERROR_TIMEOUT.format(seconds=timeout_s)
    if isinstance(exc, (_NoToolsError, _RepeatedCursorError)):
        return str(exc)
    if isinstance(exc, (httpx2.ConnectError, httpx2.ConnectTimeout)):
        # Сервер не запущен. Как его запустить, модуль не знает — подсказку
        # показывает панель.
        return ERROR_NO_ANSWER.format(reason=f"{type(exc).__name__}: {exc}")
    return f"{type(exc).__name__}: {exc}"


def _start(server: McpServer) -> tuple[str, str, Launch]:
    """Общее начало подключения: момент, префикс лога и описание запуска."""
    return (
        datetime.now().isoformat(timespec="seconds"),
        _server_prefix(server),
        resolve_launch(server, os.environ),
    )


def _launch_failed(prefix: str, launch: Launch) -> None:
    logger.warning(
        "%s сбой на стадии «%s» через 0.00 с: %s (%s)",
        prefix, STAGE_LAUNCH, launch.error, launch.origin,
    )


def _run(connect, prefix: str, launch: Launch, progress: _Progress,
         timeout_s: float, *args) -> tuple[str, str, float]:
    """Одно подключение `connect` под `anyio.run()` — общий хвост
    `list_tools()` и `call_tool()`: строка «подключение», разбор сбоя по
    стадиям, строка «соединение закрыто». Возвращает стадию сбоя, текст сбоя
    (оба "" при успехе) и полное время."""
    http = launch.transport == TRANSPORT_HTTP
    if http:
        # Сервер уже работает — ни процесса, ни окружения: только адрес.
        logger.info(
            "%s подключение: %s (%s), транспорт %s",
            prefix, launch.command_line, launch.origin, TRANSPORT_LABELS[TRANSPORT_HTTP],
        )
        params: StdioServerParameters | str = launch.command_line
    else:
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
    stage, error = "", ""
    try:
        anyio.run(connect, params, progress, timeout_s, *args)
    except Exception as exc:  # KeyboardInterrupt должен останавливать приложение
        leaf = _first_leaf(exc)
        stage = progress.stage or progress.after_connect
        # Нет файла, нет прав — процесс не запустился, хоть SDK и был уже
        # внутри подключения. `TimeoutError` — тоже `OSError`, но таймаут
        # значит, что процесс запущен и молчит. У HTTP-сервера процесса
        # модуль не запускает — стадия остаётся «соединение» (день 18).
        if (
            not http and isinstance(leaf, OSError) and not isinstance(leaf, TimeoutError)
            and stage == STAGE_CONNECT
        ):
            stage = STAGE_LAUNCH
        error = _error_text(leaf, timeout_s)
        total_s = time.perf_counter() - progress.started
        # stderr процесса есть только у stdio-сервера: он в терминале
        # приложения. HTTP-сервер пишет в свой терминал.
        hint = (
            " — сообщение сервера выше, в его stderr"
            if not http and stage == STAGE_CONNECT and isinstance(leaf, MCPError) else ""
        )
        # Неожиданный тип — скорее ошибка в коде, чем у сервера: с трассировкой.
        logger.warning(
            "%s сбой на стадии «%s» через %.2f с: %s%s",
            prefix, stage, total_s, error, hint,
            exc_info=None if isinstance(leaf, _EXPECTED_ERRORS) else exc,
        )
    else:
        total_s = time.perf_counter() - progress.started
        logger.info("%s соединение закрыто, всего %.2f с", prefix, total_s)
    return stage, error, total_s


def list_tools(server: McpServer, *, timeout_s: float) -> ToolListing:
    """Одно полное подключение: запуск → согласование → tools/list → закрытие.

    Синхронная, исключений на ожидаемых сбоях не бросает — любой сбой
    возвращается `ToolListing` со стадией и текстом.
    """
    at, prefix, launch = _start(server)
    if launch.error:
        _launch_failed(prefix, launch)
        return ToolListing(
            server=server.name, launch=launch, at=at,
            stage=STAGE_LAUNCH, error=launch.error,
        )

    progress = _Progress(prefix=prefix, started=time.perf_counter())
    stage, error, total_s = _run(_connect_and_list, prefix, launch, progress, timeout_s)
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


def call_tool(
    server: McpServer, name: str, arguments: dict, *, timeout_s: float,
    sampler: Callable[[dict], dict] | None = None,
) -> ToolCall:
    """Одно полное подключение: запуск → согласование → tools/call → закрытие
    (день 17, §4.1).

    Синхронная, исключений на ожидаемых сбоях не бросает — ровно как
    `list_tools()`. Сбой соединения или протокола — `error` со стадией; ответ
    сервера с `isError` — `is_error` и текст ошибки в `text`.

    С дня 19 (§4.1) `sampler` — функция сэмплинга для этого вызова; решает,
    пользоваться ли ею, сервер (`McpServer.sampling`): без разрешения клиент
    не объявляет возможность sampling, даже если `sampler` передан.
    """
    at, prefix, launch = _start(server)
    if launch.error:
        _launch_failed(prefix, launch)
        return ToolCall(
            server=server.name, launch=launch, at=at, name=name,
            arguments=dict(arguments), stage=STAGE_LAUNCH, error=launch.error,
        )

    progress = _Progress(
        prefix=prefix, started=time.perf_counter(), after_connect=STAGE_CALL,
    )
    active_sampler = sampler if (sampler is not None and server.sampling) else None
    stage, error, total_s = _run(
        _connect_and_call, prefix, launch, progress, timeout_s, name, arguments, active_sampler,
    )
    return ToolCall(
        server=server.name,
        launch=launch,
        at=at,
        name=name,
        arguments=dict(arguments),
        stage=stage,
        error=error,
        is_error=progress.is_error if not error else False,
        text=progress.text if not error else "",
        server_name=progress.server_name,
        server_version=progress.server_version,
        protocol_version=progress.protocol_version,
        handshake=progress.handshake,
        connect_s=progress.connect_s,
        call_s=progress.call_s,
        total_s=total_s,
        samples=progress.samples,
    )


# --- Адаптер для агента (день 17, §4.2) ------------------------------------

class McpToolBox:
    """Инструменты одного MCP-сервера для агента: каталог и вызов, каждый раз
    новое соединение.

    Отвечает протоколу `agent.ToolBox` структурно и его не импортирует: обмен
    идёт словарями, как у хранилищ `storage.py`. Один на процесс и общий для
    всех агентов — состояния у него нет, кроме описания сервера и потолков
    времени. Исключений на ожидаемых сбоях не бросает, как и функции выше.

    С дня 19 (§4.2) у вызова свой потолок — `call_timeout_s`, отдельно от
    потолка каталога (`timeout_s`): вызов с сэмплингом ждёт ответа модели
    клиента, а каталог — нет, и общий потолок держал бы ход лишнюю минуту,
    если сторож завис.
    """

    def __init__(
        self, server: McpServer, *, timeout_s: float, call_timeout_s: float | None = None,
    ) -> None:
        self._server = server
        self._timeout_s = timeout_s
        self._call_timeout_s = call_timeout_s if call_timeout_s is not None else timeout_s

    @property
    def name(self) -> str:
        """Короткое имя сервера — для логов агента и строки каталога."""
        return self._server.name

    def catalog(self) -> dict:
        """Каталог инструментов: `{"ok", "error", "elapsed", "tools": [{"name",
        "description", "input_schema"}, …]}`. `input_schema` — `inputSchema`
        из ответа как пришла (из `pages`, по именам полей протокола)."""
        listing = list_tools(self._server, timeout_s=self._timeout_s)
        if not listing.ok:
            return {
                "ok": False,
                "error": f"сбой на стадии «{listing.stage}»: {listing.error}",
                "elapsed": listing.total_s,
                "tools": [],
            }
        tools = [
            {
                "name": tool["name"],
                "description": tool.get("description", ""),
                "input_schema": tool.get("inputSchema") or {"type": "object", "properties": {}},
            }
            for page in listing.pages
            for tool in page.get("tools", [])
        ]
        return {"ok": True, "error": "", "elapsed": listing.total_s, "tools": tools}

    def call(self, name: str, arguments: dict, sample: Callable[[dict], dict] | None = None) -> dict:
        """Вызов инструмента: `{"ok", "is_error", "error", "stage", "text",
        "elapsed", "samples"}`. `error` и `stage` — сбой соединения или
        протокола (до ответа не дошли); `is_error` — сервер ответил ошибкой
        инструмента, её текст — в `text`.

        С дня 19 (§4.2) `sample` уходит в `call_tool()` как `sampler` —
        пользуется ли им сервер, решает `McpServer.sampling`, не этот метод."""
        call = call_tool(
            self._server, name, arguments, timeout_s=self._call_timeout_s, sampler=sample,
        )
        return {
            "ok": call.ok,
            "is_error": call.is_error,
            "error": call.error,
            "stage": call.stage,
            "text": call.text,
            "elapsed": call.total_s,
            "samples": call.samples,
        }


# --- Группа серверов для агента (день 18, §5.4) ----------------------------

class McpToolBoxGroup:
    """Несколько `McpToolBox` как один `ToolBox`: каталоги сливаются, вызов
    уходит серверу, который объявил инструмент в последнем каталоге.

    Агент при этом не меняется — группа отвечает протоколу `agent.ToolBox`
    структурно, как и отдельный `McpToolBox`. Один сервер недоступен — ход идёт
    с инструментами остальных, а сбой уходит в необязательный ключ
    `"warning"` каталога; ни одного — каталог не получен, как у одного сервера.

    Единственное состояние — карта «имя инструмента → сервер», которую
    удачные каталоги только пополняют: каталог дописывает свои имена в копию
    карты и подменяет её одним присваиванием, читается она без замка. Имена
    не удаляются (правка по ревью дня 18): иначе каталог хода другого агента,
    снятый, пока сторож лежит, выкинул бы из карты инструмент, законный по
    каталогу этого хода. Набор инструментов у сервера постоянный — устаревшая
    запись стоит самое большее одного неудачного подключения с честной
    ошибкой.
    """

    def __init__(self, boxes: tuple[McpToolBox, ...]) -> None:
        self._boxes = boxes
        self._routes: dict[str, McpToolBox] = {}

    @property
    def name(self) -> str:
        """Имена серверов через «+» — для логов агента и строки каталога: «faq+watch»."""
        return "+".join(box.name for box in self._boxes)

    def catalog(self) -> dict:
        """Каталоги серверов по очереди, в порядке `boxes`; инструменты
        сливаются в том же порядке. Имя, уже встреченное у предыдущего
        сервера, пропускается с предупреждением. Формат — как у
        `McpToolBox.catalog()` плюс необязательный `"warning"`; `elapsed` —
        сумма."""
        tools: list[dict] = []
        routes: dict[str, McpToolBox] = {}
        failures: list[str] = []
        warnings: list[str] = []
        elapsed = 0.0
        received = 0
        for box in self._boxes:
            catalog = box.catalog()
            elapsed += catalog.get("elapsed") or 0.0
            if not catalog.get("ok"):
                failures.append(f"{box.name}: {catalog.get('error') or 'причина не названа'}")
                continue
            received += 1
            for tool in catalog.get("tools") or []:
                owner = routes.get(tool["name"])
                if owner is not None:
                    warnings.append(
                        f"{box.name}: инструмент {tool['name']} уже есть у {owner.name} — пропущен"
                    )
                    continue
                routes[tool["name"]] = box
                tools.append(tool)
        if not received:
            return {"ok": False, "error": "; ".join(failures), "elapsed": elapsed, "tools": []}
        self._routes = {**self._routes, **routes}
        result = {"ok": True, "error": "", "elapsed": elapsed, "tools": tools}
        if failures or warnings:
            result["warning"] = "; ".join(failures + warnings)
        return result

    def call(self, name: str, arguments: dict, sample: Callable[[dict], dict] | None = None) -> dict:
        """Вызов — серверу инструмента по карте, которую пополняют каталоги.
        Имени нет в карте — ответ без подключения (агент и так не отправляет
        имена вне каталога хода, это защита). `sample` (день 19, §4.2) уходит
        тому же серверу — пользуется ли им сервер инструмента, решает он сам."""
        box = self._routes.get(name)
        if box is None:
            return {
                "ok": False,
                "is_error": False,
                "error": ERROR_NOT_IN_GROUP,
                "stage": "",
                "text": "",
                "elapsed": 0.0,
                "samples": 0,
            }
        return box.call(name, arguments, sample)
