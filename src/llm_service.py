# TooManyRules — локальная модель как приватный сервис (день 30, неделя 6).
#
# **Отдельная долгоживущая программа**, как сервер FAQ дня 18: её запускает
# человек (`./run.sh service`), приложение её не импортирует и не запускает, без
# неё приложение работает как на дне 29. Сервис — OpenAI-совместимый прокси
# перед LM Studio для домашней сети: слушает `SERVICE_HOST:SERVICE_PORT`, на `/`
# отдаёт страницу чата (`llm_service_chat.html`), на `/v1/chat/completions`
# пересылает запрос клиента в LM Studio. Перед пересылкой — по порядку (§3.4
# спецификации): ключ клиента, размер тела, форма запроса, `stream`, лимит
# клиента в минуту, окно модели, очередь.
#
# **Модель от себя сервис не вызывает** — LLM API по-прежнему вызывается только в
# `agent.py`. Сервис пересылает чужой запрос как есть (меняется только
# `max_tokens`) и отдаёт ответ LM Studio как есть: это прокси, а не второй агент.
# LM Studio он не управляет: не запускает сервер, не грузит и не выгружает
# модели, поле `model` не разбирает — только спрашивает `local_server.check()`,
# какая модель загружена и какое у неё окно, и подсказывает команду.
#
# **Приватность**: ключи — только в `src/.env` (`presets.SERVICE_KEYS_ENV`), ни в
# логе, ни в ответах, ни в `/health`; на диск сервис ничего не пишет; в логе
# (stdout, префикс `[Сервис]`) — клиент, время, токены и статус, но не тексты
# вопросов и ответов. Очередь и лимиты — память процесса, перезапуск их
# обнуляет.
#
# Из проекта импортирует `presets.py` (константы), `tokens.py` (оценка токенов,
# та же эвристика, что в панели дня 8), `local_server.py` (окно и id загруженной
# модели) и константу `agent.CONTEXT_OVERFLOW_MARKERS`; `Agent` не создаёт.
# Разбор ключей (`parse_keys()`) берёт и программа нагрузки `service_load.py` —
# ребро между программами, как `rag_dialog.py` → `rag_eval.py`.
#
# Коды выхода: 0 — остановлен Ctrl+C или SIGTERM, 2 — отказ старта (ключи, порт,
# `--upstream` на сам сервис, нет страницы чата).

import argparse
import asyncio
import hmac
import ipaddress
import json
import logging
import math
import os
import secrets
import signal
import socket
import sys
import time
from collections import deque
from contextlib import asynccontextmanager, closing
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import quote, urlsplit

import httpx
import uvicorn
from fastapi import FastAPI, Request
from fastapi.responses import HTMLResponse, JSONResponse, Response
from starlette.exceptions import HTTPException as StarletteHTTPException

import local_server
import presets
import tokens
from agent import CONTEXT_OVERFLOW_MARKERS

logger = logging.getLogger("llm_service")

CHAT_PAGE = Path(__file__).resolve().with_name("llm_service_chat.html")

# Скользящее окно лимита запросов, секунды (§3.6).
RATE_WINDOW_S = 60
# `Retry-After` у 503 до первого ответа модели и наименьший, секунды (§3.5).
RETRY_AFTER_DEFAULT_S = 5
RETRY_AFTER_MIN_S = 1
# Сколько последних ответов модели усредняется для `Retry-After`.
RECENT_ANSWERS = 5
# Как часто ожидающий в очереди проверяет, не отключился ли клиент, секунды.
DISCONNECT_POLL_S = 0.5
# Сколько ждать незаконченные запросы при остановке, секунды: ожидающие в очереди
# получают обрыв соединения (§3.10).
SHUTDOWN_GRACE_S = 1

# Адреса, по которым `--upstream` указывал бы на сам сервис (§3.1, п. 3).
_SELF_HOSTS = {"127.0.0.1", "localhost", "0.0.0.0", "::1", "[::1]"}


# --- Ключи клиентов (§3.2) ---------------------------------------------------------

@dataclass(frozen=True)
class Client:
    """Клиент сервиса: имя (в логе и лимите), ключ и лимит запросов в минуту."""

    name: str
    key: str
    rate: int


class KeysError(ValueError):
    """Ключи не разбираются. Текст — без ключей: он печатается в лог."""


def parse_keys(raw: str | None, default_rate: int = presets.SERVICE_RATE_PER_MIN) -> list[Client]:
    """Записи `имя:ключ` или `имя:ключ:лимит` через запятую. Пустые записи (лишняя
    запятая) пропускаются. Ошибка — `KeysError` с номером записи, **без текста
    ключа**; ни одной записи — тоже ошибка."""
    clients: list[Client] = []
    records = [record.strip() for record in (raw or "").split(",")]
    for number, record in enumerate(records, start=1):
        if not record:
            continue
        parts = record.split(":")
        if len(parts) not in (2, 3):
            raise KeysError(f"запись {number}: нужно «имя:ключ» или «имя:ключ:лимит» (через двоеточие)")
        name, key = parts[0].strip(), parts[1].strip()
        if not name:
            raise KeysError(f"запись {number}: пустое имя клиента")
        if not key:
            raise KeysError(f"запись {number} («{name}»): пустой ключ")
        if len(key) < presets.SERVICE_KEY_MIN_CHARS:
            raise KeysError(
                f"запись {number} («{name}»): ключ короче {presets.SERVICE_KEY_MIN_CHARS} символов"
            )
        rate = default_rate
        if len(parts) == 3:
            text = parts[2].strip()
            if not text.isdigit() or int(text) < 1:
                raise KeysError(f"запись {number} («{name}»): лимит — целое число запросов в минуту, не меньше 1")
            rate = int(text)
        if any(client.name == name for client in clients):
            raise KeysError(f"запись {number}: имя «{name}» уже есть")
        if any(hmac.compare_digest(client.key.encode(), key.encode()) for client in clients):
            raise KeysError(f"запись {number} («{name}»): такой ключ уже есть у другого клиента")
        clients.append(Client(name, key, rate))
    if not clients:
        raise KeysError(f"нет ни одного ключа в {presets.SERVICE_KEYS_ENV}")
    return clients


def keys_hint() -> str:
    return (
        f"формат: {presets.SERVICE_KEYS_ENV}=телефон:<ключ>:3,ноутбук:<ключ> в src/.env "
        f"(лимит — запросов в минуту, без него {presets.SERVICE_RATE_PER_MIN}); ключ даст "
        f"./run.sh service --new-key"
    )


def find_client(clients: list[Client], token: str) -> Client | None:
    """Клиент по ключу. Сравнение — `hmac.compare_digest` со всеми ключами, без
    раннего выхода."""
    found = None
    for client in clients:
        if hmac.compare_digest(client.key.encode(), token.encode()):
            found = client
    return found


# --- Лимит запросов (§3.6) ---------------------------------------------------------

class RateLimiter:
    """Скользящее окно `RATE_WINDOW_S` на клиента. В памяти процесса."""

    def __init__(self) -> None:
        self._hits: dict[str, deque[float]] = {}

    def take(self, client: Client, now: float) -> tuple[bool, int, int]:
        """(пропущен, осталось в окне, `Retry-After`). Отказанный запрос в окно не
        добавляется."""
        hits = self._hits.setdefault(client.name, deque())
        while hits and hits[0] <= now - RATE_WINDOW_S:
            hits.popleft()
        if len(hits) >= client.rate:
            return False, 0, max(1, math.ceil(hits[0] + RATE_WINDOW_S - now))
        hits.append(now)
        return True, client.rate - len(hits), 0


# --- Очередь (§3.5) -----------------------------------------------------------------

class QueueFull(Exception):
    pass


class ModelQueue:
    """FIFO: в модели не больше `concurrency`, ждут не больше `max_waiting`.

    Место в модели передаётся следующему ожидающему напрямую (`release()`), без
    гонки за освободившийся слот. Ожидающий, которого отменили (таймаут, клиент
    ушёл), из очереди убирается; если место ему уже передали — оно отдаётся
    следующему.
    """

    def __init__(self, concurrency: int, max_waiting: int) -> None:
        self.concurrency = concurrency
        self.max_waiting = max_waiting
        self.active = 0
        self._waiters: deque[asyncio.Future] = deque()
        self._logged: tuple[int, int] = (0, 0)

    @property
    def waiting(self) -> int:
        return len(self._waiters)

    def _changed(self) -> None:
        """Строка лога, когда меняется число ожидающих: «в модели 1, ждут 3».
        Один запрос без очереди строк не даёт — их хватает в строке запроса."""
        state = (self.active, self.waiting)
        if state[1] != self._logged[1]:
            logger.info("очередь: в модели %d, ждут %d", *state)
        self._logged = state

    async def acquire(self, timeout_s: float) -> None:
        """Место в модели. Очередь полна — `QueueFull`; не дождались —
        `TimeoutError`; отменили — `CancelledError` (место не занято)."""
        if self.active < self.concurrency and not self._waiters:
            self.active += 1
            self._changed()
            return
        if len(self._waiters) >= self.max_waiting:
            raise QueueFull
        future = asyncio.get_running_loop().create_future()
        self._waiters.append(future)
        self._changed()
        try:
            await asyncio.wait_for(future, timeout_s)
        except BaseException:
            if future.done() and not future.cancelled():
                # Место передали в тот же миг, что отменили ожидание.
                self.release()
            else:
                future.cancel()
                try:
                    self._waiters.remove(future)
                except ValueError:
                    pass
                self._changed()
            raise

    def release(self) -> None:
        while self._waiters:
            future = self._waiters.popleft()
            if not future.done():
                future.set_result(None)   # место переходит к нему, active не меняется
                self._changed()
                return
        self.active -= 1
        self._changed()


# --- Окно модели (§3.7) -------------------------------------------------------------

@dataclass(frozen=True)
class ModelState:
    """Что известно о LM Studio: id загруженной модели, окно и состояние."""

    at: float
    model: str | None
    context: int | None
    upstream: str          # "ok" или причина


def model_state(check: local_server.ServerCheck) -> ModelState:
    status = check.status
    now = time.monotonic()
    if status == local_server.STATUS_NO_ANSWER:
        return ModelState(now, None, None, f"LM Studio не отвечает по {check.url} — lms server start")
    if status == local_server.STATUS_FAILED:
        return ModelState(now, None, None, check.error)
    if status == local_server.STATUS_NOT_LOADED:
        return ModelState(now, None, None, "модель не загружена — lms load <id>")
    if status == local_server.STATUS_SEVERAL:
        names = ", ".join(model.id for model in check.loaded)
        return ModelState(now, None, None, f"загружено несколько моделей ({names}) — оставьте одну: lms unload <id>")
    if status == local_server.STATUS_UNKNOWN:
        return ModelState(now, None, None, "ok")
    model = check.loaded[0]
    return ModelState(now, model.id, model.loaded_context, "ok")


def estimate_request(messages: list[dict]) -> int:
    """Оценка вопроса в токенах — как `tokens.count_request()` для сообщений:
    служебные токены запроса и каждого сообщения плюс `estimate_tokens()` текста."""
    total = tokens.TOKENS_PER_REQUEST
    for message in messages:
        total += tokens.TOKENS_PER_MESSAGE + tokens.estimate_tokens(message_text(message))
    return total


def message_text(message: dict) -> str:
    content = message.get("content")
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "\n".join(part.get("text", "") for part in content if isinstance(part, dict))
    return ""


def validate_messages(payload: object) -> str:
    """Пустая строка — запрос в порядке; иначе что не так (§3.4, п. 3)."""
    if not isinstance(payload, dict):
        return "тело запроса — не JSON-объект"
    messages = payload.get("messages")
    if not isinstance(messages, list) or not messages:
        return "нужен непустой список messages"
    for number, message in enumerate(messages, start=1):
        if not isinstance(message, dict):
            return f"сообщение {number} — не объект"
        if not isinstance(message.get("role"), str) or not message["role"]:
            return f"у сообщения {number} нет role"
        content = message.get("content")
        if isinstance(content, str):
            continue
        if isinstance(content, list):
            if all(isinstance(part, dict) and part.get("type") == "text" and isinstance(part.get("text"), str)
                   for part in content):
                continue
            return f"сообщение {number}: поддерживаются только текстовые части content"
        # Раунд function calling агента: у ответа модели с tool_calls content
        # бывает пустым (null) — это не «не текст».
        if content is None and message["role"] == "assistant" and message.get("tool_calls"):
            continue
        return f"у сообщения {number} content — не строка и не список текстовых частей"
    max_tokens = payload.get("max_tokens")
    if max_tokens is not None and (not isinstance(max_tokens, int) or isinstance(max_tokens, bool) or max_tokens < 1):
        return "max_tokens — целое число, не меньше 1"
    return ""


# --- Ответы об ошибках (§3.3, §9) ------------------------------------------------------

_ERROR_TYPES = {
    400: "invalid_request_error", 401: "authentication_error", 404: "invalid_request_error",
    405: "invalid_request_error", 413: "invalid_request_error", 429: "rate_limit_error",
}


def error_response(status: int, code: str, message: str, headers: dict | None = None) -> JSONResponse:
    """Ошибка в форме OpenAI API: SDK `openai` показывает `message`."""
    body = {"error": {"message": message, "type": _ERROR_TYPES.get(status, "server_error"), "code": code}}
    return JSONResponse(body, status_code=status, headers=headers)


def upstream_message(response: httpx.Response) -> str:
    """Текст ошибки LM Studio: `error` строкой, `error.message` или тело как есть."""
    try:
        payload = response.json()
    except ValueError:
        return response.text.strip()[:500] or f"{response.status_code} {response.reason_phrase}"
    error = payload.get("error") if isinstance(payload, dict) else None
    if isinstance(error, dict):
        error = error.get("message")
    if isinstance(error, str) and error:
        return error
    return json.dumps(payload, ensure_ascii=False)[:500]


def header_text(value: str) -> str:
    """Значение заголовка — только latin-1: имя клиента («телефон») кодируется
    процентами, страница чата раскодирует его `decodeURIComponent()`."""
    return quote(value, safe="")


# --- Сервис ----------------------------------------------------------------------------

@dataclass
class Settings:
    host: str
    port: int
    upstream: str
    clients: list[Client]
    page: str


class Service:
    """Состояние процесса: клиенты, лимит, очередь, окно модели, клиент HTTP."""

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.upstream = settings.upstream.rstrip("/")
        self.limiter = RateLimiter()
        self.queue = ModelQueue(presets.SERVICE_CONCURRENCY, presets.SERVICE_QUEUE_MAX)
        self.recent: deque[float] = deque(maxlen=RECENT_ANSWERS)
        self.http: httpx.AsyncClient | None = None
        self._state: ModelState | None = None
        self._state_lock = asyncio.Lock()

    # Окно модели — кэш на `SERVICE_MODEL_REFRESH_S`.
    async def state(self) -> ModelState:
        async with self._state_lock:
            stale = self._state is None or time.monotonic() - self._state.at > presets.SERVICE_MODEL_REFRESH_S
            if stale:
                check = await asyncio.to_thread(local_server.check, self.upstream, presets.LOCAL_CHECK_TIMEOUT_S)
                self._state = model_state(check)
            return self._state

    def forget_state(self) -> None:
        self._state = None

    def retry_after(self) -> int:
        """Оценка для 503: (ожидающих + 1) × среднее время последних ответов."""
        if not self.recent:
            return RETRY_AFTER_DEFAULT_S
        average = sum(self.recent) / len(self.recent)
        return max(RETRY_AFTER_MIN_S, math.ceil((self.queue.waiting + 1) * average))

    def authorize(self, request: Request) -> tuple[Client | None, str]:
        """(клиент, подпись для лога и отказа). Подпись — «без ключа» или «чужой
        ключ»; ключ нигде не печатается."""
        header = request.headers.get("authorization", "")
        scheme, _, token = header.partition(" ")
        token = token.strip()
        if scheme.lower() != "bearer" or not token:
            return None, "без ключа"
        client = find_client(self.settings.clients, token)
        return (client, client.name) if client else (None, "чужой ключ")


def who(label: str, request: Request, client: Client | None) -> str:
    """Клиент для лога; у отказа по ключу — ещё адрес отправителя."""
    address = request.client.host if request.client else "?"
    return label if client is not None else f"{label} ({address})"


def log_request(request: Request, label: str, status: int, *details: str) -> None:
    parts = [label, f"{request.method} {request.url.path}", str(status), *[d for d in details if d]]
    logger.info("%s", " · ".join(parts))


def build_app(service: Service) -> FastAPI:
    @asynccontextmanager
    async def lifespan(_app: FastAPI):
        service.http = httpx.AsyncClient(timeout=presets.SERVICE_UPSTREAM_TIMEOUT_S)
        try:
            yield
        finally:
            await service.http.aclose()

    app = FastAPI(lifespan=lifespan, docs_url=None, redoc_url=None, openapi_url=None)

    @app.exception_handler(StarletteHTTPException)
    async def not_found(request: Request, exc: StarletteHTTPException):
        if exc.status_code == 404:
            return error_response(404, "not_found", f"Нет такого адреса: {request.url.path}")
        if exc.status_code == 405:
            return error_response(405, "method_not_allowed", f"Метод {request.method} здесь не поддерживается")
        return error_response(exc.status_code, "http_error", str(exc.detail))

    @app.get("/")
    async def chat_page():
        return HTMLResponse(service.settings.page, media_type="text/html; charset=utf-8")

    @app.get("/health")
    async def health():
        state = await service.state()
        return {
            "ok": True,
            "model": state.model,
            "context": state.context,
            "upstream": state.upstream,
            "queue": {"active": service.queue.active, "waiting": service.queue.waiting,
                      "max": presets.SERVICE_QUEUE_MAX},
            "limits": {
                "concurrency": presets.SERVICE_CONCURRENCY,
                "rate_per_min": presets.SERVICE_RATE_PER_MIN,
                "max_tokens": presets.SERVICE_MAX_TOKENS,
                "max_body_bytes": presets.SERVICE_MAX_BODY_BYTES,
            },
        }

    async def proxy_get(request: Request, url: str):
        client, label = service.authorize(request)
        name = who(label, request, client)
        if client is None:
            return refuse_key(request, name, label)
        try:
            response = await service.http.get(url)
        except httpx.TimeoutException:
            log_request(request, name, 504, "upstream_timeout")
            return error_response(504, "upstream_timeout", f"LM Studio не ответил по {url}")
        except httpx.HTTPError:
            log_request(request, name, 502, "upstream_unavailable")
            return error_response(
                502, "upstream_unavailable", f"LM Studio не отвечает по {service.upstream} — lms server start"
            )
        log_request(request, name, response.status_code)
        return Response(response.content, status_code=response.status_code,
                        media_type=response.headers.get("content-type"))

    @app.get("/v1/models")
    async def models(request: Request):
        return await proxy_get(request, f"{service.upstream}/models")

    @app.get("/api/v0/models")
    async def rest_models(request: Request):
        return await proxy_get(request, f"{local_server.origin(service.upstream)}/api/v0/models")

    @app.post("/v1/chat/completions")
    async def chat(request: Request):
        return await chat_completions(service, request)

    return app


def refuse_key(request: Request, name: str, label: str) -> JSONResponse:
    log_request(request, name, 401, "invalid_api_key")
    message = ("Нет доступа: нужен ключ клиента (заголовок Authorization: Bearer …)." if label == "без ключа"
               else "Нет доступа: ключ не подходит.")
    return error_response(401, "invalid_api_key", message)


async def read_body(request: Request) -> bytes | None:
    """Тело не больше `SERVICE_MAX_BODY_BYTES`: по `Content-Length` и по факту
    чтения. Больше — `None`."""
    limit = presets.SERVICE_MAX_BODY_BYTES
    length = request.headers.get("content-length", "")
    if length.isdigit() and int(length) > limit:
        return None
    chunks: list[bytes] = []
    size = 0
    async for chunk in request.stream():
        size += len(chunk)
        if size > limit:
            return None
        chunks.append(chunk)
    return b"".join(chunks)


async def wait_slot(service: Service, request: Request) -> None:
    """Место в модели; клиент отключился, пока ждал, — `ConnectionAbortedError`
    (место отдаётся следующему)."""
    task = asyncio.create_task(service.queue.acquire(presets.SERVICE_QUEUE_TIMEOUT_S))
    while True:
        done, _ = await asyncio.wait({task}, timeout=DISCONNECT_POLL_S)
        if done:
            task.result()
            return
        if await request.is_disconnected():
            task.cancel()
            try:
                await task
            except asyncio.CancelledError:
                pass
            else:
                service.queue.release()   # место успели передать — отдаём дальше
            raise ConnectionAbortedError


async def chat_completions(service: Service, request: Request):
    """`POST /v1/chat/completions` — проверки §3.4 по порядку, затем пересылка §3.8."""
    client, label = service.authorize(request)
    name = who(label, request, client)
    # 1. Ключ.
    if client is None:
        return refuse_key(request, name, label)
    # 2. Размер тела.
    body = await read_body(request)
    if body is None:
        kb = presets.SERVICE_MAX_BODY_BYTES // 1000
        log_request(request, name, 413, "body_too_large")
        return error_response(413, "body_too_large", f"Запрос больше {kb} КБ.")
    # 3. Форма запроса.
    try:
        payload = json.loads(body)
    except ValueError:
        payload = None
        problem = "тело запроса — не JSON"
    else:
        problem = validate_messages(payload)
    if problem:
        log_request(request, name, 400, "invalid_request")
        return error_response(400, "invalid_request", f"Неверный запрос: {problem}.")
    # 4. Потоковой выдачи нет.
    if payload.get("stream") is True:
        log_request(request, name, 400, "stream_not_supported")
        return error_response(400, "stream_not_supported",
                              "Потоковая выдача не поддерживается — отправьте запрос без stream.")
    # 5. Лимит клиента.
    allowed, remaining, retry = service.limiter.take(client, time.monotonic())
    if not allowed:
        log_request(request, name, 429, "rate_limited", f"Retry-After {retry} с")
        return error_response(
            429, "rate_limited",
            f"Слишком много запросов: не больше {client.rate} в минуту для клиента «{client.name}». "
            f"Повторите через {retry} с.",
            headers={"Retry-After": str(retry), "X-RateLimit-Remaining": "0"},
        )
    # 6. Окно модели. Потолок ответа ставится всегда.
    requested = payload.get("max_tokens")
    cap = presets.SERVICE_MAX_TOKENS
    max_tokens = cap if requested is None or requested > cap else requested
    payload["max_tokens"] = max_tokens
    capped = requested is not None and requested > cap
    cap_note = f"потолок ответа уменьшен {requested} → {cap}" if capped else ""
    state = await service.state()
    estimate = estimate_request(payload["messages"])
    if state.context is not None and estimate + max_tokens > state.context:
        log_request(request, name, 400, "context_length_exceeded",
                    f"≈{estimate} + {max_tokens} > {state.context}", cap_note)
        return error_response(
            400, "context_length_exceeded",
            f"Запрос не поместится в окно модели: ≈{estimate} токенов вопроса + {max_tokens} на ответ > "
            f"{state.context} (окно {state.model}). Сократите текст или начните новый разговор.",
        )
    # 7–8. Очередь.
    queued = time.monotonic()
    try:
        await wait_slot(service, request)
    except QueueFull:
        retry = service.retry_after()
        log_request(request, name, 503, "queue_full", f"Retry-After {retry} с")
        return error_response(
            503, "queue_full",
            f"Модель занята, очередь полна (ждут {service.queue.waiting}). Повторите через {retry} с.",
            headers={"Retry-After": str(retry)},
        )
    except TimeoutError:
        retry = service.retry_after()
        log_request(request, name, 503, "queue_timeout",
                    f"очередь {time.monotonic() - queued:.1f} с", f"Retry-After {retry} с")
        return error_response(
            503, "queue_timeout",
            f"Модель не освободилась за {presets.SERVICE_QUEUE_TIMEOUT_S} с. Повторите через {retry} с.",
            headers={"Retry-After": str(retry)},
        )
    except ConnectionAbortedError:
        logger.info("%s · ушёл из очереди через %.1f с — в модель не отправлено", name, time.monotonic() - queued)
        return Response(status_code=499)
    wait = time.monotonic() - queued
    try:
        return await forward(service, request, name, payload, wait, remaining, cap_note, requested if capped else None)
    finally:
        service.queue.release()


async def forward(service: Service, request: Request, name: str, payload: dict, wait: float, remaining: int,
                  cap_note: str, requested: int | None):
    """Пересылка в LM Studio (§3.8): ответ как есть; ошибка — тот же статус в форме
    OpenAI API; нет соединения — 502, таймаут — 504. Повторов нет."""
    url = f"{service.upstream}/chat/completions"
    started = time.monotonic()
    wait_note = f"очередь {wait:.2f} с"
    try:
        response = await service.http.post(url, json=payload)
    except httpx.TimeoutException:
        log_request(request, name, 504, "upstream_timeout", wait_note, f"модель {time.monotonic() - started:.1f} с")
        return error_response(
            504, "upstream_timeout",
            f"LM Studio не ответил за {presets.SERVICE_UPSTREAM_TIMEOUT_S} с.",
        )
    except httpx.HTTPError:
        service.forget_state()
        log_request(request, name, 502, "upstream_unavailable", wait_note)
        return error_response(
            502, "upstream_unavailable", f"LM Studio не отвечает по {service.upstream} — lms server start"
        )
    elapsed = time.monotonic() - started
    model_note = f"модель {elapsed:.2f} с"
    if response.is_success:
        service.recent.append(elapsed)
        details = [wait_note, model_note]
        try:
            data = response.json()
        except ValueError:
            data = {}
        usage = data.get("usage") if isinstance(data, dict) else None
        if isinstance(usage, dict):
            details.append(f"prompt {usage.get('prompt_tokens', '?')} / completion {usage.get('completion_tokens', '?')}")
        if isinstance(data, dict) and data.get("model"):
            details.append(str(data["model"]))
        log_request(request, name, response.status_code, *details, cap_note)
        headers = {
            "X-Service-Client": header_text(name),
            "X-Queue-Wait": f"{wait:.2f}",
            "X-RateLimit-Remaining": str(remaining),
        }
        if requested is not None:
            headers["X-Max-Tokens-Capped"] = f"{requested}->{presets.SERVICE_MAX_TOKENS}"
        return Response(response.content, status_code=response.status_code,
                        media_type=response.headers.get("content-type"), headers=headers)
    text = upstream_message(response)
    if "no models loaded" in text.lower():
        service.forget_state()
    if response.status_code == 400 and any(marker in text.lower() for marker in CONTEXT_OVERFLOW_MARKERS):
        log_request(request, name, 400, "context_length_exceeded", "ответ LM Studio", wait_note, model_note)
        return error_response(
            400, "context_length_exceeded",
            f"Запрос не поместится в окно модели. Сократите текст или начните новый разговор. LM Studio: {text}",
        )
    log_request(request, name, response.status_code, "upstream_error", wait_note, model_note)
    return error_response(response.status_code, "upstream_error", f"LM Studio: {text}")


# --- Старт (§3.1) ------------------------------------------------------------------------

def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="./run.sh service",
        description="Локальная модель как приватный сервис: OpenAI-совместимый прокси перед LM Studio с ключами, "
                    "лимитом, проверкой окна, очередью и страницей чата (день 30).",
    )
    parser.add_argument("--host", default=presets.SERVICE_HOST,
                        help=f"адрес, который слушает сервис (по умолчанию {presets.SERVICE_HOST} — все адреса Mac)")
    parser.add_argument("--port", type=int, default=presets.SERVICE_PORT,
                        help=f"порт (по умолчанию {presets.SERVICE_PORT})")
    parser.add_argument("--upstream", default=presets.SERVICE_UPSTREAM_URL,
                        help=f"адрес LM Studio (по умолчанию {presets.SERVICE_UPSTREAM_URL})")
    parser.add_argument("--new-key", action="store_true", help="напечатать новый ключ клиента и выйти")
    return parser.parse_args(argv)


def port_free(host: str, port: int) -> str:
    """Пустая строка — порт свободен; иначе причина (как `_port_free()` сервера FAQ)."""
    with closing(socket.socket(socket.AF_INET, socket.SOCK_STREAM)) as probe:
        probe.setsockopt(socket.SOL_SOCKET, socket.SO_REUSEADDR, 1)
        try:
            probe.bind((host, port))
        except OSError as exc:
            return str(exc)
    return ""


def lan_addresses() -> list[str]:
    """Адреса Mac в домашней сети: частные IPv4 (192.168/16, 10/8, 172.16/12).
    Кандидаты — адреса имени хоста (`getaddrinfo`, на macOS — все интерфейсы) и
    исходящий адрес UDP-сокета, «подключённого» без отправки пакетов. Адрес VPN
    (100.64/10 — его выбирает UDP-сокет, когда включён VPN) частным не считается
    и в список не попадает. Сети нет — пустой список."""
    candidates: list[str] = []
    try:
        candidates += [info[4][0] for info in socket.getaddrinfo(socket.gethostname(), None, socket.AF_INET)]
    except OSError:
        pass
    try:
        with closing(socket.socket(socket.AF_INET, socket.SOCK_DGRAM)) as probe:
            probe.connect(("192.0.2.1", 9))   # TEST-NET-1: пакеты не уходят
            candidates.append(probe.getsockname()[0])
    except OSError:
        pass
    found: list[str] = []
    for address in candidates:
        try:
            ip = ipaddress.ip_address(address)
        except ValueError:
            continue
        if ip.is_private and not ip.is_loopback and not ip.is_link_local and address not in found:
            found.append(address)
    return found


def points_to_self(upstream: str, port: int, lan: list[str]) -> bool:
    parts = urlsplit(upstream)
    hosts = _SELF_HOSTS | set(lan)
    try:
        upstream_port = parts.port or (443 if parts.scheme == "https" else 80)
    except ValueError:
        return False
    return (parts.hostname or "") in hosts and upstream_port == port


def log_upstream(upstream: str) -> None:
    """Строка о LM Studio при старте — не отказ: его можно поднять и после сервиса."""
    check = local_server.check(upstream, presets.LOCAL_CHECK_TIMEOUT_S)
    status = check.status
    if status == local_server.STATUS_LOADED:
        model = check.loaded[0]
        context = model.context_text
        logger.info("LM Studio %s: загружена %s%s", check.url, model.label.replace("`", ""),
                    f" · {context}" if context else "")
    elif status == local_server.STATUS_NO_ANSWER:
        logger.warning("LM Studio не отвечает по %s — lms server start (сервис работает, запросы получат 502)",
                       check.url)
    elif status == local_server.STATUS_NOT_LOADED:
        logger.warning("LM Studio %s отвечает, но модель не загружена — lms load <id>", check.url)
    elif status == local_server.STATUS_SEVERAL:
        logger.warning("LM Studio %s: загружено несколько моделей (%s) — оставьте одну: lms unload <id>",
                       check.url, ", ".join(model.id for model in check.loaded))
    elif status == local_server.STATUS_UNKNOWN:
        logger.warning("сервер %s отвечает, но не сообщает окно модели — проверка окна пропускается", check.url)
    else:
        logger.warning("LM Studio %s: %s", check.url, check.error)


def new_key() -> None:
    key = secrets.token_urlsafe(24)
    print(key)
    print("Строка для src/.env (несколько клиентов — через запятую, третье поле — лимит в минуту):")
    print(f"{presets.SERVICE_KEYS_ENV}=<имя>:{key}")


def main() -> None:
    args = _args()
    if args.new_key:
        new_key()
        return
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [Сервис] %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)
    # Строка `httpx` на каждый запрос в LM Studio — шум рядом со строками сервиса.
    logging.getLogger("httpx").setLevel(logging.WARNING)

    try:
        clients = parse_keys(os.getenv(presets.SERVICE_KEYS_ENV))
    except KeysError as exc:
        logger.error("ключи клиентов: %s — сервис не запущен; %s", exc, keys_hint())
        sys.exit(2)
    busy = port_free(args.host, args.port)
    if busy:
        logger.error("порт %d занят — сервис уже запущен? (%s)", args.port, busy)
        sys.exit(2)
    lan = lan_addresses()
    if points_to_self(args.upstream, args.port, lan):
        logger.error("--upstream %s указывает на сам сервис — запрос ходил бы по кругу; сервис не запущен",
                     args.upstream)
        sys.exit(2)
    try:
        page = CHAT_PAGE.read_text(encoding="utf-8")
    except OSError as exc:
        logger.error("страница чата %s не читается: %s — сервис не запущен", CHAT_PAGE.name, exc)
        sys.exit(2)

    log_upstream(args.upstream)
    if args.host in ("0.0.0.0", ""):
        addresses = [f"http://127.0.0.1:{args.port}"] + [f"http://{address}:{args.port} (домашняя сеть)" for address in lan]
        if not lan:
            addresses.append("адреса в домашней сети не найдено — Mac не в Wi-Fi?")
    else:
        addresses = [f"http://{args.host}:{args.port}"]
    logger.info(
        "сервис запущен: %s · LM Studio %s · клиенты: %s · в модели %d, очередь %d (ждать до %d с) · потолок ответа "
        "%d · тело до %d КБ · ключи — в src/.env, тексты не логируются, на диск ничего не пишется",
        ", ".join(addresses), args.upstream,
        ", ".join(f"{client.name} ({client.rate}/мин)" for client in clients),
        presets.SERVICE_CONCURRENCY, presets.SERVICE_QUEUE_MAX, presets.SERVICE_QUEUE_TIMEOUT_S,
        presets.SERVICE_MAX_TOKENS, presets.SERVICE_MAX_BODY_BYTES // 1000,
    )

    service = Service(Settings(args.host, args.port, args.upstream, clients, page))
    # SIGTERM — как Ctrl+C (правило сервера FAQ дня 20): uvicorn останавливается
    # штатно, а после остановки поднимает сигнал снова — его ловит except ниже.
    signal.signal(signal.SIGTERM, signal.default_int_handler)
    config = uvicorn.Config(
        build_app(service), host=args.host, port=args.port, access_log=False, log_level="warning",
        timeout_graceful_shutdown=SHUTDOWN_GRACE_S,
    )
    try:
        uvicorn.Server(config).run()
    except KeyboardInterrupt:
        pass
    logger.info("сервис остановлен")


if __name__ == "__main__":
    main()
