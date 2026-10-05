# TooManyRules — проверка локального сервера LLM (день 26, неделя 6).
#
# Отвечает на вопрос «работает ли сервер и какая модель на нём сейчас
# загружена». **Лист графа**: из проекта не импортирует ничего, LLM не
# вызывает — модель вызывает только `agent.py`. Его берут `app.py` (кнопка блока
# дня) и программа `local_llm.py`, поэтому приложение не импортирует программу.
#
# Id модели модуль не принимает и не хранит: какая модель загружена, решает
# человек в LM Studio, а модуль только спрашивает сервер. Два вопроса:
# - `GET <base_url>/models` — OpenAI-совместимый список моделей; отдаёт все
#   скачанные модели без признака «загружена»;
# - `GET <origin>/api/v0/models` — REST LM Studio: те же модели с полями `type`,
#   `state` (`loaded` / `not-loaded`), `arch`, `compatibility_type`,
#   `quantization`, `max_context_length` и у загруженной — `loaded_context_length`.
#   Это источник для «какая модель сейчас активна». Нет такого адреса (другой
#   OpenAI-совместимый сервер) — не сбой: состояние моделей неизвестно.
#
# Исключений наружу не бросает, логов у модуля нет — логируют вызывающие (как
# `rag_answer.py`).

import time
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import httpx

# Тип записи REST LM Studio, который не считается языковой моделью.
_EMBEDDINGS = "embeddings"
_LOADED = "loaded"


@dataclass(frozen=True)
class ModelInfo:
    """Модель сервера. `state` — `None`, если сервер не отдаёт состояние."""

    id: str
    type: str = ""
    state: str | None = None
    arch: str = ""
    format: str = ""              # compatibility_type: mlx, gguf, …
    quantization: str = ""
    max_context: int | None = None
    loaded_context: int | None = None


@dataclass(frozen=True)
class ServerCheck:
    """Результат проверки сервера. `extended` — ответил ли REST LM Studio."""

    ok: bool
    url: str
    elapsed: float
    error: str = ""
    models: tuple[ModelInfo, ...] = ()
    extended: bool = False

    @property
    def loaded(self) -> tuple[ModelInfo, ...]:
        """Загруженные языковые модели. Без REST LM Studio состояние неизвестно —
        пусто."""
        if not self.extended:
            return ()
        return tuple(
            model for model in self.models
            if model.state == _LOADED and model.type != _EMBEDDINGS
        )


def origin(base_url: str) -> str:
    """Адрес сервера без пути (`http://127.0.0.1:1234/v1` → `http://127.0.0.1:1234`)."""
    parts = urlsplit(base_url)
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def _int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _rows(payload: object) -> list[dict]:
    """Записи моделей из ответа `{"data": [...]}`; не то, что ждали, — `[]`."""
    data = payload.get("data") if isinstance(payload, dict) else None
    if not isinstance(data, list):
        return []
    return [row for row in data if isinstance(row, dict) and isinstance(row.get("id"), str)]


def _get_json(client: httpx.Client, url: str) -> object:
    response = client.get(url)
    response.raise_for_status()
    return response.json()


def check(base_url: str, timeout_s: float) -> ServerCheck:
    """Спросить сервер, какие модели у него есть и какая загружена.

    Без исключений: сбой — `ok=False` с причиной. Нет ответа на `/models` —
    «сервер не отвечает по <адрес>» для ошибки соединения, текст ошибки для
    остального. REST LM Studio, которого нет (404, не JSON, иной сбой), — не
    сбой: `extended=False`, модели из первого ответа, `state` = `None`.
    """
    base = base_url.rstrip("/")
    started = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout_s) as client:
            try:
                basic = _rows(_get_json(client, f"{base}/models"))
            except httpx.TransportError as exc:
                # Таймаут и обрыв — тоже «не отвечает»: по смыслу для человека это
                # одно и то же («сервер не запущен или не успел»).
                reason = (
                    f"сервер не отвечает по {base}"
                    if isinstance(exc, (httpx.ConnectError, httpx.TimeoutException))
                    else f"{type(exc).__name__}: {exc}"
                )
                return ServerCheck(False, base, time.perf_counter() - started, reason)
            except (httpx.HTTPError, ValueError) as exc:
                # HTTPStatusError или ответ не JSON.
                return ServerCheck(
                    False, base, time.perf_counter() - started,
                    f"{type(exc).__name__}: {exc}",
                )
            models = tuple(
                ModelInfo(
                    id=row["id"], type=_text(row.get("type")), state=None,
                    arch=_text(row.get("arch")),
                )
                for row in basic
            )
            extended = False
            try:
                payload = _get_json(client, f"{origin(base)}/api/v0/models")
            except (httpx.HTTPError, ValueError):
                payload = None
            # Ответ REST — словарь со списком `data` (он может быть и пустым:
            # моделей нет, но сервер LM Studio).
            if isinstance(payload, dict) and isinstance(payload.get("data"), list):
                rows = _rows(payload)
                extended = True
                models = tuple(
                    ModelInfo(
                        id=row["id"],
                        type=_text(row.get("type")),
                        state=_text(row.get("state")) or None,
                        arch=_text(row.get("arch")),
                        format=_text(row.get("compatibility_type")),
                        quantization=_text(row.get("quantization")),
                        max_context=_int(row.get("max_context_length")),
                        loaded_context=_int(row.get("loaded_context_length")),
                    )
                    for row in rows
                )
    except Exception as exc:  # noqa: BLE001 — контракт модуля: без исключений
        return ServerCheck(
            False, base, time.perf_counter() - started, f"{type(exc).__name__}: {exc}"
        )
    return ServerCheck(True, base, time.perf_counter() - started, "", models, extended)
