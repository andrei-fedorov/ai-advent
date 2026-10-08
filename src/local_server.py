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
# `rag_answer.py`). Состояние сервера (`ServerCheck.status`) и подписи модели
# определены здесь один раз: блок дня и программа решают по ним, а не по тексту
# ошибки и не своими копиями (правка по ревью).
#
# День 29 — память процесса модели (`footprint_mb()`, `server_memory_mb()`):
# сколько занимает процесс, в котором LM Studio держит модель. Мера — вывод
# `footprint` (macOS), а не RSS: на Apple Silicon память видеокарты общая с
# системной, RSS её не видит, а `footprint` включает регион «IOAccelerator
# (graphics)» — у 4-битной Qwen3.5-2B в покое это 1,7 ГБ из 2,4 ГБ. Путь процесса
# LM Studio модуль не знает: шаблон `pgrep -f` приходит параметром из
# `presets.py`. Только чтение — `pgrep` и `footprint`, без сигналов и записи в
# чужой процесс; без исключений, как `check()`: нет программы, не macOS, нет
# процесса, вывод не разобран — `None`. Только стандартная библиотека и `httpx`
# — лист графа остаётся листом.

import re
import subprocess
import time
from dataclasses import dataclass
from urllib.parse import urlsplit, urlunsplit

import httpx

# Тип записи REST LM Studio, который не считается языковой моделью.
_EMBEDDINGS = "embeddings"
_LOADED = "loaded"

# Потолок одного запуска `footprint` и `pgrep`, секунды: обычно — десятые доли.
FOOTPRINT_TIMEOUT_S = 5

# «node [58260]: 64-bit    Footprint: 2364 MB (16384 bytes per page)».
_FOOTPRINT_LINE = re.compile(r"Footprint:\s*([\d.,]+)\s*([KMG]B)\b")
_UNIT_MB = {"KB": 1 / 1024, "MB": 1.0, "GB": 1024.0}

# Состояние сервера (`ServerCheck.status`): по нему блок дня выбирает строку
# статуса, а программа — отказ или предупреждение.
STATUS_NO_ANSWER = "не отвечает"         # отказ соединения или таймаут
STATUS_FAILED = "сбой"                   # ответил, но не как OpenAI-совместимый сервер
STATUS_UNKNOWN = "состояние неизвестно"  # нет REST LM Studio: какая загружена, неизвестно
STATUS_NOT_LOADED = "модель не загружена"
STATUS_SEVERAL = "загружено несколько"
STATUS_LOADED = "загружена"


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

    @property
    def label(self) -> str:
        """«`qwen3.5-2b-mlx` (MLX, 4bit, qwen3_5)» — формат, квантование, архитектура."""
        parts = [part for part in (self.format.upper(), self.quantization, self.arch) if part]
        return f"`{self.id}`" + (f" ({', '.join(parts)})" if parts else "")

    @property
    def context_text(self) -> str:
        """«контекст 132 096 из 262 144»; сервер не назвал — пусто."""
        if self.loaded_context and self.max_context:
            return f"контекст {_int_text(self.loaded_context)} из {_int_text(self.max_context)}"
        if self.max_context:
            return f"контекст до {_int_text(self.max_context)}"
        return ""


@dataclass(frozen=True)
class ServerCheck:
    """Результат проверки сервера. `extended` — ответил ли REST LM Studio."""

    ok: bool
    url: str
    elapsed: float
    error: str = ""
    models: tuple[ModelInfo, ...] = ()
    extended: bool = False
    # Сбой — отказ соединения или таймаут: сервер не запущен, помочь может
    # `lms server start`. Остальные сбои (ответ не тот, обрыв посреди ответа) —
    # на порту что-то отвечает, и подсказка не поможет.
    no_answer: bool = False

    @property
    def llms(self) -> tuple[ModelInfo, ...]:
        """Языковые модели сервера — все, кроме эмбеддингов."""
        return tuple(model for model in self.models if model.type != _EMBEDDINGS)

    @property
    def loaded(self) -> tuple[ModelInfo, ...]:
        """Загруженные языковые модели. Без REST LM Studio состояние неизвестно —
        пусто."""
        if not self.extended:
            return ()
        return tuple(model for model in self.llms if model.state == _LOADED)

    @property
    def status(self) -> str:
        """Состояние сервера — одна из констант `STATUS_*`."""
        if not self.ok:
            return STATUS_NO_ANSWER if self.no_answer else STATUS_FAILED
        if not self.extended:
            return STATUS_UNKNOWN
        count = len(self.loaded)
        if count == 0:
            return STATUS_NOT_LOADED
        return STATUS_SEVERAL if count > 1 else STATUS_LOADED


def origin(base_url: str) -> str:
    """Адрес сервера без пути (`http://127.0.0.1:1234/v1` → `http://127.0.0.1:1234`)."""
    parts = urlsplit(base_url)
    return urlunsplit((parts.scheme, parts.netloc, "", "", ""))


def _int_text(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def _int(value: object) -> int | None:
    return value if isinstance(value, int) and not isinstance(value, bool) else None


def _text(value: object) -> str:
    return value if isinstance(value, str) else ""


def _has_data(payload: object) -> bool:
    """Ответ вида `{"data": [...]}` — список моделей (может быть и пустым)."""
    return isinstance(payload, dict) and isinstance(payload.get("data"), list)


def _rows(payload: object) -> list[dict]:
    """Записи моделей из ответа `{"data": [...]}`; не то, что ждали, — `[]`."""
    if not _has_data(payload):
        return []
    return [row for row in payload["data"] if isinstance(row, dict) and isinstance(row.get("id"), str)]


def _get_json(client: httpx.Client, url: str) -> object:
    response = client.get(url)
    response.raise_for_status()
    return response.json()


def check(base_url: str, timeout_s: float) -> ServerCheck:
    """Спросить сервер, какие модели у него есть и какая загружена.

    Без исключений: сбой — `ok=False` с причиной. Нет ответа на `/models` —
    «сервер не отвечает по <адрес>» и `no_answer` для отказа соединения и
    таймаута, текст ошибки для остального; ответ не JSON или JSON без списка
    `data` — тоже сбой (на порту не OpenAI-совместимый сервер). REST LM Studio,
    которого нет (404, не JSON, иной сбой), — не сбой: `extended=False`, модели
    из первого ответа, `state` = `None`.
    """
    base = base_url.rstrip("/")
    started = time.perf_counter()
    try:
        with httpx.Client(timeout=timeout_s) as client:
            try:
                listing = _get_json(client, f"{base}/models")
            except (httpx.ConnectError, httpx.TimeoutException):
                # Отказ соединения и таймаут — «не отвечает»: по смыслу для
                # человека это одно и то же («сервер не запущен или не успел»).
                # Обрыв посреди ответа (ReadError, RemoteProtocolError) сюда не
                # относится: на порту что-то отвечает — это текст ошибки ниже.
                return ServerCheck(
                    False, base, time.perf_counter() - started,
                    f"сервер не отвечает по {base}", no_answer=True,
                )
            except (httpx.HTTPError, ValueError) as exc:
                # HTTPStatusError, иной транспортный сбой или ответ не JSON.
                return ServerCheck(
                    False, base, time.perf_counter() - started,
                    f"{type(exc).__name__}: {exc}",
                )
            if not _has_data(listing):
                return ServerCheck(
                    False, base, time.perf_counter() - started,
                    f"ответ {base}/models — не список моделей OpenAI API (нет списка data)",
                )
            basic = _rows(listing)
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
            if _has_data(payload):
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


def _run(args: list[str]) -> str | None:
    """Вывод программы; не запустилась, таймаут — `None`. Ненулевой код выхода
    не сбой: у `pgrep` это «никого не нашёл» — разбирает вызывающий."""
    try:
        done = subprocess.run(args, capture_output=True, text=True, timeout=FOOTPRINT_TIMEOUT_S, check=False)
    except (OSError, subprocess.SubprocessError, ValueError):
        return None
    return done.stdout


def footprint_mb(pid: int) -> float | None:
    """Память процесса по `footprint <pid>` (macOS), МБ — с памятью видеокарты.
    Нет программы, не macOS, нет процесса, не разобрано — `None`."""
    try:
        output = _run(["footprint", str(int(pid))])
    except Exception:  # noqa: BLE001 — контракт модуля: без исключений
        return None
    match = _FOOTPRINT_LINE.search(output or "")
    if match is None:
        return None
    try:
        value = float(match.group(1).replace(",", "."))
    except ValueError:
        return None
    return value * _UNIT_MB[match.group(2)]


def server_memory_mb(pattern: str) -> float | None:
    """Наибольший `footprint_mb()` среди процессов `pgrep -f <pattern>`: у LM
    Studio процессов с одним путём несколько (служебный и с моделью), модель —
    в самом большом. Никого не нашёл или ни один не измерен — `None`."""
    try:
        output = _run(["pgrep", "-f", pattern]) if pattern else None
        pids = [int(line) for line in (output or "").split() if line.isdigit()]
        values = [value for value in (footprint_mb(pid) for pid in pids) if value is not None]
    except Exception:  # noqa: BLE001 — контракт модуля: без исключений
        return None
    return max(values) if values else None
