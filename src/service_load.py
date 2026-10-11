# TooManyRules — проверка стабильности сервиса (день 30, неделя 6).
#
# **Отдельная программа**, по образцу `local_llm.py`: её запускает человек
# (`./run.sh service-load`), работает до конца. Ходит в сервис дня 30
# (`llm_service.py`) по HTTP (`httpx`, async) как обычный клиент — к LM Studio и
# агенту не обращается, LLM от себя не вызывает. Четыре сценария (§5
# спецификации): «ключ» (без ключа и с чужим — 401), «контекст» (вопрос больше
# окна и вопрос, который помещается, а с потолком ответа — нет, — 400),
# «параллельно» (N запросов сразу — очередь, 503 «очередь полна», 429 по лимиту)
# и «лимит» (один клиент подряд до 429).
#
# Ожидаемое — рядом с фактом: у каждого запроса список ожидаемых статусов, код
# смотрит только статус и код отказа. Ответы модели не оценивает.
#
# Ключи — из `src/.env` (тот же `presets.SERVICE_KEYS_ENV`, что у сервиса), разбор
# — `llm_service.parse_keys()`: ребро между программами, как `rag_dialog.py` →
# `rag_eval.py`. Ключи не попадают ни в лог, ни в отчёт. Текст вопроса нужного
# размера — `tokens.filler_text()`: та же оценка, по которой отказывает сервис.
# Память процесса модели LM Studio — `local_server.server_memory_mb()`, только если
# сервис на этой машине.
#
# Логи — stdout, `[Нагрузка]`. Коды выхода: 0 — все ответы из ожидаемых и сервис
# жив после; 1 — неожиданный статус, сетевой сбой или сервис не ответил после;
# 2 — отказ до сценариев; 130 — Ctrl+C, отчёт по пройденному.

import argparse
import asyncio
import json
import logging
import os
import secrets
import socket
import sys
import time
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from urllib.parse import urlsplit

import httpx

import local_server
import presets
import tokens
from llm_service import Client, KeysError, parse_keys

logger = logging.getLogger("service_load")

SCENARIOS = ("ключ", "контекст", "параллельно", "лимит")
CHAT_PATH = "/v1/chat/completions"
# Потолки ответа сценариев (§5): короткие ответы — очередь видна, а прогон быстрый.
PARALLEL_MAX_TOKENS = 300
LIMIT_MAX_TOKENS = 5
# Запас под окно в сценарии «контекст (б)»: вопрос ≈ окно − 1000 по оценке сервиса.
CONTEXT_MARGIN = 1000
# Потолок ожидания одного запроса: очередь + модель + запас.
REQUEST_TIMEOUT_S = presets.SERVICE_QUEUE_TIMEOUT_S + presets.SERVICE_UPSTREAM_TIMEOUT_S + 10
HEALTH_TIMEOUT_S = 10


@dataclass
class Result:
    """Один запрос сценария: что ждали и что пришло."""

    scenario: str
    label: str
    client: str
    expected: tuple[str, ...]          # «200», «401», «400 context_length_exceeded», …
    status: int | None = None
    code: str = ""
    wait: float | None = None          # X-Queue-Wait
    elapsed: float = 0.0
    prompt: int | None = None
    completion: int | None = None
    retry_after: str = ""
    error: str = ""                    # сетевой сбой или текст отказа
    note: str = ""

    @property
    def outcome(self) -> str:
        if self.status is None:
            return "сбой"
        return f"{self.status} {self.code}".strip()

    @property
    def as_expected(self) -> bool:
        if self.status is None:
            return False
        return any(self.outcome == item or str(self.status) == item for item in self.expected)


@dataclass
class Run:
    url: str
    clients: list[Client]
    started: datetime
    health_before: dict = field(default_factory=dict)
    health_after: dict | None = None
    memory_before: float | None = None
    memory_after: float | None = None
    results: list[Result] = field(default_factory=list)
    skipped: list[str] = field(default_factory=list)
    interrupted: str = ""


# --- Запросы -------------------------------------------------------------------------

def chat_body(text: str, max_tokens: int) -> dict:
    return {
        "model": presets.LOCAL_MODEL_ALIAS,
        "messages": [{"role": "user", "content": text}],
        "max_tokens": max_tokens,
        "temperature": 0,
        "reasoning_effort": "none",
    }


async def send(http: httpx.AsyncClient, result: Result, body: dict, key: str | None) -> Result:
    """Запрос в сервис; сетевой сбой — `status=None` с причиной."""
    headers = {"Authorization": f"Bearer {key}"} if key else {}
    started = time.monotonic()
    try:
        response = await http.post(CHAT_PATH, json=body, headers=headers, timeout=REQUEST_TIMEOUT_S)
    except httpx.HTTPError as exc:
        result.elapsed = time.monotonic() - started
        result.error = f"{type(exc).__name__}: {exc}" if str(exc) else type(exc).__name__
        return result
    result.elapsed = time.monotonic() - started
    result.status = response.status_code
    result.retry_after = response.headers.get("retry-after", "")
    wait = response.headers.get("x-queue-wait")
    result.wait = float(wait) if wait else None
    try:
        data = response.json()
    except ValueError:
        data = {}
    if response.is_success and isinstance(data, dict):
        usage = data.get("usage") or {}
        result.prompt, result.completion = usage.get("prompt_tokens"), usage.get("completion_tokens")
    elif isinstance(data, dict) and isinstance(data.get("error"), dict):
        result.code = str(data["error"].get("code") or "")
        result.error = str(data["error"].get("message") or "")
    return result


def log_result(result: Result) -> None:
    mark = "✓" if result.as_expected else f"⚠️ ждали {' или '.join(result.expected)}"
    parts = [f"{result.scenario} · {result.label}", result.client, result.outcome, mark]
    if result.wait is not None:
        parts.append(f"очередь {result.wait:.2f} с")
    parts.append(f"{result.elapsed:.2f} с")
    if result.prompt is not None:
        parts.append(f"prompt {result.prompt} / completion {result.completion}")
    if result.retry_after:
        parts.append(f"Retry-After {result.retry_after} с")
    if result.status is None or (result.error and not result.as_expected):
        parts.append(result.error)
    logger.info("%s", " · ".join(parts))


async def health(http: httpx.AsyncClient) -> dict | None:
    try:
        response = await http.get("/health", timeout=HEALTH_TIMEOUT_S)
        data = response.json()
    except (httpx.HTTPError, ValueError):
        return None
    return data if response.is_success and isinstance(data, dict) else None


# --- Сценарии (§5) ---------------------------------------------------------------------

async def scenario_key(http: httpx.AsyncClient, run: Run) -> None:
    body = chat_body(presets.SERVICE_LOAD_QUESTIONS[0], LIMIT_MAX_TOKENS)
    for label, key in (("без заголовка", None), ("выдуманный ключ", "wrong-" + secrets.token_urlsafe(24))):
        result = await send(http, Result("ключ", label, "—", ("401 invalid_api_key",)), body, key)
        run.results.append(result)
        log_result(result)


async def scenario_context(http: httpx.AsyncClient, run: Run, client: Client) -> None:
    window = run.health_before.get("context")
    if not isinstance(window, int):
        run.skipped.append("контекст — сервис не сообщает окно модели (проверка окна у него пропускается)")
        logger.warning("контекст пропущен: окно модели неизвестно")
        return
    limit = run.health_before.get("limits", {}).get("max_body_bytes", presets.SERVICE_MAX_BODY_BYTES)
    # Оценка сервиса: служебные токены запроса и сообщения + текст.
    overhead = tokens.TOKENS_PER_REQUEST + tokens.TOKENS_PER_MESSAGE
    cases = (
        ("(а) вопрос ≈ 3 × окно", 3 * window - overhead, None),
        ("(б) вопрос ≈ окно − 1000, ответ 2000", window - CONTEXT_MARGIN - overhead, presets.SERVICE_MAX_TOKENS),
    )
    for label, target, max_tokens in cases:
        text = tokens.filler_text(target)
        body = chat_body(text, max_tokens or 0)
        if max_tokens is None:
            body.pop("max_tokens")   # потолок ставит сервис
        size = len(json.dumps(body, ensure_ascii=False).encode())
        # Тело больше потолка сервиса — отказ раньше окна (413): так бывает при
        # большом окне модели. Это тоже отказ, но другой — ожидание по факту.
        expected = ("400 context_length_exceeded",) if size <= limit else ("413 body_too_large",)
        result = Result("контекст", label, client.name, expected,
                        note=f"≈{tokens.estimate_tokens(text) + overhead} ток. по оценке, тело {size // 1000} КБ")
        await send(http, result, body, client.key)
        run.results.append(result)
        log_result(result)


async def scenario_parallel(http: httpx.AsyncClient, run: Run, clients: list[Client], count: int) -> None:
    questions = presets.SERVICE_LOAD_QUESTIONS
    expected = ("200", "503 queue_full", "429 rate_limited")
    jobs = []
    for number in range(count):
        client = clients[number % len(clients)]
        result = Result("параллельно", f"#{number + 1}", client.name, expected)
        jobs.append(send(http, result, chat_body(questions[number % len(questions)], PARALLEL_MAX_TOKENS), client.key))
    results = await asyncio.gather(*jobs)
    for result in sorted(results, key=lambda item: (item.status != 200, item.wait or 0, item.label)):
        run.results.append(result)
        log_result(result)


async def scenario_limit(http: httpx.AsyncClient, run: Run, client: Client) -> None:
    for number in range(client.rate + 1):
        last = number == client.rate
        # Последний разрешённый запрос + 1 обязан получить 429.
        expected = ("429 rate_limited",) if last else ("200", "429 rate_limited")
        result = Result("лимит", f"#{number + 1} из ≤ {client.rate + 1}", client.name, expected)
        await send(http, result, chat_body(presets.SERVICE_LOAD_QUESTIONS[number % 2], LIMIT_MAX_TOKENS), client.key)
        if result.status == 429 and not result.retry_after:
            result.note = "нет Retry-After"
            result.expected = ("429 rate_limited с Retry-After",)
        run.results.append(result)
        log_result(result)
        if result.status == 429 or result.status is None:
            return


# --- Отчёт -------------------------------------------------------------------------------

def gb(value: float | None) -> str:
    return "н/д" if value is None else f"{value / 1024:.2f} ГБ".replace(".", ",")


def cell(value: object) -> str:
    return str(value).replace("|", "\\|").replace("\n", " ") if value not in (None, "") else "—"


def health_text(data: dict | None) -> str:
    if data is None:
        return "не ответил"
    queue = data.get("queue", {})
    return (
        f"модель `{data.get('model')}`, окно {data.get('context')}, LM Studio: {data.get('upstream')}, "
        f"очередь: в модели {queue.get('active')}, ждут {queue.get('waiting')} из {queue.get('max')}"
    )


def build_report(run: Run, scenarios: list[str]) -> str:
    limits = run.health_before.get("limits", {})
    lines = [
        f"# Нагрузка на сервис локальной модели (день 30) — {run.started:%d.%m.%Y %H:%M:%S}",
        "",
    ]
    if run.interrupted:
        lines += [f"> ⚠️ Прогон прерван {run.interrupted} — отчёт по пройденному.", ""]
    lines += [
        "## 1. Условия",
        "",
        f"- Сервис: `{run.url}`",
        f"- `/health` до: {health_text(run.health_before)}",
        f"- `/health` после: {health_text(run.health_after)}",
        f"- Лимиты сервиса: в модели {limits.get('concurrency')}, лимит по умолчанию {limits.get('rate_per_min')}/мин, "
        f"потолок ответа {limits.get('max_tokens')}, тело до {limits.get('max_body_bytes')} байт",
        f"- Клиенты (ключи не печатаются): {', '.join(f'{c.name} ({c.rate}/мин)' for c in run.clients)}",
        f"- Сценарии: {', '.join(scenarios)}",
        f"- Память процесса модели LM Studio (`footprint`): до {gb(run.memory_before)}, после {gb(run.memory_after)}",
        "",
        "## 2. Сводка",
        "",
    ]
    counts: dict[str, int] = {}
    for result in run.results:
        key = str(result.status) if result.status is not None else "сбой"
        counts[key] = counts.get(key, 0) + 1
    unexpected = [r for r in run.results if not r.as_expected]
    waits = [r.wait for r in run.results if r.wait is not None]
    answered = [r.elapsed for r in run.results if r.status == 200]
    lines += [
        f"- Запросов: {len(run.results)} · " + " · ".join(f"{key}: {value}" for key, value in sorted(counts.items())),
        f"- Неожиданных: {len(unexpected)}" + (
            " — " + "; ".join(f"{r.scenario} {r.label}: {r.outcome}" for r in unexpected) if unexpected else ""),
        f"- Наибольшее ожидание в очереди: {max(waits):.2f} с" if waits else "- Ожидания в очереди не было",
        f"- Время ответов 200: от {min(answered):.2f} до {max(answered):.2f} с" if answered else "- Ответов 200 нет",
        f"- Сервис после прогона: {'жив, очередь пуста' if alive_after(run) else 'не ответил или очередь не пуста'}",
        f"- Память процесса модели LM Studio: до {gb(run.memory_before)}, после {gb(run.memory_after)}",
    ]
    for item in run.skipped:
        lines.append(f"- Пропущено: {item}")
    for scenario in scenarios:
        rows = [r for r in run.results if r.scenario == scenario]
        if not rows:
            continue
        lines += [
            "",
            f"## Сценарий «{scenario}»",
            "",
            "| запрос | клиент | статус | код | ожидалось | совпало | очередь, с | время, с | prompt / completion "
            "| Retry-After | примечание |",
            "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |",
        ]
        for r in rows:
            tokens_text = f"{r.prompt} / {r.completion}" if r.prompt is not None else "—"
            note = r.note or ("" if r.as_expected and r.status == 200 else r.error)
            lines.append(
                f"| {cell(r.label)} | {cell(r.client)} | {cell(r.status if r.status is not None else 'сбой')} | "
                f"{cell(r.code)} | {cell(' или '.join(r.expected))} | {'✓' if r.as_expected else '⚠️'} | "
                f"{cell(f'{r.wait:.2f}' if r.wait is not None else '')} | {r.elapsed:.2f} | {tokens_text} | "
                f"{cell(r.retry_after)} | {cell(note)} |"
            )
        if scenario == "параллельно":
            cap = limits.get("concurrency", presets.SERVICE_CONCURRENCY) + run.health_before.get("queue", {}).get(
                "max", presets.SERVICE_QUEUE_MAX)
            ok = sum(1 for r in rows if r.status == 200)
            lines += ["", f"Ответов 200: {ok}; ожидалось не больше {cap} (в модели + очередь), остальные — 503 "
                          f"«очередь полна» или 429 по лимиту клиента."]
    return "\n".join(lines) + "\n"


def alive_after(run: Run) -> bool:
    queue = (run.health_after or {}).get("queue", {})
    return run.health_after is not None and queue.get("active") == 0 and queue.get("waiting") == 0


# --- Запуск ------------------------------------------------------------------------------

def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="./run.sh service-load",
        description="Проверка стабильности сервиса дня 30: ключи, окно модели, параллельные запросы, лимит.",
    )
    parser.add_argument("--url", default=f"http://127.0.0.1:{presets.SERVICE_PORT}",
                        help=f"адрес сервиса (по умолчанию http://127.0.0.1:{presets.SERVICE_PORT})")
    parser.add_argument("--scenarios", default=",".join(SCENARIOS),
                        help=f"сценарии через запятую (по умолчанию все: {','.join(SCENARIOS)})")
    parser.add_argument("--requests", type=int, default=presets.SERVICE_LOAD_REQUESTS,
                        help=f"запросов в сценарии «параллельно» (по умолчанию {presets.SERVICE_LOAD_REQUESTS})")
    parser.add_argument("--clients", default="", help="имена клиентов через запятую (по умолчанию все)")
    parser.add_argument("--out", type=Path, default=None, help="файл отчёта .md")
    return parser.parse_args(argv)


def local_host(url: str) -> bool:
    """Сервис на этой машине — память процесса модели можно померить."""
    host = urlsplit(url).hostname or ""
    if host in ("127.0.0.1", "localhost", "::1"):
        return True
    try:
        return host in socket.gethostbyname_ex(socket.gethostname())[2]
    except OSError:
        return False


async def run_all(run: Run, scenarios: list[str], requests: int) -> None:
    async with httpx.AsyncClient(base_url=run.url, limits=httpx.Limits(max_connections=None)) as http:
        by_rate_desc = sorted(run.clients, key=lambda c: -c.rate)
        try:
            for scenario in scenarios:
                logger.info("— сценарий «%s»", scenario)
                if scenario == "ключ":
                    await scenario_key(http, run)
                elif scenario == "контекст":
                    await scenario_context(http, run, by_rate_desc[0])
                elif scenario == "параллельно":
                    await scenario_parallel(http, run, run.clients, requests)
                elif scenario == "лимит":
                    await scenario_limit(http, run, by_rate_desc[-1])
        finally:
            run.health_after = await health(http)


def main() -> None:
    args = _args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [Нагрузка] %(message)s", datefmt="%H:%M:%S",
                        stream=sys.stdout)
    logging.getLogger("httpx").setLevel(logging.WARNING)   # строка на каждый запрос — шум
    url = args.url.rstrip("/")
    scenarios = [item.strip() for item in args.scenarios.split(",") if item.strip()]
    unknown = [item for item in scenarios if item not in SCENARIOS]
    if unknown or not scenarios:
        logger.error("нет сценариев: %s; есть: %s", ", ".join(unknown) or "(пусто)", ", ".join(SCENARIOS))
        sys.exit(2)
    if args.requests < 1:
        logger.error("--requests — не меньше 1")
        sys.exit(2)
    if args.out is not None and (args.out.suffix.lower() != ".md" or args.out.is_dir()):
        logger.error("--out — файл .md")
        sys.exit(2)

    before = asyncio.run(_health_once(url))
    if before is None:
        logger.error("сервис не отвечает по %s — ./run.sh service", url)
        sys.exit(2)
    try:
        clients = parse_keys(os.getenv(presets.SERVICE_KEYS_ENV))
    except KeysError as exc:
        logger.error("ключи клиентов: %s — нагрузка не запущена", exc)
        sys.exit(2)
    if args.clients:
        wanted = [item.strip() for item in args.clients.split(",") if item.strip()]
        missing = [name for name in wanted if name not in {c.name for c in clients}]
        if missing or not wanted:
            logger.error("нет клиентов %s в %s; есть: %s", ", ".join(missing) or "(пусто)", presets.SERVICE_KEYS_ENV,
                         ", ".join(c.name for c in clients))
            sys.exit(2)
        clients = [c for c in clients if c.name in wanted]
    if not before.get("model"):
        logger.error("у сервиса нет модели: %s — нагрузка не запущена", before.get("upstream"))
        sys.exit(2)

    measure = local_host(url)
    run = Run(url, clients, datetime.now(), health_before=before)
    run.memory_before = local_server.server_memory_mb(presets.LOCAL_SERVER_PROCESS) if measure else None
    logger.info("сервис %s · %s · клиенты: %s · сценарии: %s", url, health_text(before),
                ", ".join(f"{c.name} ({c.rate}/мин)" for c in clients), ", ".join(scenarios))
    code = 0
    try:
        asyncio.run(run_all(run, scenarios, args.requests))
    except KeyboardInterrupt:
        run.interrupted = f"после {len(run.results)} запросов"
        code = 130
        logger.error("прервано — пишу отчёт по пройденному")
    except Exception as exc:  # noqa: BLE001 — отчёт по пройденному важнее трейсбека (правило local_llm)
        run.interrupted = f"после {len(run.results)} запросов: сбой — {exc}"
        code = 1
        logger.exception("сбой — пишу отчёт по пройденному")
    run.memory_after = local_server.server_memory_mb(presets.LOCAL_SERVER_PROCESS) if measure else None
    if code == 0 and (any(not r.as_expected for r in run.results) or not alive_after(run)):
        code = 1

    out = args.out.expanduser() if args.out is not None else presets.SERVICE_LOAD_DIR / f"{run.started:%Y-%m-%d-%H%M%S}.md"
    report = build_report(run, scenarios)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(report, encoding="utf-8")
    logger.info("отчёт: %s", out)
    print()
    print("Сводка" + report.partition("## 2. Сводка")[2].partition("\n## ")[0].rstrip())
    sys.exit(code)


async def _health_once(url: str) -> dict | None:
    async with httpx.AsyncClient(base_url=url) as http:
        return await health(http)


if __name__ == "__main__":
    main()
