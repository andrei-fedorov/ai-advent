# TooManyRules — сравнение локальной и облачной модели в RAG (день 28, неделя 6).
#
# **Отдельная программа**, как `rag_eval.py`, `rag_dialog.py` и `local_llm.py`: её
# запускает человек (`./run.sh rag-compare`), и работает она до конца, а не
# живёт. Приложение её не импортирует. Качество (форма ответа), скорость и
# стабильность в одном отчёте: те же 16 вопросов, что у `rag-eval` (К1-К10,
# У1-У3, Н1, Н2, Ф1), в режимах `фильтр` и `полный`, каждый вопрос — по N
# повторов, на каждый повтор — свежий голый агент (`rag_eval.run_mode()`).
# К модели ходит только через `Agent.ask()` (LLM API вызывается только в
# `agent.py`), выдержки ищет тот же `rag_search.RulesIndex`, один на прогон.
#
# **Колонка** — пресет, фактическая модель (из ответов API,
# `AgentReply.answered_model`) и перекрытая температура. Программа не
# переключает модели LM Studio и не управляет самим LM Studio: локальная колонка
# — та модель, что загружена сейчас; две локальные модели — два прогона и сборка
# `--merge`. Id моделей в коде нет.
#
# **Порядок проходов — «повтор за повтором»** — ради кэша префикса LM Studio:
# при t = 0 ответ локальной модели зависит от того, был ли такой запрос уже в
# кэше (спецификация дня 28, §2.5), а кэш живёт до выгрузки модели. Если прогон
# начат сразу после `lms load`, у каждого вопроса повтор 1 — с холодным кэшем,
# повторы 2..N — с тёплым. Состояния кэша программа не знает и не меняет: начать
# сразу после загрузки — шаг человека.
#
# **Отчёт строится только из записей** (плоский словарь на ход) — и при прогоне,
# и при `--merge`: путь один, общий отчёт и отчёт прогона не расходятся. Рядом с
# `.md` пишется `.json` с теми же записями. **Оценок в отчёте нет**: код считает
# форму (проверка ответа дня 24) и числа, смысл ответа — пустая колонка автора,
# «лучшей модели» и подсветки колонок нет.
#
# Логи — stdout: строки агентов (как в приложении) и свои с префиксом
# `[RAG-сравнение]`. Коды выхода: 0 — все ходы удались, 1 — сбой посреди прогона
# (ход `ok=False` или исключение), 2 — отказ до вызова модели, 130 — Ctrl+C; при
# 1 и 130 отчёт пишется по пройденному (правила `local_llm.py`). Сессий на диск
# программа не пишет, индекс только читает, модели поиска из сети не качает.
#
# День 29 — **ресурсы**: память процесса модели LM Studio (перед первым ходом
# колонки и пик) и пик памяти самой программы (e5 и реранкер), отсчётами раз в
# `presets.LOCAL_MEMORY_SAMPLE_S` в потоке-демоне на колонку
# (`MemorySampler`). Меряет `local_server.footprint_mb()` / `server_memory_mb()`
# — только чтение, без сигналов; не нашли процесс — «н/д», прогон не
# отменяется. Квантование и формат загруженной модели — из
# `local_server.check()`, в условиях и разделе «Ресурсы». JSON дня 28 без поля
# `memory` читаются: строки раздела — «н/д».

import argparse
import dataclasses
import json
import logging
import os
import statistics
import sys
import threading
from datetime import datetime
from pathlib import Path

import agent
import local_llm
import local_server
import presets
import rag_answer
import rag_eval
import rag_search
from rag_eval import cell, money, num, quote

logger = logging.getLogger("rag_compare")

PROGRAM = "rag_compare"

# Все вопросы `rag-eval` в его порядке и по id — для прогона и для `--merge`.
QUESTIONS: tuple[dict, ...] = (
    *presets.RAG_CONTROL_QUESTIONS, *presets.RAG_FOLLOWUP_QUESTIONS, *presets.RAG_ANSWER_QUESTIONS,
)
QUESTION_BY_ID = {q["id"]: q for q in QUESTIONS}
CONTROLS = {q["id"]: q for q in presets.RAG_CONTROL_QUESTIONS}

# Поля индекса, которые идут в JSON и «Условия».
_INDEX_KEYS = (
    "path", "built_at", "model", "strategy", "lang", "chunks", "total", "candidates_k", "rerank_model",
    "threshold", "top_k",
)

# Поля записи и колонки, без которых отчёт не строится, — их проверяет `--merge`.
# Поля правок по ревью (`rounds`, `all_cost_usd`, `prev_error`) необязательны:
# JSON, записанные до правок, читаются, строки без данных — «н/д».
_RECORD_KEYS = frozenset({
    "col", "question", "mode", "repeat", "ok", "error", "answered_model", "finish_reason", "elapsed",
    "prompt_tokens", "completion_tokens", "cost_usd", "reasoning", "rewrite_elapsed", "rewrite_ok", "queries",
    "search_ok", "search_elapsed", "load_s", "rerank_load_s", "rerank_ok", "empty", "best", "hits", "before",
    "after", "has_check", "kind", "check_empty", "asks", "sources", "quotes", "verbatim", "violations",
    "repaired", "raw", "text", "prev_ok", "prev_text",
})
_COLUMN_KEYS = frozenset({
    "preset", "base_url", "temperature", "temperature_override", "max_tokens", "reasoning_effort", "thinking",
    "texts", "model_param", "server", "started", "built_at", "models",
})

# Служебные вызовы хода в `AgentReply` — для стоимости всех вызовов. У голого
# агента программы из них бывает только переписывание.
_SERVICE_CALLS = ("service_call", "memory_call", "route_call", "task_call", "guard_call", "rewrite_call")


# --- Мелочи ---------------------------------------------------------------------

def median(values: list[float]) -> float | None:
    return statistics.median(values) if values else None


def sec(value: float | None) -> str:
    return "—" if value is None else f"{value:.2f}"


def whole(value: float | None) -> str:
    return "—" if value is None else f"{value:,.0f}".replace(",", " ")


def speed(record: dict) -> float | None:
    """≈ток/с хода: `completion_tokens / elapsed` — с префиллом и раундами, а не
    чистая генерация (как в отчёте `./run.sh local`)."""
    if not record["ok"] or not record["completion_tokens"] or not record["elapsed"] or record["elapsed"] <= 0:
        return None
    return record["completion_tokens"] / record["elapsed"]


def answer_key(record: dict) -> str:
    """Текст модели с схлопнутыми пробелами — то, что сравнивается между
    повторами (`answer.raw`; у хода без проверки — собранный ответ)."""
    return " ".join((record.get("raw") or record.get("text") or "").split())


def of(count: int, base: int) -> str:
    return f"{count} из {base}"


def failure(record: dict) -> str | None:
    """Почему ход не годится в показатели: не удался он сам или его предыдущий
    ход (уточняющий тогда задан агенту без контекста — правка по ревью);
    `None` — ход удался."""
    if not record["ok"]:
        return f"ход не удался — {record['error']}"
    if record["prev_ok"] is False:
        return f"предыдущий ход не удался — {record.get('prev_error') or 'причина не записана'}"
    return None


def good(record: dict) -> bool:
    return failure(record) is None


def hit_ceiling(record: dict) -> bool:
    """Упор в потолок: хоть один раунд основного запроса кончился `length`. В
    JSON до правки по ревью раундов нет — по `finish_reason` хода."""
    rounds = record.get("rounds")
    if rounds is None:
        return record["finish_reason"] == "length"
    return any(item["finish_reason"] == "length" for item in rounds)


def unbroken_lengths(record: dict) -> list[int]:
    """`completion_tokens` раундов без обрыва: у хода с повтором формата сумма
    хода — два ответа, а не один (правка по ревью)."""
    rounds = record.get("rounds")
    if rounds is None:
        rounds = [record]
    return [item["completion_tokens"] for item in rounds
            if item["finish_reason"] != "length" and item["completion_tokens"]]


def known_costs(reply: agent.AgentReply | None) -> list[float]:
    """Известные стоимости всех вызовов хода: раунды основного запроса и
    служебные работы (у голого агента — переписывание)."""
    if reply is None:
        return []
    calls = [getattr(reply, name) for name in _SERVICE_CALLS] + list(reply.sampling_calls)
    costs = [reply.cost_usd] + [call.cost_usd for call in calls if call is not None]
    return [cost for cost in costs if cost is not None]


# --- Записи ---------------------------------------------------------------------

def make_record(col: int, question: dict, mode: rag_eval.Mode, repeat: int, run: rag_eval.Run) -> dict:
    """Плоская запись хода (§5.5): посчитанные поля и тексты, `AgentReply` целиком
    не идёт."""
    reply = run.reply
    rag = reply.rag
    found = rag_eval.record_of(run)
    check = rag_eval.answer_of(run)
    call = reply.rewrite_call
    # Модели поиска грузятся при первом поиске — у уточняющего это может быть
    # его предыдущий ход (`--only У1`).
    searches = [item for item in (rag, run.prev.rag if run.prev is not None else None) if item is not None]
    costs = known_costs(reply) + known_costs(run.prev)
    record = {
        "col": col, "question": question["id"], "mode": mode.key, "repeat": repeat,
        "ok": reply.ok, "error": reply.error, "answered_model": reply.answered_model,
        "finish_reason": reply.finish_reason, "elapsed": reply.elapsed,
        "prompt_tokens": reply.prompt_tokens, "completion_tokens": reply.completion_tokens,
        "cost_usd": reply.cost_usd, "reasoning": bool(reply.reasoning),
        "rounds": [
            {"completion_tokens": item.completion_tokens, "finish_reason": item.finish_reason, "repair": item.repair}
            for item in reply.rounds
        ],
        "all_cost_usd": sum(costs) if costs else None,
        "rewrite_elapsed": call.elapsed if call is not None else None,
        "rewrite_ok": call.ok if call is not None else None,
        "queries": list(rag.queries) if rag is not None else [],
        "search_ok": rag.ok if rag is not None else None,
        "search_elapsed": rag.elapsed if rag is not None else None,
        "load_s": sum(item.load_s for item in searches),
        "rerank_load_s": sum(item.rerank_load_s for item in searches),
        "rerank_ok": rag.rerank_ok if rag is not None else None,
        "empty": found.empty if found is not None else None,
        "best": found.best if found is not None else None,
        "hits": [hit.get("chunk_id") for hit in found.hits] if found is not None else [],
        "before": rag_eval.places_before(run, question) if mode.rag else [],
        "after": rag_eval.places_after(run, question) if mode.rag else [],
        "has_check": check is not None,
        "kind": check.kind if check is not None else None,
        "check_empty": check.empty if check is not None else None,
        "asks": check.asks if check is not None else None,
        "sources": list(check.sources) if check is not None else [],
        "quotes": len(check.quotes) if check is not None else 0,
        "verbatim": check.verbatim if check is not None else 0,
        "violations": list(check.violations) if check is not None else [],
        "repaired": bool(rag_eval.repair_rounds(run)),
        "raw": check.raw if check is not None else reply.text,
        "text": reply.text,
        "prev_ok": run.prev.ok if run.prev is not None else None,
        "prev_error": run.prev.error if run.prev is not None else None,
        "prev_text": run.prev.text if run.prev is not None else None,
    }
    return record


def make_column(config: agent.AgentConfig, override: float | None, started: datetime, index: dict) -> dict:
    return {
        "preset": config.name, "base_url": config.base_url, "temperature": config.temperature,
        "temperature_override": override, "max_tokens": config.max_tokens,
        "reasoning_effort": config.reasoning_effort, "thinking": config.thinking,
        "texts": presets.rag_texts_label(config.name), "model_param": config.model, "server": None,
        "started": f"{started:%Y-%m-%d %H:%M:%S}", "built_at": index.get("built_at", ""), "models": [],
        "memory": None,
    }


# --- Память (день 29) -----------------------------------------------------------

class MemorySampler:
    """Отсчёты памяти одной колонки (спецификация дня 29, §5.2): поток-демон раз
    в `step_s` снимает `footprint` процесса модели LM Studio (только у локальной
    колонки — `pattern` не пуст) и своего процесса. Только читает память — в
    агента не лезет. `start()` — перед первым ходом колонки (отсчёт «в покое»),
    `stop()` — после последнего, в том числе при Ctrl+C и сбое."""

    def __init__(self, pattern: str | None, step_s: float) -> None:
        self.pattern = pattern
        self.step_s = step_s
        self.server_idle: float | None = None
        self.server_peak: float | None = None
        self.program_peak: float | None = None
        self.samples = 0
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None

    def _sample(self, server: float | None = None) -> None:
        if self.pattern and server is None:
            server = local_server.server_memory_mb(self.pattern)
        program = local_server.footprint_mb(os.getpid())
        if server is not None:
            self.server_peak = server if self.server_peak is None else max(self.server_peak, server)
        if program is not None:
            self.program_peak = program if self.program_peak is None else max(self.program_peak, program)
        self.samples += 1

    def _run(self) -> None:
        while not self._stop.wait(self.step_s):
            self._sample()

    def start(self) -> None:
        if self.pattern:
            self.server_idle = local_server.server_memory_mb(self.pattern)
        self._sample(self.server_idle)
        self._thread = threading.Thread(target=self._run, name="rag-compare-memory", daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._stop.set()
        if self._thread is not None:
            # Отсчёт в работе кончится сам: у `footprint` и `pgrep` свой потолок.
            self._thread.join(timeout=3 * local_server.FOOTPRINT_TIMEOUT_S)

    def as_dict(self) -> dict:
        return {
            "server_idle_mb": self.server_idle, "server_peak_mb": self.server_peak,
            "program_peak_mb": self.program_peak, "samples": self.samples, "sample_s": self.step_s,
        }


def mb(value: float | None) -> str:
    return "н/д" if value is None else whole(value)


def _mb_unit(value: float | None) -> str:
    return "н/д" if value is None else f"{whole(value)} МБ"


def finalize_columns(columns: list[dict], records: list[dict]) -> None:
    """Фактические модели колонки — из ответов API, в порядке появления."""
    for number, column in enumerate(columns):
        models: list[str] = []
        for record in records:
            if record["col"] == number and record["answered_model"] and record["answered_model"] not in models:
                models.append(record["answered_model"])
        column["models"] = models or [column["model_param"]]


def column_label(column: dict) -> str:
    """Подпись колонки: пресет · фактическая модель · перекрытая температура (§2.1)."""
    label = f"{column['preset']} · {', '.join(column['models'])}"
    if column.get("temperature_override") is not None:
        label += f" · t={column['temperature_override']:g}"
    return label


def column_key(column: dict) -> tuple:
    return (column["preset"], tuple(column["models"]), column.get("temperature_override"))


# --- Показатели -----------------------------------------------------------------

class Data:
    """Записи прогона: колонки, режимы, вопросы и группы повторов. `interrupted` —
    пометки прерванных прогонов для начала отчёта (у `--merge` — по файлу)."""

    def __init__(
        self, columns: list[dict], records: list[dict], indexes: list[dict], interrupted: list[str] | None = None,
    ) -> None:
        self.columns = columns
        self.records = records
        self.indexes = indexes
        self.interrupted = interrupted or []
        modes_present = {r["mode"] for r in records}
        self.modes = [mode for mode in rag_eval.MODES if mode.key in modes_present]
        ids_present = {r["question"] for r in records}
        self.questions = [q for q in QUESTIONS if q["id"] in ids_present]
        self.groups: dict[tuple[int, str, str], list[dict]] = {}
        for record in sorted(records, key=lambda r: r["repeat"]):
            self.groups.setdefault((record["col"], record["mode"], record["question"]), []).append(record)
        # Пары «колонка × режим», в которых есть ходы, в порядке колонок и режимов.
        self.pairs = [
            (number, mode) for number in range(len(columns)) for mode in self.modes
            if any(r["col"] == number and r["mode"] == mode.key for r in records)
        ]
        self.repeats = max((r["repeat"] for r in records), default=0)

    def of(self, col: int, mode: rag_eval.Mode) -> list[dict]:
        return [r for r in self.records if r["col"] == col and r["mode"] == mode.key]

    def group(self, col: int, mode: rag_eval.Mode, question: dict) -> list[dict]:
        return self.groups.get((col, mode.key, question["id"]), [])

    def header(self, col: int, mode: rag_eval.Mode) -> str:
        return f"{column_label(self.columns[col])}<br>`{mode.key}`"


def quality_rows(records: list[dict]) -> list[tuple[str, str]]:
    """Показатели формы по записям одной колонки и режима (§2.3)."""
    ok = [r for r in records if good(r)]
    checked = [r for r in ok if r["has_check"]]
    excerpts = [r for r in checked if not r["check_empty"]]
    empty = [r for r in checked if r["check_empty"]]
    kinds = {kind: sum(1 for r in checked if r["kind"] == kind)
             for kind in (rag_answer.KIND_ANSWER, rag_answer.KIND_IDK, rag_answer.KIND_NO_REFS)}
    idk_empty = [r for r in empty if r["kind"] == rag_answer.KIND_IDK]
    with_sources = [r for r in records if r["after"]]
    total_sources = sum(len(r["after"]) for r in with_sources if r["search_ok"])
    before = sum(1 for r in with_sources if r["search_ok"] for place in r["before"] if place is not None)
    after = sum(1 for r in with_sources if r["search_ok"] for place in r["after"] if place is not None)
    return [
        ("ходов удачных / всего", of(len(ok), len(records))),
        ("по выдержкам / «не знаю» / без ссылок", f"{kinds[rag_answer.KIND_ANSWER]} / {kinds[rag_answer.KIND_IDK]} / "
                                                   f"{kinds[rag_answer.KIND_NO_REFS]}"),
        ("«не знаю» при выдержках, прошедших порог",
         of(sum(1 for r in excerpts if r["kind"] == rag_answer.KIND_IDK), len(excerpts))),
        ("пустых выдач / с «Не знаю» / из них с вопросом",
         f"{len(empty)} / {len(idk_empty)} / {sum(1 for r in idk_empty if r['asks'])}"),
        ("со списком источников (при выдержках)", of(sum(1 for r in excerpts if r["sources"]), len(excerpts))),
        ("цитат всего / дословно", f"{sum(r['quotes'] for r in checked)} / {sum(r['verbatim'] for r in checked)}"),
        ("ответов без нарушений", of(sum(1 for r in checked if not r["violations"]), len(checked))),
        ("повторов формата", str(sum(1 for r in checked if r["repaired"]))),
        ("ожидаемые источники: до → после (поиск, не модель)", f"{before} → {after} из {total_sources}"),
    ]


def cost_text(values: list[float | None]) -> str:
    known = [value for value in values if value is not None]
    return money(sum(known)) if known else "н/д"


def speed_rows(records: list[dict]) -> list[tuple[str, str]]:
    ok = [r for r in records if good(r)]
    times = [r["elapsed"] for r in ok]
    rates = [value for value in (speed(r) for r in ok) if value is not None]
    searched = [r for r in ok if r["search_ok"] and r["search_elapsed"] is not None]
    rewritten = [r["rewrite_elapsed"] for r in ok if r["rewrite_elapsed"] is not None]
    unbroken = [length for r in ok for length in unbroken_lengths(r)]
    # Деньги потрачены и на неудачных ходах — суммы по всем записям. Полной
    # стоимости нет в JSON до правки по ревью.
    all_costs = ("н/д — в записях нет" if any("all_cost_usd" not in r for r in records)
                 else cost_text([r["all_cost_usd"] for r in records]))
    load_e5 = sum(r["load_s"] or 0 for r in records)
    load_rank = sum(r["rerank_load_s"] or 0 for r in records)
    return [
        ("время основного вызова, с: медиана / макс / сумма",
         f"{sec(median(times))} / {sec(max(times) if times else None)} / {sec(sum(times)) if times else '—'}"),
        ("время переписывания, с: медиана", sec(median(rewritten))),
        ("время поиска, с: медиана (без загрузки моделей)", sec(median([r["search_elapsed"] for r in searched]))),
        ("первая загрузка e5 / реранкера, с (в медианы не входит)", f"{load_e5:.1f} / {load_rank:.1f}"),
        ("prompt_tokens: медиана", whole(median([r["prompt_tokens"] for r in ok if r["prompt_tokens"]]))),
        ("completion_tokens: медиана", whole(median([r["completion_tokens"] for r in ok if r["completion_tokens"]]))),
        ("≈ток/с: медиана (с префиллом)", whole(median(rates))),
        ("самый длинный ответ без обрыва (один раунд), токенов", whole(max(unbroken) if unbroken else None)),
        ("стоимость основного вызова, сумма", cost_text([r["cost_usd"] for r in records])),
        ("стоимость всех вызовов хода (с переписыванием и предыдущим ходом), сумма", all_costs),
    ]


def stability(group: list[dict]) -> dict:
    """Стабильность N повторов одного вопроса в одной колонке и режиме (§2.3).
    Меряется, только если удачных ответов не меньше двух (`measured`): одному
    ответу не с чем совпадать (правка по ревью); «повтор 1 отличается» — только
    если повтор 1 удался."""
    ok = [r for r in group if good(r)]
    keys = [answer_key(r) for r in ok]
    measured = len(ok) >= 2
    first = measured and ok[0]["repeat"] == 1
    rest = keys[1:] if first else []
    times = [r["elapsed"] for r in ok]
    return {
        "n": len(group), "ok": len(ok), "measured": measured, "distinct": len(set(keys)),
        "all_same": measured and len(set(keys)) == 1,
        "first_differs": first and keys[0] not in rest, "rest_same": first and len(set(rest)) == 1,
        "kind_same": measured and len({r["kind"] for r in ok}) == 1,
        "hits_same": measured and len({tuple(r["hits"]) for r in ok}) == 1,
        "queries": len({tuple(r["queries"]) for r in ok}),
        "failed": len(group) - len(ok), "length": sum(1 for r in ok if hit_ceiling(r)),
        "t_min": min(times) if times else None, "t_med": median(times), "t_max": max(times) if times else None,
        "spread": (max(times) - min(times)) if measured else None,
    }


def stability_label(info: dict) -> str:
    if not info["measured"]:
        return "— (меньше двух ответов)"
    if info["all_same"]:
        return f"✓ {info['ok']} одинаковых"
    if info["first_differs"] and info["rest_same"]:
        return "≠ повтор 1"
    return f"≠ {info['distinct']} разных"


def stability_summary(data: Data, col: int, mode: rag_eval.Mode) -> dict:
    infos = [stability(data.group(col, mode, q)) for q in data.questions if data.group(col, mode, q)]
    spreads = [i["spread"] for i in infos if i["spread"] is not None]
    return {
        "questions": len(infos),
        "measured": sum(1 for i in infos if i["measured"]),
        "all_same": sum(1 for i in infos if i["all_same"]),
        "first_differs": sum(1 for i in infos if i["first_differs"]),
        "kind_same": sum(1 for i in infos if i["kind_same"]),
        "hits_same": sum(1 for i in infos if i["hits_same"]),
        "failed": sum(i["failed"] for i in infos), "length": sum(i["length"] for i in infos),
        "spread_med": median(spreads), "spread_max": max(spreads) if spreads else None,
    }


# --- Отчёт ----------------------------------------------------------------------

def metrics_text(record: dict) -> str:
    parts = [f"время модели {record['elapsed']:.2f} с"]
    parts.append(f"prompt {num(record['prompt_tokens'])} · completion {num(record['completion_tokens'])}")
    rate = speed(record)
    parts.append(f"≈{rate:.0f} ток/с" if rate is not None else "≈ток/с н/д")
    parts.append(f"стоимость {money(record['cost_usd'])}")
    parts.append(f"finish_reason `{record['finish_reason'] or 'н/д'}`")
    if record["rewrite_elapsed"] is not None:
        parts.append(f"переписывание {record['rewrite_elapsed']:.2f} с")
    if record["search_ok"] and record["search_elapsed"] is not None:
        parts.append(f"поиск {record['search_elapsed']:.2f} с · выдержек {len(record['hits'])}")
    if record["has_check"]:
        parts.append(f"вид «{record['kind']}» · цитат {record['quotes']}/{record['verbatim']} дословно · "
                     f"нарушений {len(record['violations'])}")
    return " · ".join(parts)


def conditions(data: Data) -> list[str]:
    columns = data.columns
    index = data.indexes[0] if data.indexes else {}
    questions = ", ".join(q["id"] for q in data.questions)
    out = [
        "## 1. Условия", "",
        "- Прогоны: " + "; ".join(f"{column_label(c)} — {c['started']}" for c in columns),
        (
            f"- Индекс: `{index.get('path', 'н/д')}` · собран {index.get('built_at', 'н/д')} · модель эмбеддингов "
            f"`{index.get('model', 'н/д')}` · нарезка `{index.get('strategy', 'н/д')}` · язык "
            f"`{index.get('lang', 'н/д')}` · кусков: {index.get('chunks', 'н/д')} (всего в индексе "
            f"{index.get('total', 'н/д')})"
        ),
        (
            f"- Этапы поиска: этап 1 — {index.get('candidates_k', presets.RAG_CANDIDATES)} ближайших кусков → "
            f"этап 2 — реранкер `{index.get('rerank_model', presets.RAG_RERANK_MODEL)}` ≥ "
            f"{rag_eval.threshold_text(index.get('threshold', presets.RAG_RERANK_THRESHOLD))} → в запрос не больше "
            f"{index.get('top_k', presets.RAG_TOP_K)}"
        ),
        f"- Вопросы ({len(data.questions)}): {questions}; в уточняющих предыдущим ходом идёт их контрольный вопрос",
        "- Режимы: " + "; ".join(f"`{m.key}` — {m.label}" for m in data.modes)
        + ". `фильтр` — переписывания нет, выдача у всех колонок одна и та же, сравнивается только генерация; "
        "`полный` — запрос переписывает модель колонки, сравнивается система целиком",
        f"- Повторов на вопрос: до {data.repeats}; на каждый — свежий голый агент. Порядок проходов: повтор за "
        "повтором — повтор 1 по всем вопросам и режимам, потом повтор 2 и так далее",
        "- Агенты: голые — пресет и системный промпт, без хранилища, памяти, профиля, инвариантов, задачи и "
        "инструментов MCP; тексты RAG — по пресету (день 27)",
        "",
        "Колонки:", "",
    ]
    for column in columns:
        line = (
            f"- **{column_label(column)}** — пресет «{column['preset']}» · API `{column['base_url']}` · температура "
            f"{column['temperature']} · `max_tokens` {num(column['max_tokens']) if column['max_tokens'] else 'нет'} · "
            f"`reasoning_effort` {column['reasoning_effort'] if column['reasoning_effort'] is not None else 'не передаётся'}"
            f" · thinking {'включён' if column['thinking'] else 'выключен'} · инструкции RAG: {column['texts']}"
        )
        server = column.get("server")
        if server:
            line += " · загружено на старте прогона: " + (
                ", ".join(f"`{m['id']}`" + (f" ({model_details(m)})" if model_details(m) else "")
                          for m in server["models"])
                or "сервер не сообщил"
            )
        out.append(line)
    out += [""]
    if any(column["base_url"] == presets.LOCAL_BASE_URL or column.get("server") for column in columns):
        out += [
            "- Кэш префикса LM Studio (проверка дня 28, §2.5): при t = 0 ответ локальной модели зависит от того, был ли "
            "такой запрос в кэше, а кэш живёт до выгрузки модели. Если прогон начат сразу после `lms load`, у "
            "локальной колонки повтор 1 идёт с холодным кэшем, повторы 2..N — с тёплым; программа это не проверяет.",
        ]
    if len({column["built_at"] for column in columns}) > 1:
        out += ["- **Колонки считались на разных индексах** (даты сборки: "
                + ", ".join(f"{column_label(c)} — {c['built_at'] or 'н/д'}" for c in columns) + ")."]
    out += [
        "- В отчёте нет оценок: код считает форму (проверка ответа дня 24) и числа; верно ли ответила модель, "
        "оценивает автор; «лучшей модели» и подсветки колонок нет. «≈ток/с» — `completion_tokens` / время "
        "основного вызова, то есть с префиллом (день 26), а не чистая генерация.",
        "",
    ]
    return out


def model_details(model: dict) -> str:
    """«MLX · 4bit · контекст 132 096 из 262 144» — что знает запись модели
    колонки; у записей дня 28 формата и квантования нет."""
    parts = [(model.get("format") or "").upper(), model.get("quantization") or "", model.get("context") or ""]
    return " · ".join(part for part in parts if part)


def resources_section(data: Data) -> list[str]:
    """Раздел «Ресурсы» (день 29, §5.2): по колонкам — память на прогон колонки,
    а не на режим."""
    out = ["## 4. Ресурсы", ""]
    if not data.columns:
        return out + ["Колонок нет.", ""]
    step = next((c["memory"]["sample_s"] for c in data.columns if c.get("memory")), presets.LOCAL_MEMORY_SAMPLE_S)

    def started_model(column: dict) -> str:
        server = column.get("server")
        if not server:
            return "— (облако)" if column["base_url"] != presets.LOCAL_BASE_URL else "н/д"
        return "<br>".join(f"`{m['id']}`" + (f" · {model_details(m)}" if model_details(m) else "")
                           for m in server["models"]) or "сервер не сообщил"

    def server_memory(column: dict) -> str:
        memory = column.get("memory")
        if not memory:
            return "н/д — в записях нет"
        if not column.get("server") and column["base_url"] != presets.LOCAL_BASE_URL:
            return "— (облако)"
        return f"{mb(memory['server_idle_mb'])} / {mb(memory['server_peak_mb'])}"

    def program_memory(column: dict) -> str:
        memory = column.get("memory")
        return mb(memory["program_peak_mb"]) if memory else "н/д — в записях нет"

    def samples(column: dict) -> str:
        memory = column.get("memory")
        return f"{memory['samples']} · раз в {memory['sample_s']:g} с" if memory else "н/д"

    rows = (
        ("модель на старте: id · формат · квантование · контекст", started_model),
        ("память процесса модели LM Studio, МБ: перед первым ходом / пик", server_memory),
        ("память программы (поиск: e5 и реранкер), пик, МБ", program_memory),
        ("отсчётов и шаг", samples),
    )
    out += [
        "| показатель | " + " | ".join(cell(column_label(c)) for c in data.columns) + " |",
        "| --- | " + " | ".join("---" for _ in data.columns) + " |",
    ]
    for label, fn in rows:
        out.append(f"| {label} | " + " | ".join(cell(fn(c)) for c in data.columns) + " |")
    out += [
        "",
        f"`footprint` (macOS) — с памятью видеокарты; пик — по отсчётам раз в {step:g} с, короткий всплеск может не "
        "попасть; у облачной колонки памяти модели нет; время загрузки модели программа не меряет — его пишет "
        "человек.",
        "",
    ]
    return out


def metric_table(data: Data, rows_fn) -> list[str]:
    """Таблица «показатель × (колонка, режим)»; `rows_fn(records)` — пары
    (показатель, значение) по записям пары."""
    per_pair = [rows_fn(data.of(col, mode)) for col, mode in data.pairs]
    if not per_pair:
        return ["Ходов нет.", ""]
    labels = [label for label, _ in per_pair[0]]
    out = [
        "| показатель | " + " | ".join(data.header(col, mode) for col, mode in data.pairs) + " |",
        "| --- | " + " | ".join("---" for _ in data.pairs) + " |",
    ]
    for number, label in enumerate(labels):
        out.append(f"| {label} | " + " | ".join(cell(rows[number][1]) for rows in per_pair) + " |")
    return out + [""]


def stability_section(data: Data) -> list[str]:
    out = ["## 5. Стабильность", ""]
    if data.repeats < 2:
        out += ["Повторов меньше двух: стабильность не измеряется (`--repeat` ≥ 2).", ""]
    summary = {pair: stability_summary(data, pair[0], pair[1]) for pair in data.pairs}

    def row(label: str, fn) -> str:
        return f"| {label} | " + " | ".join(cell(fn(summary[pair])) for pair in data.pairs) + " |"

    out += [
        "Сводка по колонке и режиму. Стабильность меряется у вопросов, где удачных ответов не меньше двух, — "
        "«из K» в строках ниже первой считается от них; «одинаковы» — тексты модели после схлопывания пробелов:", "",
        "| показатель | " + " | ".join(data.header(col, mode) for col, mode in data.pairs) + " |",
        "| --- | " + " | ".join("---" for _ in data.pairs) + " |",
        row("вопросов с двумя и больше удачными ответами", lambda s: of(s["measured"], s["questions"])),
        row("все N ответов одинаковы", lambda s: of(s["all_same"], s["measured"])),
        row("повтор 1 отличается от остальных", lambda s: of(s["first_differs"], s["measured"])),
        row("вид ответа одинаков", lambda s: of(s["kind_same"], s["measured"])),
        row("выдача одинакова", lambda s: of(s["hits_same"], s["measured"])),
        row("сбоев / упоров в потолок всего", lambda s: f"{s['failed']} / {s['length']}"),
        row("разброс времени (макс − мин), с: медиана / максимум",
            lambda s: f"{sec(s['spread_med'])} / {sec(s['spread_max'])}"),
        "",
        "По вопросам: «✓ N одинаковых», «≠ повтор 1» (повтор 1 отличается, остальные совпали) или «≠ K разных»; "
        "вид, выдача, запросов поиска (`полный`), упоры в потолок, время мин / медиана / макс.", "",
        "| № | " + " | ".join(data.header(col, mode) for col, mode in data.pairs) + " |",
        "| --- | " + " | ".join("---" for _ in data.pairs) + " |",
    ]
    for question in data.questions:
        cells = []
        for col, mode in data.pairs:
            group = data.group(col, mode, question)
            if not group:
                cells.append("—")
                continue
            info = stability(group)
            lines = [stability_label(info)]
            if info["measured"]:
                lines.append(f"вид: {'✓' if info['kind_same'] else '≠'} · выдача: {'✓' if info['hits_same'] else '≠'}")
            if mode.rewrite:
                lines.append(f"запросов поиска: {info['queries']}")
            lines.append(f"упоров: {info['length']} · сбоев: {info['failed']}")
            if info["t_med"] is not None:
                lines.append(f"время: {info['t_min']:.1f} / {info['t_med']:.1f} / {info['t_max']:.1f}")
            cells.append("<br>".join(cell(line) for line in lines))
        out.append(f"| {question['id']} | " + " | ".join(cells) + " |")
    return out + [""]


def by_question_section(data: Data) -> list[str]:
    out = ["## 6. По вопросам", "",
           "Первая ячейка колонки — вид ответа, нарушений, медиана времени, стабильность; вторая — «смысл (автор)», "
           "её заполняет автор.", ""]
    columns = sorted({col for col, _ in data.pairs})
    for mode in data.modes:
        cols = [col for col in columns if (col, mode) in data.pairs]
        if not cols:
            continue
        out += [f"### Режим `{mode.key}` — {mode.label}", ""]
        head = ["№"]
        for col in cols:
            head += [column_label(data.columns[col]), "смысл (автор)"]
        out += ["| " + " | ".join(cell(h) for h in head) + " |", "| " + " | ".join("---" for _ in head) + " |"]
        for question in data.questions:
            cells = [question["id"]]
            for col in cols:
                group = data.group(col, mode, question)
                if not group:
                    cells += ["—", ""]
                    continue
                ok = [r for r in group if good(r)]
                kinds = list(dict.fromkeys(r["kind"] or "—" for r in ok))
                violations = [len(r["violations"]) for r in ok]
                info = stability(group)
                lines = [
                    "вид: " + (" / ".join(kinds) if kinds else "ход не удался"),
                    "нарушений: " + (str(violations[0]) if len(set(violations)) == 1 else "/".join(map(str, violations)))
                    if violations else "нарушений: —",
                    f"время, медиана: {sec(info['t_med'])} с",
                    stability_label(info),
                ]
                cells += ["<br>".join(cell(line) for line in lines), ""]
            out.append("| " + " | ".join(cells) + " |")
        out.append("")
    return out


def answers_section(data: Data) -> list[str]:
    out = ["## 7. Ответы", "",
           "Собранный ответ (`AgentReply.text`) каждого различающегося ответа с кратностью, строка метрик и, в "
           "`полном`, запрос поиска. Различие — по тексту модели после схлопывания пробелов.", ""]
    for question in data.questions:
        for mode in data.modes:
            members = [(col, data.group(col, mode, question)) for col in range(len(data.columns))]
            members = [(col, group) for col, group in members if group]
            if not members:
                continue
            out += [f"### {question['id']} · `{mode.key}`", "", f"**Вопрос:** {question['question']}", ""]
            for col, group in members:
                out += [f"**{column_label(data.columns[col])}**", ""]
                first = group[0]
                if question.get("after") and first["prev_text"] is not None:
                    out += [f"_Предыдущий ход ({question['after']}) — ответ первого повтора:_", "",
                            quote(first["prev_text"]) if first["prev_ok"] else "⚠️ ход не удался", ""]
                variants: dict[str, list[dict]] = {}
                for record in group:
                    reason = failure(record)
                    variants.setdefault(answer_key(record) if reason is None else f"сбой: {reason}", []).append(record)
                for records in variants.values():
                    lead = records[0]
                    reason = failure(lead)
                    numbers = [r["repeat"] for r in records]
                    count = f"×{len(records)}" + ("" if len(records) == len(group) else
                                                  f" (повтор{'ы' if len(numbers) > 1 else ''} " + ", ".join(map(str, numbers)) + ")")
                    if not lead["ok"]:
                        out += [f"{count} ⚠️ {reason}", ""]
                        continue
                    # Предыдущий ход не удался — ответ есть, но задан без контекста.
                    out += [count + (f" ⚠️ {reason}; вопрос задан без контекста" if reason else ""), "",
                            quote(lead["text"]), "", f"_{metrics_text(lead)}_", ""]
                    if mode.rewrite and lead["queries"]:
                        out += ["_Запросы поиска: " + " · ".join(f"«{cell(q)}»" for q in lead["queries"]) + "_", ""]
    return out


def build_report(data: Data) -> str:
    out = ["# Сравнение моделей в RAG: качество, скорость, ресурсы, стабильность (дни 28–29)", ""]
    for note in data.interrupted:
        out += [f"**Прервано:** {note}", ""]
    out += conditions(data)
    out += ["## 2. Качество (форма)", "",
            "Код проверяет форму: номера выдержек, дословность цитат, «Не знаю» и уточняющий вопрос (день 24). "
            "«Не знаю» при выдержках, прошедших порог, — не обязательно ошибка: решает автор. Строки «до → после» "
            "относятся к поиску, а не к модели: в `фильтре` у колонок они совпадают — это проверка стенда.", "",
            *metric_table(data, quality_rows)]
    out += ["## 3. Скорость", "", *metric_table(data, speed_rows)]
    out += resources_section(data)
    out += stability_section(data)
    out += by_question_section(data)
    out += answers_section(data)
    return "\n".join(out).rstrip() + "\n"


def print_summaries(report: str) -> None:
    for title, start, stop in (
        ("Качество (форма)", "## 2. Качество (форма)", "## 3. "),
        ("Скорость", "## 3. Скорость", "## 4. "),
        ("Ресурсы", "## 4. Ресурсы", "## 5. "),
        ("Стабильность", "## 5. Стабильность", "По вопросам:"),
    ):
        print()
        print(title + report.partition(start)[2].partition(stop)[0].rstrip())


# --- JSON и --merge -------------------------------------------------------------

def dump_json(path: Path, data: dict) -> None:
    path.write_text(json.dumps(data, ensure_ascii=False, indent=1), encoding="utf-8")


def load_json(path: Path) -> dict:
    """Читает запись прошлого прогона; ошибка — `ValueError` с причиной."""
    try:
        data = json.loads(path.expanduser().read_text(encoding="utf-8"))
    except OSError as exc:
        raise ValueError(f"{path}: не читается ({exc.strerror or exc})") from exc
    except ValueError as exc:
        raise ValueError(f"{path}: не JSON ({exc})") from exc
    if (
        not isinstance(data, dict) or data.get("program") != PROGRAM
        or not isinstance(data.get("columns"), list) or not isinstance(data.get("records"), list)
    ):
        raise ValueError(f"{path}: не записи программы {PROGRAM}")
    # Форма записей — до сборки отчёта: битая запись дала бы трейсбек посреди
    # отчёта, а не отказ (правка по ревью).
    columns = data["columns"]
    for number, column in enumerate(columns, 1):
        missing = _COLUMN_KEYS - column.keys() if isinstance(column, dict) else _COLUMN_KEYS
        if missing or not isinstance(column["models"], list):
            raise ValueError(f"{path}: колонка {number} не той формы (нет полей: {', '.join(sorted(missing)) or 'models'})")
    for number, record in enumerate(data["records"], 1):
        missing = _RECORD_KEYS - record.keys() if isinstance(record, dict) else _RECORD_KEYS
        if missing:
            raise ValueError(f"{path}: запись {number} не той формы (нет полей: {', '.join(sorted(missing))})")
        if not isinstance(record["col"], int) or not 0 <= record["col"] < len(columns):
            raise ValueError(f"{path}: запись {number} ссылается на колонку {record['col']!r}, а колонок {len(columns)}")
        if record["mode"] not in rag_eval.MODE_BY_KEY or record["question"] not in QUESTION_BY_ID:
            raise ValueError(f"{path}: запись {number} — неизвестный режим или вопрос "
                             f"({record['mode']!r}, {record['question']!r})")
    return data


def merge(paths: list[Path]) -> Data:
    """Общий набор записей из файлов прошлых прогонов в порядке аргумента; колонки
    — в том же порядке. Одна и та же колонка дважды — `ValueError`."""
    columns: list[dict] = []
    records: list[dict] = []
    indexes: list[dict] = []
    interrupted: list[str] = []
    seen: dict[tuple, Path] = {}
    for path in paths:
        data = load_json(path)
        if data.get("interrupted"):
            interrupted.append(f"`{path}` — {data['interrupted']}; колонки этого файла неполные")
        offset = len(columns)
        for column in data["columns"]:
            key = column_key(column)
            if key in seen:
                raise ValueError(
                    f"колонка «{column_label(column)}» есть и в {seen[key]}, и в {path} — "
                    "одна и та же колонка (пресет, модели, температура) дважды"
                )
            seen[key] = path
            columns.append(column)
        records += [{**record, "col": record["col"] + offset} for record in data["records"]]
        if data.get("index"):
            indexes.append(data["index"])
    return Data(columns, records, indexes, interrupted)


# --- Прогон ---------------------------------------------------------------------

def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="./run.sh rag-compare",
        description=(
            "Сравнение локальной и облачной модели в RAG (день 28): 16 вопросов rag-eval в режимах `фильтр` и "
            "`полный`, каждый — по N повторов, на повтор свежий голый агент; отчёт Markdown (качество — форма, "
            "скорость, стабильность) и JSON записей. Колонка — пресет; локальная колонка — модель, загруженная в "
            "LM Studio (запускайте сразу после lms load: так повтор 1 идёт с холодным кэшем префикса). Локальные "
            "колонки двух моделей — два прогона и сборка --merge. Нужны индекс (./run.sh index), реранкер в кэше "
            "(./run.sh index --probe) и, для «Базового», ключ DeepSeek (src/.env)."
        ),
    )
    parser.add_argument("--presets", metavar="Локальный,Базовый", default=None,
                        help="колонки через запятую — имена пресетов (по умолчанию "
                        + ",".join(presets.RAG_COMPARE_PRESETS) + ")")
    parser.add_argument("--modes", metavar="фильтр,полный", default=None,
                        help="режимы через запятую из: " + ", ".join(m.key for m in rag_eval.MODES)
                        + " (по умолчанию " + ",".join(presets.RAG_COMPARE_MODES) + ")")
    parser.add_argument("--repeat", type=int, default=None, metavar="N",
                        help=f"повторов на вопрос, N ≥ 1 (по умолчанию {presets.RAG_COMPARE_REPEAT})")
    parser.add_argument("--only", metavar="К5,К10,У1", default=None,
                        help="часть вопросов, как у rag-eval; уточняющий тянет свой контрольный предыдущим ходом")
    parser.add_argument("--temperature", type=float, default=None, metavar="T",
                        help="температура основного запроса у всех колонок прогона (по умолчанию — пресета)")
    parser.add_argument("--db", type=Path, default=None, metavar="ПУТЬ",
                        help=f"файл индекса (по умолчанию {rag_search.shown(presets.RAG_INDEX_DB)})")
    parser.add_argument("--out", type=Path, default=None, metavar="ФАЙЛ.md",
                        help="куда писать отчёт; рядом пишется .json с тем же именем (по умолчанию "
                        f"{rag_search.shown(presets.RAG_COMPARE_DIR)}/<дата-время>.md)")
    parser.add_argument("--merge", metavar="a.json,b.json", default=None,
                        help="собрать общий отчёт из записей прошлых прогонов, без индекса и без вызовов модели; "
                        "с ним допустим только --out")
    return parser.parse_args(argv)


def _refuse(message: str) -> None:
    logger.error("%s", message)
    sys.exit(2)


def _items(value: str | None, default: tuple[str, ...]) -> list[str]:
    """Значения флага через запятую; не задан — умолчание. Пустая строка — пустой
    список, а не умолчание: `--presets ""` — отказ (правка по ревью)."""
    text = ",".join(default) if value is None else value
    return [item.strip() for item in text.split(",") if item.strip()]


def _out_path(args: argparse.Namespace, default: Path) -> Path:
    """Путь отчёта: `.json` — отказ (JSON прогона пишется рядом с тем же именем и
    затёр бы отчёт, у `--merge` — вход), каталог — отказ (правка по ревью)."""
    path = args.out.expanduser() if args.out is not None else default
    if path.suffix.lower() == ".json":
        _refuse(f"--out {path}: отчёт — Markdown (.md), JSON записей пишется рядом с тем же именем")
    if path.is_dir():
        _refuse(f"--out {path}: это каталог, нужен файл отчёта .md")
    return path


def run_merge(args: argparse.Namespace) -> None:
    flags = ("presets", "modes", "repeat", "only", "temperature", "db")
    given = [f"--{name}" for name in flags if getattr(args, name) is not None]
    if given:
        _refuse(f"--merge не берёт флаги прогона ({', '.join(given)}); допустим только --out")
    paths = [Path(item.strip()) for item in args.merge.split(",") if item.strip()]
    if not paths:
        _refuse("--merge: не назван ни один файл")
    out_path = _out_path(args, presets.RAG_COMPARE_DIR / f"{datetime.now():%Y-%m-%d-%H%M%S}-merge.md")
    try:
        data = merge(paths)
    except ValueError as exc:
        _refuse(f"--merge: {exc}")
    report = build_report(data)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("общий отчёт из %d файлов, колонок %d, записей %d: %s", len(paths), len(data.columns),
                len(data.records), rag_search.shown(out_path))
    print_summaries(report)
    sys.exit(0)


def run_turn(question: dict, mode: rag_eval.Mode, config: agent.AgentConfig,
             index: rag_search.RulesIndex) -> rag_eval.Run:
    """Ход `rag_eval.run_mode()`; созданные им агенты после хода снимаются с
    учёта (правка по ревью): иначе реестр процесса держит каждого вместе с его
    клиентом OpenAI и соединением до конца прогона — по умолчанию 192 агента,
    по дескриптору на каждого (замер на заглушке сервера: 30 агентов — +30).
    Снятого агента освобождает сборщик мусора (ссылки циклические): открытых
    дескрипторов остаются единицы-десятки, а не по одному на ход."""
    created = agent.process_stats()["agents_created"]
    try:
        return rag_eval.run_mode(question, CONTROLS, mode, config, index)
    finally:
        for bare in agent.agents():
            if bare.number > created:
                agent.delete_agent(bare.number)


def run_all(configs: list[agent.AgentConfig], questions: list[dict], modes: list[rag_eval.Mode], repeat: int,
            index: rag_search.RulesIndex, records: list[dict], columns: list[dict]) -> str | None:
    """Ходы всех колонок в порядке §5.1 — повтор за повтором; записи дописываются
    в `records` (при Ctrl+C пройденное остаётся у вызывающего). Возвращает
    причину остановки на первом сбое хода, `None` — все ходы удались. Память
    колонки (день 29) пишется в `columns[номер]["memory"]` и при прерывании — по
    пройденному."""
    for number, config in enumerate(configs):
        logger.info("колонка %d из %d: пресет «%s» · %s", number + 1, len(configs), config.name, config.base_url)
        # Шаблон читается при запуске колонки, а не при импорте: проверка «н/д»
        # (§13, п. 8) перекрывает его из REPL.
        local = config.base_url == presets.LOCAL_BASE_URL
        sampler = MemorySampler(presets.LOCAL_SERVER_PROCESS if local else None, presets.LOCAL_MEMORY_SAMPLE_S)
        sampler.start()
        if local:
            if sampler.server_idle is None:
                logger.info("память процесса модели: н/д — процесс LM Studio не найден по шаблону «%s»",
                            presets.LOCAL_SERVER_PROCESS)
            else:
                logger.info("память процесса модели: %s МБ", whole(sampler.server_idle))
        try:
            reason = _run_column(number, config, questions, modes, repeat, index, records)
        finally:
            sampler.stop()
            columns[number]["memory"] = sampler.as_dict()
            logger.info(
                "колонка %d: пик памяти процесса модели %s · программы %s · отсчётов %d", number + 1,
                _mb_unit(sampler.server_peak) if local else "— (облако)", _mb_unit(sampler.program_peak),
                sampler.samples,
            )
        if reason is not None:
            return reason
    return None


def _run_column(number: int, config: agent.AgentConfig, questions: list[dict], modes: list[rag_eval.Mode],
                repeat: int, index: rag_search.RulesIndex, records: list[dict]) -> str | None:
    """Ходы одной колонки — повтор за повтором; причина остановки на первом
    сбое, `None` — все удались."""
    for attempt in range(1, repeat + 1):
        for question in questions:
            for mode in modes:
                run = run_turn(question, mode, config, index)
                record = make_record(number, question, mode, attempt, run)
                records.append(record)
                log_record(config, question, mode, attempt, repeat, record)
                reason = failure(record)
                if reason is not None:
                    logger.error("сбой — пишу отчёт по пройденному")
                    return f"{config.name} · {question['id']} · {mode.key} · повтор {attempt}: {reason}"
    return None


def main() -> None:
    args = _args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[RAG-сравнение] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    if args.merge is not None:
        run_merge(args)

    # --- Отказы до вызова модели ---
    names = _items(args.presets, presets.RAG_COMPARE_PRESETS)
    unknown = [name for name in names if name not in presets.PRESETS]
    if unknown or not names:
        _refuse(f"нет пресетов: {', '.join(unknown) or '(пусто)'}; есть: "
                + ", ".join(f"«{name}»" for name in presets.PRESETS))
    names = list(dict.fromkeys(names))
    keys = _items(args.modes, presets.RAG_COMPARE_MODES)
    unknown = [key for key in keys if key not in rag_eval.MODE_BY_KEY]
    if unknown or not keys:
        _refuse(f"нет режимов: {', '.join(unknown) or '(пусто)'}; есть: " + ", ".join(m.key for m in rag_eval.MODES))
    modes = [rag_eval.MODE_BY_KEY[key] for key in dict.fromkeys(keys)]
    repeat = presets.RAG_COMPARE_REPEAT if args.repeat is None else args.repeat
    if repeat < 1:
        _refuse(f"--repeat {repeat}: нужно не меньше 1")
    questions = list(QUESTIONS)
    if args.only is not None:
        wanted = [item.translate(rag_eval._LATIN_ID) for item in _items(args.only, ())]
        known = {q["id"] for q in questions}
        missing = [item for item in wanted if item not in known]
        if missing or not wanted:
            _refuse(f"нет вопросов: {', '.join(missing) or '(пусто)'}; есть: " + ", ".join(q["id"] for q in questions))
        questions = [q for q in questions if q["id"] in wanted]

    configs: list[agent.AgentConfig] = []
    for name in names:
        config = presets.PRESETS[name]
        if args.temperature is not None:
            try:
                config = dataclasses.replace(config, temperature=args.temperature)
            except ValueError as exc:
                _refuse(str(exc))
        configs.append(config)

    started = datetime.now()
    out_path = _out_path(args, presets.RAG_COMPARE_DIR / f"{started:%Y-%m-%d-%H%M%S}.md")
    try:
        out_path.parent.mkdir(parents=True, exist_ok=True)
    except OSError as exc:
        _refuse(f"--out {out_path}: каталог не создаётся ({exc.strerror or exc}) — прогон отменён, модель не вызывалась")

    index = rag_search.RulesIndex(
        (args.db or presets.RAG_INDEX_DB).expanduser(), presets.RAG_SEARCH_STRATEGY, presets.RAG_SEARCH_LANG,
        presets.RAG_TOP_K, candidates_k=presets.RAG_CANDIDATES, rerank_model=presets.RAG_RERANK_MODEL,
        threshold=presets.RAG_RERANK_THRESHOLD, rerank_max_length=presets.RAG_RERANK_MAX_LENGTH,
        rerank_batch=presets.RAG_RERANK_BATCH, allow_download=False, expected_model=presets.RAG_EMBED_MODEL,
    )
    info = index.info()
    if not info["ok"]:
        _refuse(f"{info['error']} — прогон отменён, модель не вызывалась")
    if any(mode.rerank for mode in modes) and not info["rerank_cached"]:
        _refuse(
            f"реранкера {presets.RAG_RERANK_MODEL} нет в кэше Hugging Face — ./run.sh index --probe скачает ≈1,5 ГБ; "
            "прогон отменён, модель не вызывалась"
        )

    index_info = {key: info.get(key) for key in _INDEX_KEYS}
    columns: list[dict] = []
    for config in configs:
        column = make_column(config, args.temperature, started, info)
        if config.base_url == presets.LOCAL_BASE_URL:
            check = local_llm.check_server(config.base_url)
            column["server"] = {"models": [
                {"id": model.id, "context": model.context_text or "", "format": model.format,
                 "quantization": model.quantization}
                for model in check.loaded
            ]}
        elif config.api_key_env is not None and not os.environ.get(config.api_key_env):
            _refuse(f"нет ключа {config.api_key_env} в src/.env — прогон отменён, модель не вызывалась")
        columns.append(column)

    # Вопросы уточняющих ходят с предыдущим ходом — это ещё один вызов `ask()`.
    calls = len(questions) + sum(1 for q in questions if q.get("after"))
    logger.info(
        "индекс %s · собран %s · колонки: %s · режимы: %s · повторов: %d · вопросов: %d · ходов ≈%d на колонку",
        info["path"], info["built_at"], ", ".join(config.name for config in configs),
        ", ".join(mode.key for mode in modes), repeat, len(questions), calls * len(modes) * repeat,
    )

    records: list[dict] = []
    try:
        interrupted = run_all(configs, questions, modes, repeat, index, records, columns)
        code = 0 if interrupted is None else 1
    except KeyboardInterrupt:
        interrupted, code = _passed(records), 130
        logger.error("прервано %s — пишу отчёт по пройденному", interrupted)
    except Exception as exc:  # noqa: BLE001 — оплаченные ходы важнее трейсбека (правило rag_dialog)
        interrupted, code = f"{_passed(records)}: сбой — {exc}", 1
        logger.exception("сбой %s — пишу отчёт по пройденному", _passed(records))

    finalize_columns(columns, records)
    # Колонки, до которых прогон не дошёл, в отчёт не идут.
    used = sorted({record["col"] for record in records})
    remap = {old: new for new, old in enumerate(used)}
    kept = [columns[old] for old in used]
    kept_records = [{**record, "col": remap[record["col"]]} for record in records]
    payload = {
        "program": PROGRAM, "started": f"{started:%Y-%m-%d %H:%M:%S}", "index": index_info, "columns": kept,
        "records": kept_records, "interrupted": interrupted,
    }
    # JSON — первым: отчёт строится из записей, и сбой его сборки или записи не
    # должен терять оплаченные ходы — по JSON отчёт соберёт `--merge` (правка по
    # ревью).
    json_path = out_path.with_suffix(".json")
    dump_json(json_path, payload)
    logger.info("записи: %s (%d ходов)", rag_search.shown(json_path), len(records))
    notes = [] if interrupted is None else [f"{interrupted} — в отчёте пройденные ходы ({len(records)})"]
    report = build_report(Data(kept, kept_records, [index_info], notes))
    out_path.write_text(report, encoding="utf-8")
    logger.info("отчёт: %s", rag_search.shown(out_path))
    print_summaries(report)
    sys.exit(code)


def log_record(config: agent.AgentConfig, question: dict, mode: rag_eval.Mode, attempt: int, repeat: int,
               record: dict) -> None:
    head = f"{config.name} · {question['id']} · {mode.key} · повтор {attempt}/{repeat}: "
    if not record["ok"]:
        logger.info("%s⚠️ %s", head, failure(record))
        return
    rate = speed(record)
    line = (
        f"{head}{record['elapsed']:.2f} с · prompt {num(record['prompt_tokens'])} / completion "
        f"{num(record['completion_tokens'])} · ≈{'н/д' if rate is None else f'{rate:.0f}'} ток/с"
    )
    if record["kind"]:
        line += f" · {record['kind']}"
    if hit_ceiling(record):
        line += " · ⚠️ finish_reason length (упор в потолок)"
    if record["prev_ok"] is False:
        line += f" · ⚠️ {failure(record)}"
    logger.info("%s", line)


def _passed(records: list[dict]) -> str:
    if not records:
        return "до первого хода"
    last = records[-1]
    return f"после {last['question']} · {last['mode']} · повтор {last['repeat']}"


if __name__ == "__main__":
    main()
