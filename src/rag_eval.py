# TooManyRules — прогон контрольных вопросов в режимах RAG (день 22, неделя 5;
# режимы второго этапа — день 23).
#
# **Отдельная программа**, как `rag_index.py`: её запускает человек
# (`./run.sh rag-eval`), и работает она до конца, а не живёт. Приложение её не
# импортирует. Вопросы, пресет и параметры поиска берёт из `presets.py`, к модели
# ходит только через `Agent.ask()` (LLM API вызывается только в `agent.py`),
# выдержки ищет тот же `rag_search.RulesIndex`, что и приложение.
#
# Агент **голый** (спецификация дня 22, §2.7): пресет и его системный промпт, без
# хранилища, стратегий кроме «Всей истории», памяти, профиля, инвариантов,
# задачи и инструментов — режимы отличаются только последним сообщением и тем,
# что искал поиск, это чистое A/B. Режимов пять (день 23, §2.5, §7.1): `без`
# RAG, `простой` (день 22), `фильтр` (второй этап без переписывания),
# `переписывание` (без второго этапа) и `полный` (оба); по умолчанию — четыре
# режима RAG. На каждый вопрос и каждый режим — свежий агент, по очереди, без
# параллельности; уточняющий вопрос тот же агент задаёт сразу после своего
# контрольного (в отчёт идёт второй ход, первый — строкой «предыдущий ход»).
# Сессий на диск программа не пишет — данные автора не трогает.
#
# Отчёт — Markdown, готовый к переносу в документ сравнения. **Оценок в нём нет**:
# «найден» относится только к выдаче (`rag_search.sources_found()`), а совпадение
# ответа с ожиданием, верность ссылок и выдумки оценивает автор; «лучший режим»
# и «лучший порог» программа не называет — таблица порогов это пересчёт без
# новых вызовов (§2.7).
#
# Логи — stdout: строки агентов (как в приложении) и свои с префиксом
# `[RAG-прогон]`.

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import agent
import presets
import rag_search

logger = logging.getLogger("rag_eval")

# Латинские K/k и Y/y в номере вопроса принимаем за кириллические: на клавиатуре
# «К1» и «У1» легко набрать не той раскладкой.
_LATIN_ID = str.maketrans({"K": "К", "k": "К", "к": "К", "Y": "У", "y": "У", "у": "У"})


@dataclass(frozen=True)
class Mode:
    """Режим RAG: комбинация флагов агента (§2.5). Это не данные игры, поэтому
    определения здесь, а не в `presets.py`."""

    key: str
    label: str
    rag: bool
    rewrite: bool
    rerank: bool


MODES: tuple[Mode, ...] = (
    Mode("без", "без RAG", False, False, False),
    Mode("простой", "простой (день 22)", True, False, False),
    Mode("фильтр", "фильтр", True, False, True),
    Mode("переписывание", "переписывание", True, True, False),
    Mode("полный", "переписывание + фильтр", True, True, True),
)
DEFAULT_MODES = ("простой", "фильтр", "переписывание", "полный")
MODE_BY_KEY = {mode.key: mode for mode in MODES}


@dataclass
class Run:
    """Один ход одного режима: ответ на вопрос и, у уточняющего, ответ на его
    контрольный (`prev`)."""

    reply: agent.AgentReply
    prev: agent.AgentReply | None = None


@dataclass
class Result:
    """Вопрос и ходы по режимам (в порядке прогона; при прерывании — только
    пройденные)."""

    question: dict
    runs: dict[str, Run] = field(default_factory=dict)


@dataclass(frozen=True)
class Settings:
    """Числа второго этапа этого прогона (`--threshold`, `--candidates`,
    `--top-k` перекрывают `presets.py`)."""

    threshold: float
    candidates: int
    top_k: int


def num(value: int | float | None) -> str:
    if value is None:
        return "н/д"
    return f"{value:,}".replace(",", " ") if isinstance(value, int) else f"{value:.2f}"


def money(value: float | None) -> str:
    return "н/д" if value is None else f"${value:.5f}"


def found_text(places: list[int | None]) -> str:
    """«2 из 2: места 1, 3» / «1 из 2: место 4» / «0 из 2» / «—» (у вопроса без
    источников). Про ответ модели это не говорит ничего — только про выдачу."""
    if not places:
        return "—"
    found = [place for place in places if place is not None]
    if not found:
        return f"0 из {len(places)}"
    word = "места" if len(found) > 1 else "место"
    return f"{len(found)} из {len(places)}: {word} " + ", ".join(str(place) for place in found)


def pages(hit: dict) -> str:
    first, last = hit["page_from"], hit["page_to"]
    return str(first) if first == last else f"{first}–{last}"


def cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def quote(text: str) -> str:
    return "\n".join(f"> {line}" if line.strip() else ">" for line in text.strip().splitlines()) or ">"


def threshold_text(value: float) -> str:
    return f"{value:.2f}"


# --- Что нашла выдача -----------------------------------------------------------

def record_of(run: Run) -> agent.RagRecord | None:
    """Запись поиска хода, если поиск удался; иначе `None` (режим без RAG, сбой
    поиска)."""
    record = run.reply.rag
    return record if record is not None and record.ok else None


def places_after(run: Run, question: dict) -> list[int | None]:
    """Места ожидаемых источников в том, что ушло в модель."""
    record = record_of(run)
    if record is None:
        return [None] * len(question["sources"])
    return rag_search.sources_found(record.hits, question["sources"])


def places_before(run: Run, question: dict) -> list[int | None]:
    """Места ожидаемых источников среди кандидатов этапа 1; без второго этапа
    «до» и «после» — одна и та же выдача."""
    record = record_of(run)
    if record is None:
        return [None] * len(question["sources"])
    if record.rerank_ok:
        return rag_search.sources_found(record.candidates, question["sources"])
    return rag_search.sources_found(record.hits, question["sources"])


def source_status(source: tuple[str, str, int], run: Run | None) -> str:
    """Где ожидаемый источник в режиме `run` — строка `rag_search.source_status()`
    (одно определение с панелью приложения); поиска нет — «н/д»."""
    record = record_of(run) if run is not None else None
    if record is None:
        return "н/д"
    return rag_search.source_status(source, record.hits, record.candidates, record.rerank_ok)


# --- Метрики --------------------------------------------------------------------

def metrics_line(run: Run, mode: Mode) -> str:
    reply = run.reply
    parts = [
        f"время модели {reply.elapsed:.2f} с",
        f"prompt {num(reply.prompt_tokens)} · completion {num(reply.completion_tokens)}",
        f"стоимость {money(reply.cost_usd)}",
        f"finish_reason `{reply.finish_reason or 'н/д'}`",
    ]
    call = reply.rewrite_call
    if mode.rewrite and call is not None:
        parts.append(
            f"переписывание {call.elapsed:.2f} с · ≈{num(call.total_tokens)} ток. · {money(call.cost_usd)}"
            + ("" if call.ok else f" · ⚠️ {call.error}")
        )
    record = reply.rag
    if mode.rag and record is not None:
        found = f"поиск {record.elapsed:.2f} с"
        extra = []
        if record.load_s:
            extra.append(f"загрузка модели эмбеддингов {record.load_s:.1f} с")
        if record.rerank_load_s:
            extra.append(f"загрузка реранкера {record.rerank_load_s:.1f} с")
        if extra:
            found += f" ({', '.join(extra)})"
        parts.append(found)
        parts.append(f"выдержки ≈{num(record.tokens)} ток. (оценка)")
    return " · ".join(parts)


def search_line(run: Run) -> str:
    """Запрос поиска и что с ним сделали — для логов и заголовка ответа."""
    record = run.reply.rag
    if record is None:
        return ""
    query = record.query or record.question
    if record.rewrite and not record.rewrite_ok:
        return f"⚠️ переписывание не удалось ({record.rewrite_error}) — искали по вопросу"
    if record.rewrite and " ".join(query.split()) == " ".join(record.question.split()):
        return "без изменений"
    return f"«{query}»" if record.rewrite else "вопрос как есть"


def answer_block(run: Run, mode: Mode, question: dict) -> list[str]:
    reply = run.reply
    out: list[str] = []
    if run.prev is not None:
        prev = run.prev
        out += [f"_Предыдущий ход ({question.get('after', '')}) — ответ:_", ""]
        out += [quote(prev.text) if prev.ok else f"⚠️ ход не удался: {prev.error}", ""]
    if not reply.ok:
        return out + [f"⚠️ ход не удался: {reply.error}", ""]
    out += [quote(reply.text), "", f"_{metrics_line(run, mode)}_", ""]
    return out


# --- Отчёт ----------------------------------------------------------------------

def mode_runs(results: list[Result], mode: Mode) -> list[tuple[dict, Run]]:
    return [(r.question, r.runs[mode.key]) for r in results if mode.key in r.runs]


def summary_rows(results: list[Result], modes: list[Mode]) -> list[str]:
    out = [
        "| режим | ожидаемых источников в запросе | выдержек | токенов выдержек (оценка / эмбеддинги) | "
        "пустая выдача | prompt_tokens | время, с: переписывание / поиск / модель | стоимость: основной / переписывание |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for mode in modes:
        runs = mode_runs(results, mode)
        if not runs:
            continue
        found = total = hits = tok_est = tok_emb = prompt = 0
        t_rewrite = t_search = t_model = c_main = c_rewrite = 0.0
        empty: list[str] = []
        failed = 0
        for question, run in runs:
            reply = run.reply
            if not reply.ok:
                failed += 1
                continue
            prompt += reply.prompt_tokens or 0
            t_model += reply.elapsed
            c_main += reply.cost_usd or 0.0
            if reply.rewrite_call is not None:
                t_rewrite += reply.rewrite_call.elapsed
                c_rewrite += reply.rewrite_call.cost_usd or 0.0
            record = record_of(run)
            if mode.rag:
                total += len(question["sources"])
                found += sum(1 for place in places_after(run, question) if place is not None)
                if record is not None:
                    hits += len(record.hits)
                    tok_est += record.tokens
                    tok_emb += sum(hit["tokens"] for hit in record.hits)
                    t_search += record.elapsed + record.rerank_load_s
                    if record.empty:
                        empty.append(question["id"])
        if mode.rag:
            sources = f"{found} из {total}"
            hits_text, tokens_text = str(hits), f"{num(tok_est)} / {num(tok_emb)}"
            empty_text = ", ".join(empty) if empty else "—"
            search_text = f"{t_search:.2f}"
        else:
            sources = hits_text = tokens_text = empty_text = "—"
            search_text = "—"
        note = f" (ходов не удалось: {failed})" if failed else ""
        out.append(
            f"| {mode.label}{note} | {sources} | {hits_text} | {tokens_text} | {empty_text} | {num(prompt)} | "
            f"{t_rewrite:.2f} / {search_text} / {t_model:.2f} | {money(c_main)} / {money(c_rewrite)} |"
        )
    return out


def question_cell(run: Run | None, mode: Mode, question: dict) -> str:
    if run is None:
        return "—"
    if not run.reply.ok:
        return "⚠️ ход не удался"
    record = record_of(run)
    lines: list[str] = []
    if mode.rag:
        if question["sources"]:
            after = places_after(run, question)
            if record is not None and record.rerank_ok:
                before = places_before(run, question)
                lines.append(f"источники: {found_text(before)} → {found_text(after)}")
            else:
                lines.append(f"источники: {found_text(after)}")
        if record is None:
            lines.append("⚠️ поиск не удался")
        else:
            lines.append(f"выдержек: {len(record.hits)}" + (" (пусто по порогу)" if record.empty else ""))
            if record.best is not None:
                lines.append(f"лучшая оценка: {record.best:.3f}")
    lines.append(f"prompt {num(run.reply.prompt_tokens)} · {money(run.reply.cost_usd)}")
    return "<br>".join(cell(line) for line in lines)


def sweep_table(results: list[Result], mode: Mode, settings: Settings, sweep: tuple[float, ...]) -> list[str]:
    """Пересчёт порогов по оценкам кандидатов режима (§2.7): что прошло бы при
    каждом значении — без новых вызовов. Таблица, а не рекомендация."""
    rows = [
        (question, record_of(run)) for question, run in mode_runs(results, mode)
        if record_of(run) is not None and record_of(run).rerank_ok
    ]
    out = [f"**{mode.label}** — {len(rows)} вопросов с оценками кандидатов", ""]
    if not rows:
        return out + ["Оценок кандидатов нет (второй этап не удался или режим не прошёл).", ""]
    total = sum(len(question["sources"]) for question, _ in rows)
    out += [
        f"| порог | выдержек (потолок {settings.top_k}) | ожидаемых источников (из {total}) | пустая выдача |",
        "| --- | --- | --- | --- |",
    ]
    for threshold in sweep:
        hits = found = 0
        empty: list[str] = []
        for question, record in rows:
            ranked = sorted(record.candidates, key=lambda c: c["rerank_place"])
            kept = [c for c in ranked if c["rerank_score"] >= threshold][:settings.top_k]
            hits += len(kept)
            found += sum(1 for place in rag_search.sources_found(kept, question["sources"]) if place is not None)
            if not kept:
                empty.append(question["id"])
        mark = " (в прогоне)" if abs(threshold - settings.threshold) < 1e-9 else ""
        out.append(
            f"| {threshold_text(threshold)}{mark} | {hits} | {found} | {', '.join(empty) if empty else '—'} |"
        )
    return out + [""]


def candidates_table(record: agent.RagRecord) -> list[str]:
    out = [
        "| место до | близость | оценка | место после | судьба | раздел | страницы | токенов |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for place, c in enumerate(record.candidates, 1):
        part = f" · часть {c['part']}" if c.get("part") else ""
        out.append(
            f"| {place} | {c['score']:.3f} | {c['rerank_score']:.3f} | {c['rerank_place']} | {c['fate']} | "
            f"{cell(c['section'] + part)} | {pages(c)} | {c['tokens']} |"
        )
    return out


def hits_list(record: agent.RagRecord) -> str:
    items = []
    for place, hit in enumerate(record.hits, 1):
        score = hit["rerank_score"] if hit.get("rerank_score") is not None else hit["score"]
        part = f" · часть {hit['part']}" if hit.get("part") else ""
        items.append(f"{place}. {cell(hit['section'] + part)} (стр. {pages(hit)}, {score:.3f})")
    return "; ".join(items) if items else "пусто"


def build_report(
    results: list[Result], config: agent.AgentConfig, info: dict, modes: list[Mode], settings: Settings,
    started: datetime, interrupted: str | None, total_questions: int,
) -> str:
    out: list[str] = ["# Прогон контрольных вопросов: режимы RAG (дни 22-23)", ""]
    if interrupted is not None:
        out += [f"**Прервано: {interrupted}** — в отчёте пройденные вопросы ({len(results)} из {total_questions}).", ""]

    device = (
        f"{info.get('rerank_device', 'н/д')}, {info.get('rerank_dtype', 'н/д')}"
        if info.get("rerank_loaded") else "н/д (не загружался)"
    )
    has_rerank = any(mode.rerank for mode in modes)
    out += [
        "## 1. Условия", "",
        f"- Дата и время: {started:%Y-%m-%d %H:%M:%S}",
        f"- Пресет: «{config.name}» — модель `{config.model}`, thinking {'включён' if config.thinking else 'выключен'}",
        f"- Индекс: `{info['path']}` · собран {info['built_at']} · модель эмбеддингов `{info['model']}` · "
        f"нарезка `{info['strategy']}` · язык `{info['lang']}` · "
        f"кусков нарезки и языка: {info['chunks']} (всего в индексе {info['total']})",
        f"- Этап 1: {settings.candidates} ближайших кусков (top-K «до»), близость — косинусная, модель эмбеддингов",
        (
            f"- Этап 2: реранкер `{info['rerank_model']}` ({device}), порог {threshold_text(settings.threshold)}, "
            f"в запрос — не больше {settings.top_k} (top-K «после»)"
            if has_rerank else f"- Этап 2: в прогоне не участвует; в запрос — {settings.top_k} ближайших"
        ),
        "- Агенты: голые — пресет и системный промпт, без хранилища, стратегий, памяти, профиля, инвариантов, "
        "задачи и инструментов MCP; на каждый вопрос и режим — свежий агент, уточняющий задаётся тем же агентом "
        "сразу после своего контрольного",
        "- Режимы: " + "; ".join(f"`{mode.key}` — {mode.label}" for mode in modes),
        "- Промпт переписывания — `presets.RAG_REWRITE_PROMPT` (потолок ответа "
        f"`RAG_REWRITE_MAX_TOKENS` = {presets.RAG_REWRITE_MAX_TOKENS}), инструкция пустой выдачи — "
        "`presets.RAG_EMPTY_INSTRUCTION`, инструкция к выдержкам — `presets.RAG_INSTRUCTION`; "
        "близость и оценки реранкера в запрос не уходят",
        "- В отчёте нет оценок: «найден» относится только к выдаче, ответы оценивает автор; «лучший режим» и "
        "«лучший порог» программа не называет",
        "",
    ]

    out += ["## 2. Сводка по режимам", "", *summary_rows(results, modes), ""]
    out += [
        "Ходы уточняющих вопросов считаются, предыдущие (контрольные) ходы тех же агентов — нет. Время поиска "
        "включает загрузку реранкера, если она пришлась на этот ход; загрузка модели эмбеддингов в сводку не входит.",
        "",
    ]

    out += ["## 3. Таблица по вопросам", ""]
    header = "| № | " + " | ".join(mode.label for mode in modes) + " |"
    out += [header, "| --- | " + " | ".join("---" for _ in modes) + " |"]
    for result in results:
        cells = [
            question_cell(result.runs.get(mode.key), mode, result.question) for mode in modes
        ]
        out.append(f"| {result.question['id']} | " + " | ".join(cells) + " |")
    out.append("")

    out += ["## 4. Порог: пересчёт по оценкам кандидатов", ""]
    rerank_modes = [mode for mode in modes if mode.rerank]
    if rerank_modes:
        out += [
            "Что прошло бы в запрос при каждом пороге из `presets.RAG_THRESHOLD_SWEEP` — пересчёт оценок "
            "кандидатов этого прогона, без новых вызовов. Таблица, а не рекомендация.", "",
        ]
        for mode in rerank_modes:
            out += sweep_table(results, mode, settings, presets.RAG_THRESHOLD_SWEEP)
    else:
        out += ["В прогоне нет режимов со вторым этапом.", ""]

    out += ["## 5. По вопросам", ""]
    full = next((mode for mode in modes if mode.key == "полный"), None) or next(iter(rerank_modes), None)
    for result in results:
        question = result.question
        out += [
            f"### {question['id']}", "",
            f"**Вопрос:** {question['question']}", "",
            f"**Что проверяет:** {question['checks']}", "",
            f"**Ожидание:** {question['expected']}", "",
        ]
        if question.get("after"):
            out += [f"**Задаётся после {question['after']} тем же агентом.**", ""]
        if question["sources"]:
            out += ["**Ожидаемые источники:**", ""]
            for source in question["sources"]:
                out.append(f"- `{source[0]}` · «{source[1]}» · стр. {source[2]}")
                for mode in modes:
                    if mode.rag and mode.key in result.runs:
                        out.append(f"  - {mode.label}: {source_status(source, result.runs[mode.key])}")
        else:
            out.append("**Ожидаемые источники:** — (вопрос вне корпуса)")
        out.append("")
        queries = [
            f"- {mode.label}: {search_line(result.runs[mode.key])}"
            for mode in modes if mode.rewrite and mode.key in result.runs and result.runs[mode.key].reply.rag is not None
        ]
        if queries:
            out += ["**Запрос поиска:**", "", *queries, ""]
        if full is not None and full.key in result.runs:
            record = record_of(result.runs[full.key])
            if record is not None and record.rerank_ok:
                out += [
                    f"**Кандидаты этапа 1 — режим «{full.label}»** (запрос поиска: «{cell(record.query)}»):", "",
                    *candidates_table(record), "",
                ]
        for mode in modes:
            run = result.runs.get(mode.key)
            if run is None:
                continue
            out.append(f"**Ответ — {mode.label}:**")
            out.append("")
            record = record_of(run)
            if mode.rag:
                if record is None:
                    reason = run.reply.rag.error if run.reply.rag is not None else "запись поиска не пришла"
                    out += [f"⚠️ Поиск не удался: {reason} — запрос ушёл без выдержек.", ""]
                else:
                    notes = ""
                    if record.rerank and not record.rerank_ok:
                        notes = f" · ⚠️ второй этап не удался: {record.rerank_error}"
                    out += [f"Выдача: {hits_list(record)}{notes}", ""]
            out += answer_block(run, mode, question)
    return "\n".join(out).rstrip() + "\n"


# --- Прогон ---------------------------------------------------------------------

def make_agent(
    config: agent.AgentConfig, name: str, index: rag_search.RulesIndex, mode: Mode,
) -> agent.Agent:
    bare = agent.Agent(
        config, session_id=name, retriever=index, rag_instruction=presets.RAG_INSTRUCTION,
        rag_rewrite_prompt=presets.RAG_REWRITE_PROMPT, rag_rewrite_max_tokens=presets.RAG_REWRITE_MAX_TOKENS,
        rag_empty_instruction=presets.RAG_EMPTY_INSTRUCTION,
    )
    if not mode.rag:
        bare.set_rag_in_request(False)
    bare.set_rag_stages(mode.rewrite, mode.rerank)
    return bare


def log_run(question: dict, mode: Mode, run: Run) -> None:
    reply = run.reply
    line = f"{question['id']} · {mode.label}: "
    if reply.ok:
        line += (
            f"{reply.elapsed:.2f} с · prompt {num(reply.prompt_tokens)} / completion "
            f"{num(reply.completion_tokens)} · {money(reply.cost_usd)}"
        )
    else:
        line += f"⚠️ ход не удался: {reply.error}"
    call = reply.rewrite_call
    if call is not None:
        line += f" · переписывание {call.elapsed:.2f} с: {search_line(run)}"
    record = reply.rag
    if record is not None:
        if record.ok:
            places = places_after(run, question)
            line += (
                f" · поиск {record.elapsed:.2f} с"
                f"{f' (загрузка модели {record.load_s:.1f} с)' if record.load_s else ''}"
                f"{f' (загрузка реранкера {record.rerank_load_s:.1f} с)' if record.rerank_load_s else ''}"
                f" · выдержек {len(record.hits)} ≈{num(record.tokens)} ток."
            )
            if record.rerank_ok:
                line += f" · до → после: {found_text(places_before(run, question))} → {found_text(places)}"
                if record.empty:
                    line += " · пусто по порогу"
            else:
                line += f" · ожидаемые источники в выдаче: {found_text(places)}"
                if record.rerank and record.rerank_error:
                    line += f" · ⚠️ второй этап: {record.rerank_error}"
        else:
            line += f" · ⚠️ поиск: {record.error}"
    logger.info("%s", line)


def run_mode(question: dict, controls: dict[str, dict], mode: Mode, config: agent.AgentConfig,
             index: rag_search.RulesIndex) -> Run:
    """Один режим одного вопроса: свежий агент; уточняющий — после своего
    контрольного."""
    bare = make_agent(config, f"eval-{question['id']}-{mode.key}", index, mode)
    prev = None
    if question.get("after"):
        control = controls[question["after"]]
        prev = bare.ask(control["question"])
        logger.info("%s · %s: предыдущий ход %s %s", question["id"], mode.label, control["id"],
                    "удался" if prev.ok else f"⚠️ не удался: {prev.error}")
    run = Run(bare.ask(question["question"]), prev)
    log_run(question, mode, run)
    return run


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="./run.sh rag-eval",
        description=(
            "Прогон контрольных (К1-К10) и уточняющих (У1-У3) вопросов в режимах RAG: на каждый вопрос и режим "
            "свежий голый агент; отчёт Markdown. Нужен ключ DeepSeek (src/.env), собранный индекс "
            "(./run.sh index) и, для режимов со вторым этапом, реранкер в кэше (./run.sh index --probe)."
        ),
    )
    parser.add_argument(
        "--modes", metavar="простой,полный", default=",".join(DEFAULT_MODES),
        help="режимы через запятую из: " + ", ".join(mode.key for mode in MODES)
        + f" (по умолчанию {','.join(DEFAULT_MODES)}; `без,простой` повторяет прогон дня 22)",
    )
    parser.add_argument("--only", metavar="К1,У1", help="часть вопросов (номера из RAG_CONTROL_QUESTIONS и "
                        "RAG_FOLLOWUP_QUESTIONS; уточняющий тянет свой контрольный как «предыдущий ход»)")
    parser.add_argument("--threshold", type=float, default=presets.RAG_RERANK_THRESHOLD, metavar="ЧИСЛО",
                        help=f"порог второго этапа на этот прогон (по умолчанию {presets.RAG_RERANK_THRESHOLD})")
    parser.add_argument("--candidates", type=int, default=presets.RAG_CANDIDATES, metavar="N",
                        help=f"top-K «до»: кандидатов этапа 1 (по умолчанию {presets.RAG_CANDIDATES})")
    parser.add_argument("--top-k", type=int, default=presets.RAG_TOP_K, metavar="N", dest="top_k",
                        help=f"top-K «после»: потолок выдачи (по умолчанию {presets.RAG_TOP_K})")
    parser.add_argument("--preset", default=presets.RAG_EVAL_PRESET, metavar="ИМЯ",
                        help=f"пресет агента (по умолчанию «{presets.RAG_EVAL_PRESET}»)")
    parser.add_argument("--db", type=Path, default=presets.RAG_INDEX_DB, metavar="ПУТЬ",
                        help=f"файл индекса (по умолчанию {rag_search.shown(presets.RAG_INDEX_DB)})")
    parser.add_argument("--out", type=Path, default=None, metavar="ФАЙЛ.md",
                        help=f"куда писать отчёт (по умолчанию {rag_search.shown(presets.RAG_EVAL_DIR)}/<дата-время>.md)")
    return parser.parse_args(argv)


def main() -> None:
    args = _args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[RAG-прогон] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    config = presets.PRESETS.get(args.preset)
    if config is None:
        logger.error("нет пресета «%s»; есть: %s", args.preset, ", ".join(f"«{name}»" for name in presets.PRESETS))
        sys.exit(2)

    keys = [item.strip() for item in args.modes.split(",") if item.strip()]
    unknown_modes = [key for key in keys if key not in MODE_BY_KEY]
    if unknown_modes or not keys:
        logger.error("нет режимов: %s; есть: %s", ", ".join(unknown_modes) or "(пусто)",
                     ", ".join(mode.key for mode in MODES))
        sys.exit(2)
    modes = [MODE_BY_KEY[key] for key in dict.fromkeys(keys)]

    controls = {q["id"]: q for q in presets.RAG_CONTROL_QUESTIONS}
    questions = list(presets.RAG_CONTROL_QUESTIONS) + list(presets.RAG_FOLLOWUP_QUESTIONS)
    if args.only:
        wanted = [item.strip().translate(_LATIN_ID) for item in args.only.split(",") if item.strip()]
        known = {q["id"] for q in questions}
        unknown = [item for item in wanted if item not in known]
        if unknown:
            logger.error("нет вопросов: %s; есть: %s", ", ".join(unknown), ", ".join(q["id"] for q in questions))
            sys.exit(2)
        questions = [q for q in questions if q["id"] in wanted]
    settings = Settings(args.threshold, args.candidates, args.top_k)

    index = rag_search.RulesIndex(
        args.db.expanduser(), presets.RAG_SEARCH_STRATEGY, presets.RAG_SEARCH_LANG, settings.top_k,
        candidates_k=settings.candidates, rerank_model=presets.RAG_RERANK_MODEL, threshold=settings.threshold,
        rerank_max_length=presets.RAG_RERANK_MAX_LENGTH, rerank_batch=presets.RAG_RERANK_BATCH,
        allow_download=False, expected_model=presets.RAG_EMBED_MODEL,
    )
    info = index.info()
    if not info["ok"]:
        # Сравнение без индекса бессмысленно: отказ до первого вызова модели.
        logger.error("%s — прогон отменён, модель не вызывалась", info["error"])
        sys.exit(2)
    if any(mode.rerank for mode in modes) and not info["rerank_cached"]:
        # Прогон, в котором «фильтр» тихо стал «простым», бесполезен (§7.2).
        logger.error(
            "реранкера %s нет в кэше Hugging Face — ./run.sh index --probe скачает ≈1,5 ГБ; прогон отменён, "
            "модель не вызывалась", presets.RAG_RERANK_MODEL,
        )
        sys.exit(2)
    stages = (
        f"этап 1: {settings.candidates} → этап 2: {presets.RAG_RERANK_MODEL} ≥ {threshold_text(settings.threshold)} "
        f"→ ≤{settings.top_k}"
        if any(mode.rerank for mode in modes) else f"второго этапа нет, top-{settings.top_k}"
    )
    logger.info(
        "индекс %s · собран %s · %s/%s: %d кусков из %d · %s · пресет «%s» (%s) · вопросов: %d · режимы: %s",
        info["path"], info["built_at"], info["strategy"], info["lang"], info["chunks"], info["total"],
        stages, config.name, config.model, len(questions), ", ".join(mode.key for mode in modes),
    )

    started = datetime.now()
    out_path = (args.out.expanduser() if args.out is not None
                else presets.RAG_EVAL_DIR / f"{started:%Y-%m-%d-%H%M%S}.md")
    results: list[Result] = []
    interrupted: str | None = None
    try:
        for question in questions:
            result = Result(question)
            results.append(result)
            for mode in modes:
                result.runs[mode.key] = run_mode(question, controls, mode, config, index)
    except KeyboardInterrupt:
        done = [r for r in results if len(r.runs) == len(modes)]
        interrupted = f"после {done[-1].question['id']}" if done else "до первого вопроса"
        logger.error("прервано %s — пишу отчёт по пройденному", interrupted)

    report = build_report(
        results, config, index.info(), modes, settings, started, interrupted, len(questions),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("отчёт: %s (%d из %d вопросов)", rag_search.shown(out_path), len(results), len(questions))
    print()
    print("Сводка по режимам" + report.partition("## 2. Сводка по режимам")[2].partition("## 3. ")[0].rstrip())
    print()
    print("Порог" + report.partition("## 4. Порог")[2].partition("## 5. ")[0].rstrip())
    sys.exit(130 if interrupted is not None else 0)


if __name__ == "__main__":
    main()
