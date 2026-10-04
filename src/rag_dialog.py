# TooManyRules — прогон длинных сценариев с памятью задачи (день 25, неделя 5).
#
# **Отдельная программа**, как `rag_eval.py`: её запускает человек
# (`./run.sh rag-dialog`), и работает она до конца, а не живёт. Приложение её не
# импортирует. Сценарии А и Б (14 и 13 сообщений), пресет, стратегию и тексты
# берёт из `presets.py`, к модели ходит только через `Agent.ask()` (LLM API
# вызывается только в `agent.py`), выдержки ищет тот же `rag_search.RulesIndex`,
# что и приложение. Мелкие помощники отчёта (`num`, `money`, `cell`,
# `found_text`, `threshold_text`) и разбор выдачи — импортом из `rag_eval.py`,
# без копий: ребро между программами, как `faq_watch.py` → `faq_server.py`.
#
# Агент на сценарий и режим — **свежий**: пресет, `make_strategies()` и
# стратегия сценария («Скользящее окно»: цель из первого сообщения уходит из
# запроса на пятом ходе, и дальше её несёт только память задачи), `make_memory()`
# и все параметры RAG дней 22-25; без хранилища, долговременной памяти,
# профиля, инвариантов, задачи и инструментов MCP. Режимов два (§5): `память` —
# ассистент как есть, и `без` — тот же агент, у которого рабочий слой снят в
# «Слоях памяти в запросе» (`set_request_layers([долговременная])`): разбор
# памяти идёт, но память задачи не уходит ни в запрос, ни в переписывание, ни
# в инструкцию. Сессий на диск программа не пишет — данные автора не трогает;
# индекс и кэш моделей только читаются.
#
# Отчёт — Markdown, данные для видео. **Оценок в нём нет**: «найден» относится
# только к выдаче, «верная ссылка» — к тому, была ли запись в запросе; не
# теряет ли ассистент цель и условия игрока и совпадает ли смысл ответа с
# источниками (выдержками и записями памяти), решает автор — две пустые колонки
# в таблице ходов.
#
# Логи — stdout: строки агентов (как в приложении) и свои с префиксом
# `[RAG-диалог]`. Коды выхода: 0 — прогон целиком, 2 — отказ до вызова модели,
# 130 — Ctrl+C, 1 — сбой посреди прогона (оба — с отчётом по пройденному).

import argparse
import logging
import sys
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path

import agent
import memory
import presets
import rag_answer
import rag_eval
import rag_search
from rag_eval import cell, found_text, money, num, quote, threshold_text

logger = logging.getLogger("rag_dialog")

# Латинские A и B в номере сценария принимаем за кириллические: на клавиатуре
# «А» и «Б» легко набрать не той раскладкой (правило `rag_eval`).
_LATIN_ID = str.maketrans({"A": "А", "a": "А", "а": "А", "B": "Б", "b": "Б", "б": "Б"})


@dataclass(frozen=True)
class DialogMode:
    """Режим прогона (§5): положение переключателя «рабочая» в «Слоях памяти в
    запросе». Не данные игры, поэтому определения здесь, а не в `presets.py`."""

    key: str
    label: str
    memory: bool


MODES: tuple[DialogMode, ...] = (
    DialogMode("память", "с памятью задачи", True),
    DialogMode("без", "без памяти задачи", False),
)
DEFAULT_MODES = tuple(mode.key for mode in MODES)
MODE_BY_KEY = {mode.key: mode for mode in MODES}


@dataclass
class Turn:
    """Один ход одного режима: ответ агента и память задачи после хода (записи
    «ключ, значение, раздел» — копия, дальше память меняется)."""

    step: dict
    reply: agent.AgentReply
    memory: list[dict]


@dataclass
class Dialog:
    """Сценарий в одном режиме (при прерывании — только пройденные ходы)."""

    scenario: dict
    mode: DialogMode
    strategy: str
    turns: list[Turn] = field(default_factory=list)


# --- Что произошло на ходе -------------------------------------------------------

def run_of(turn: Turn) -> rag_eval.Run:
    """Ход в виде `rag_eval.Run`: функции выдачи (`places_before`, `record_of`)
    работают по нему."""
    return rag_eval.Run(turn.reply)


def answer_of(turn: Turn) -> "rag_answer.AnswerCheck | None":
    return rag_eval.answer_of(run_of(turn))


def record_of(turn: Turn) -> agent.RagRecord | None:
    return rag_eval.record_of(run_of(turn))


def search_failed(turn: Turn) -> bool:
    """Поиск не удался: запись есть, а `ok` ложно (нет индекса, нет модели в кэше)."""
    reply = turn.reply
    return reply.ok and reply.rag is not None and not reply.rag.ok


def with_source(answer: "rag_answer.AnswerCheck") -> bool:
    """Ответ с источником любого вида (§2.7): номер выдержки из выдачи (в тексте
    или у цитаты) или верная ссылка на память — при любом виде ответа. Ответ,
    все номера которого вне выдачи, источника не имеет (правка по ревью: раньше
    его засчитывал вид «по выдержкам»)."""
    return bool(answer.sources or answer.memory_refs)


def kind_of(turn: Turn) -> str:
    """Вид ответа хода словами отчёта: вид проверки, у «не знаю» — по порогу или по
    содержанию; сбой поиска и сбой хода — отдельные виды."""
    if not turn.reply.ok:
        return "ход не удался"
    if search_failed(turn):
        return "сбой поиска"
    answer = answer_of(turn)
    if answer is None:
        return "без проверки"
    if answer.kind == rag_answer.KIND_IDK:
        return "не знаю (по порогу)" if answer.empty else "не знаю (по содержанию)"
    return answer.kind


def working_changes(update: str) -> str:
    """Рабочая часть строки `MemoryState.last_update` разбора: «+цель, ~уточнено»
    или «без изменений»; `""` — разбора не было или он не удался."""
    if not update:
        return ""
    if update == "без изменений":
        return update
    head = update.split(";")[0].strip()
    prefix = "рабочая:"
    changes = head[len(prefix):].strip() if head.startswith(prefix) else head
    return changes or "без изменений"


def memory_update(turn: Turn) -> str:
    """Что изменил разбор памяти на этом ходе (рабочая часть); сбой разбора и ход
    без разбора — своими словами."""
    call = turn.reply.memory_call
    if call is None:
        return "разбора не было"
    if not call.ok:
        return f"⚠️ разбор не удался: {call.error}"
    return working_changes(call.memory_update) or "без изменений"


def memory_keys(turn: Turn) -> list[str]:
    return [entry["key"] for entry in turn.memory]


def memory_checks(turn: Turn) -> list[tuple[str, bool]]:
    """Ожидаемые ключи памяти после хода и есть ли они: «ключи, а не значения»
    (§2.6) — верно ли записано, решает автор."""
    present = set(memory_keys(turn))
    return [(key, key in present) for key in turn.step["memory"]]


def goal_changes(dialog: Dialog) -> list[tuple[str, str]]:
    """«Цель» после первого хода и на каждом ходе, где её значение менялось (§5,
    п. 4): пары «номер хода, значение»; ходы, на которых значения нет, пропускаются."""
    changes: list[tuple[str, str]] = []
    previous = None
    for turn in dialog.turns:
        value = next((entry["value"] for entry in turn.memory if entry["key"] == "цель"), None)
        if value is not None and value != previous:
            changes.append((turn.step["id"], value))
        if value is not None:
            previous = value
    return changes


# --- Метрики ---------------------------------------------------------------------

def dialog_stats(dialog: Dialog) -> dict:
    """Счётчики сценария в режиме (§2.7) — по ходам, которые удались."""
    stats = {
        "turns": len(dialog.turns), "ok": 0,
        "по выдержкам": 0, "по памяти": 0, "не знаю · порог": 0, "не знаю · содержание": 0,
        "не знаю · со ссылками": 0, "без ссылок": 0, "сбой поиска": 0, "без проверки": 0,
        "with_source": 0, "empty": 0, "mem_ok": 0, "mem_missing": 0,
        "src_found": 0, "src_total": 0, "keys_found": 0, "keys_total": 0,
        "repaired": 0, "violated": [], "prompt": 0, "completion": 0,
        "c_main": 0.0, "c_rewrite": 0.0, "c_memory": 0.0, "t_model": 0.0, "t_rewrite": 0.0, "t_memory": 0.0,
        "t_search": 0.0,
    }
    for turn in dialog.turns:
        reply = turn.reply
        found, total = 0, 0
        for key, present in memory_checks(turn):
            total += 1
            found += int(present)
        stats["keys_found"] += found
        stats["keys_total"] += total
        if not reply.ok:
            continue
        stats["ok"] += 1
        stats["prompt"] += reply.prompt_tokens or 0
        stats["completion"] += reply.completion_tokens or 0
        stats["c_main"] += reply.cost_usd or 0.0
        stats["t_model"] += reply.elapsed
        if reply.rewrite_call is not None:
            stats["c_rewrite"] += reply.rewrite_call.cost_usd or 0.0
            stats["t_rewrite"] += reply.rewrite_call.elapsed
        if reply.memory_call is not None:
            stats["c_memory"] += reply.memory_call.cost_usd or 0.0
            stats["t_memory"] += reply.memory_call.elapsed
        record = reply.rag
        if record is not None and record.ok:
            stats["t_search"] += record.elapsed + record.rerank_load_s
            stats["empty"] += int(record.empty)
        if turn.step["sources"]:
            places = rag_eval.places_after(run_of(turn), turn.step)
            stats["src_total"] += len(places)
            stats["src_found"] += sum(1 for place in places if place is not None)
        if search_failed(turn):
            stats["сбой поиска"] += 1
            continue
        answer = answer_of(turn)
        if answer is None:
            stats["без проверки"] += 1
            continue
        stats["mem_ok"] += len(answer.memory_refs)
        stats["mem_missing"] += len(answer.memory_missing)
        stats["with_source"] += int(with_source(answer))
        if answer.kind == rag_answer.KIND_ANSWER:
            stats["по выдержкам"] += 1
        elif answer.kind == rag_answer.KIND_MEMORY:
            stats["по памяти"] += 1
        elif answer.kind == rag_answer.KIND_IDK:
            stats["не знаю · порог" if answer.empty else "не знаю · содержание"] += 1
            stats["не знаю · со ссылками"] += int(bool(answer.sources or answer.memory_refs))
        else:
            stats["без ссылок"] += 1
        if rag_eval.repair_rounds(run_of(turn)):
            stats["repaired"] += 1
        if not answer.ok:
            stats["violated"].append(turn.step["id"])
    return stats


def summary_table(scenario: dict, dialogs: list[Dialog]) -> list[str]:
    """«Сводка»: показатели §2.7 по сценарию, колонки — режимы."""
    stats = {dialog.mode.key: dialog_stats(dialog) for dialog in dialogs}

    def row(label: str, make) -> str:
        return f"| {label} | " + " | ".join(make(dialog, stats[dialog.mode.key]) for dialog in dialogs) + " |"

    def goal(dialog: Dialog, _st: dict) -> str:
        changes = goal_changes(dialog)
        if not changes:
            return "цели в памяти нет"
        first_id, first = changes[0]
        later = ", ".join(step_id for step_id, _ in changes[1:]) or "не менялась"
        return cell(f"после {first_id}: «{first}»; менялась: {later}")

    lines = [
        f"**Сценарий {scenario['id']} — {scenario['title']}** ({len(scenario['steps'])} сообщений)", "",
        "| показатель | " + " | ".join(dialog.mode.label for dialog in dialogs) + " |",
        "| --- | " + " | ".join("---" for _ in dialogs) + " |",
        row("ходов (удалось)", lambda d, st: f"{st['turns']} ({st['ok']})"),
        row("ответов **по выдержкам** (источник — правила; главная цифра)", lambda d, st: str(st["по выдержкам"])),
        row("ответов по памяти задачи", lambda d, st: str(st["по памяти"])),
        row(
            "«не знаю»: по порогу / по содержанию (из них со ссылками)",
            lambda d, st: f"{st['не знаю · порог']} / {st['не знаю · содержание']} ({st['не знаю · со ссылками']})",
        ),
        row("без ссылок", lambda d, st: str(st["без ссылок"])),
        row("сбой поиска", lambda d, st: str(st["сбой поиска"])),
        row(
            "с источником любого вида (в нём ответы по памяти и «не знаю» со ссылками)",
            lambda d, st: f"{st['with_source']} из {st['ok'] - st['сбой поиска']}",
        ),
        row("пустых выдач (ни один кусок не прошёл порог)", lambda d, st: str(st["empty"])),
        row("ссылок на память: верных / без записи в запросе", lambda d, st: f"{st['mem_ok']} / {st['mem_missing']}"),
        row(
            "ожидаемых источников в запросе (по ходам с источниками)",
            lambda d, st: f"{st['src_found']} из {st['src_total']}",
        ),
        row("ожидаемых ключей памяти после своего хода", lambda d, st: f"{st['keys_found']} из {st['keys_total']}"),
        row("цель", goal),
        row("повторов формата", lambda d, st: str(st["repaired"])),
        row("ответов с нарушениями в итоге", lambda d, st: ", ".join(st["violated"]) or "—"),
        row(
            "токены основного вызова: prompt / completion",
            lambda d, st: f"{num(st['prompt'])} / {num(st['completion'])}",
        ),
        row(
            "стоимость: основной / переписывание / разбор памяти",
            lambda d, st: f"{money(st['c_main'])} / {money(st['c_rewrite'])} / {money(st['c_memory'])}",
        ),
        row(
            "время, с: модель / переписывание / разбор памяти / поиск",
            lambda d, st: f"{st['t_model']:.1f} / {st['t_rewrite']:.1f} / {st['t_memory']:.1f} / {st['t_search']:.1f}",
        ),
    ]
    return lines + [""]


# --- Таблица ходов ---------------------------------------------------------------

def turn_cell(turn: Turn | None) -> str:
    """Ячейка режима в таблице ходов: запрос поиска, ожидаемые источники до →
    после, вид, источники, нарушения и повтор, ключи памяти после хода."""
    if turn is None:
        return "—"
    reply = turn.reply
    if not reply.ok:
        return "⚠️ ход не удался"
    lines: list[str] = []
    record = reply.rag
    if record is not None:
        lines.append("запрос: " + rag_eval.search_line(run_of(turn)))
        if turn.step["sources"]:
            after = found_text(rag_eval.places_after(run_of(turn), turn.step))
            if record.ok and record.rerank_ok:
                lines.append(f"источники до → после: {found_text(rag_eval.places_before(run_of(turn), turn.step))} → {after}")
            else:
                lines.append(f"источники: {after}")
        if not record.ok:
            lines.append(f"⚠️ поиск не удался: {record.error}")
        else:
            lines.append(f"выдержек: {len(record.hits)}" + (" (пусто по порогу)" if record.empty else ""))
    lines.append("вид: " + kind_of(turn))
    answer = answer_of(turn)
    if answer is not None:
        refs = [f"[{n}]" for n in answer.sources] + [f"[{key}]" for key in answer.memory_refs]
        refs += [f"[{key}] ⚠️" for key in answer.memory_missing]
        lines.append("источники ответа: " + (", ".join(refs) or "—"))
        first = record.answer_first if record is not None else None
        if first is not None:
            lines.append(f"нарушения: {len(first.violations)} → {len(answer.violations)} · повтор: был")
        elif answer.violations:
            lines.append(f"нарушения: {len(answer.violations)}")
        else:
            lines.append("нарушений нет")
    marks = [("✓ " if present else "✗ ") + key for key, present in memory_checks(turn)]
    lines.append("память: " + (", ".join(marks) if marks else "ожидаемых ключей нет"))
    lines.append("разбор: " + memory_update(turn))
    return "<br>".join(cell(line) for line in lines)


def steps_table(scenario: dict, dialogs: list[Dialog]) -> list[str]:
    """«Таблица ходов» сценария: № · сообщение · режимы · две колонки автора."""
    out = [
        "| № | сообщение | " + " | ".join(dialog.mode.label for dialog in dialogs)
        + " | цель и условия игрока учтены (автор) | смысл совпадает с источниками (автор) |",
        "| --- | --- | " + " | ".join("---" for _ in dialogs) + " | --- | --- |",
    ]
    for index, step in enumerate(scenario["steps"]):
        cells = []
        for dialog in dialogs:
            turn = dialog.turns[index] if index < len(dialog.turns) else None
            cells.append(turn_cell(turn))
        out.append(f"| {step['id']} | {cell(step['message'])} | " + " | ".join(cells) + " | | |")
    return out + [""]


# --- Память задачи по ходам ------------------------------------------------------

def sections_text(entries: list[dict]) -> list[str]:
    """Память задачи после хода — строками по пунктам задания, в порядке
    `TASK_MEMORY_SECTIONS`; записи с другим разделом — отдельной группой."""
    out: list[str] = []
    known = [*presets.TASK_MEMORY_SECTIONS]
    extra = [e["section"] for e in entries if e["section"] not in known]
    for section in [*known, *dict.fromkeys(extra)]:
        lines = [f"- **{e['key']}:** {cell(e['value'])}" for e in entries if e["section"] == section]
        out.append(f"_{section}:_" + (" —" if not lines else ""))
        if lines:
            out += ["", *lines]
        out.append("")
    return out


def memory_tables(scenario: dict, dialogs: list[Dialog]) -> list[str]:
    """«Память задачи по ходам» (§5, п. 4): что изменил разбор на каждом ходе и
    память целиком после последнего хода; значение «цель» — после первого хода
    и на каждом ходе, где оно менялось."""
    out: list[str] = []
    for dialog in dialogs:
        out += [f"#### Сценарий {scenario['id']} — {dialog.mode.label}", ""]
        if not dialog.turns:
            out += ["Ходов нет.", ""]
            continue
        goals = dict(goal_changes(dialog))
        out += ["| ход | что изменил разбор памяти | значение «цель» |", "| --- | --- | --- |"]
        for turn in dialog.turns:
            value = goals.get(turn.step["id"])
            out.append(
                f"| {turn.step['id']} | {cell(memory_update(turn))} | {cell('«' + value + '»') if value else '—'} |"
            )
        out += ["", f"**Память задачи после {dialog.turns[-1].step['id']}:**", ""]
        entries = dialog.turns[-1].memory
        out += sections_text(entries) if entries else ["пусто"]
        out.append("")
    return out


# --- По ходам --------------------------------------------------------------------

def claims_table(answer: "rag_answer.AnswerCheck", record: agent.RagRecord) -> list[str]:
    """«Утверждения и источники»: по номеру выдержки — предложения ответа с ним и
    цитаты, по ключу памяти — предложения и значение записи на момент ответа.
    Подтверждает ли источник утверждение — смотрит автор."""
    claims = dict(answer.claims)
    numbers = sorted({*answer.refs, *(item.number for item in answer.quotes)})
    entries = {entry.key: entry for entry in record.memory_entries}
    if not numbers and not answer.memory_claims:
        return []
    out = [
        "| № или ключ | источник | утверждения ответа | цитаты или значение записи | проверка |",
        "| --- | --- | --- | --- | --- |",
    ]
    for number in numbers:
        hit = record.hits[number - 1] if 1 <= number <= len(record.hits) else None
        section = cell(hit["section"] + (f" · часть {hit['part']}" if hit.get("part") else "")) if hit else "—"
        sentences = "<br>".join(cell(text) for text in claims.get(number, ())) or "—"
        own = [item for item in answer.quotes if item.number == number]
        texts = "<br>".join(cell(f"«{item.text}»") for item in own) or "—"
        marks = "<br>".join(cell(rag_answer.quote_mark(item) or "✓ дословно") for item in own) or "—"
        out.append(f"| [{number}] | {section} | {sentences} | {texts} | {marks} |")
    for key, sentences in answer.memory_claims:
        entry = entries.get(key)
        text = "<br>".join(cell(item) for item in sentences) or "—"
        if entry is None:
            out.append(f"| [{key}] | — | {text} | — | ⚠️ записи с таким ключом в запросе не было |")
        else:
            out.append(
                f"| [{key}] | {cell(entry.label)} | {text} | "
                f"{cell('«' + rag_answer.memory_value_text(entry.value) + '»')} | ✓ запись была в запросе |"
            )
    return out


def turn_block(turn: Turn, mode: DialogMode) -> list[str]:
    """Ход в режиме: запрос поиска, был ли во входе переписывания абзац памяти
    задачи, ответ дословно (как ушёл игроку), «Утверждения и источники» и, при
    повторе, первая попытка."""
    reply = turn.reply
    out = [f"**{mode.label}:**", ""]
    if not reply.ok:
        return out + [f"⚠️ ход не удался: {reply.error}", ""]
    record = reply.rag
    if record is not None:
        memory_part = (
            f"память задачи во входе: да ({agent.task_memory_records(record.rewrite_memory)})"
            if record.rewrite_memory
            else "памяти задачи во входе не было"
        )
        out += [
            f"- запрос поиска: {rag_eval.search_line(run_of(turn))} · {memory_part}",
            f"- инструкция памяти в запросе: {'да' if record.memory_note else 'нет'}"
            f" · записей памяти в запросе: {len(record.memory_entries)}",
        ]
        if record.ok:
            out.append(f"- выдача: {rag_eval.hits_list(record)}")
        else:
            out.append(f"- ⚠️ поиск не удался: {record.error} — запрос ушёл без выдержек")
    out += ["", quote(reply.text), "", f"_{rag_eval.metrics_line(run_of(turn), rag_eval.MODE_BY_KEY['полный'])}_", ""]
    answer = answer_of(turn)
    if answer is not None and record is not None:
        table = claims_table(answer, record)
        if table:
            out += ["_Утверждения и источники (смысл сверяет автор):_", "", *table, ""]
        if answer.violations:
            out += ["_Нарушения итогового ответа:_", "", *(f"- {violation}" for violation in answer.violations), ""]
        if record.repair_note:
            out += [f"_Повтор формата: {record.repair_note}_", ""]
        if record.answer_first is not None:
            out += ["_Повтор формата: первая попытка — ответ модели как пришёл:_", "", quote(record.answer_first.raw), ""]
            out += ["_Нарушения первой попытки:_", "", *(f"- {violation}" for violation in record.answer_first.violations), ""]
    return out


def step_sections(scenario: dict, dialogs: list[Dialog]) -> list[str]:
    out: list[str] = []
    for index, step in enumerate(scenario["steps"]):
        out += [
            f"#### {step['id']}", "",
            f"**Сообщение:** {step['message']}", "",
            f"**Что проверяет:** {step['checks']}", "",
            f"**Ожидание:** {step['expected']}", "",
        ]
        if step["sources"]:
            out += ["**Ожидаемые источники:**", ""]
            for source in step["sources"]:
                out.append(f"- `{source[0]}` · «{source[1]}» · стр. {source[2]}")
                for dialog in dialogs:
                    if index < len(dialog.turns):
                        out.append(f"  - {dialog.mode.label}: {rag_eval.source_status(source, run_of(dialog.turns[index]))}")
        else:
            out.append("**Ожидаемые источники:** — (источников правил не ждём)")
        if step["memory"]:
            out += ["", "**Ожидаемые ключи памяти после хода:** " + ", ".join(step["memory"])]
        out.append("")
        for dialog in dialogs:
            if index < len(dialog.turns):
                out += turn_block(dialog.turns[index], dialog.mode)
    return out


# --- Отчёт -----------------------------------------------------------------------

def build_report(
    dialogs: list[Dialog], scenarios: list[dict], config: agent.AgentConfig, info: dict, modes: list[DialogMode],
    started: datetime, interrupted: str | None,
) -> str:
    out: list[str] = ["# Прогон длинных сценариев: RAG и память задачи (день 25)", ""]
    if interrupted is not None:
        done = sum(len(dialog.turns) for dialog in dialogs)
        total = sum(len(scenario["steps"]) for scenario in scenarios) * len(modes)
        out += [f"**Прервано: {interrupted}** — в отчёте пройденные ходы ({done} из {total}).", ""]
    device = (
        f"{info.get('rerank_device', 'н/д')}, {info.get('rerank_dtype', 'н/д')}"
        if info.get("rerank_loaded") else "н/д (не загружался)"
    )
    strategies = ", ".join(dict.fromkeys(f"«{dialog.strategy}»" for dialog in dialogs))
    out += [
        "## 1. Условия", "",
        f"- Дата и время: {started:%Y-%m-%d %H:%M:%S}",
        f"- Пресет: «{config.name}» — модель `{config.model}`, thinking {'включён' if config.thinking else 'выключен'}",
        f"- Индекс: `{info['path']}` · собран {info['built_at']} · модель эмбеддингов `{info['model']}` · "
        f"нарезка `{info['strategy']}` · язык `{info['lang']}` · "
        f"кусков нарезки и языка: {info['chunks']} (всего в индексе {info['total']})",
        f"- Поиск (дни 22-23): этап 1 — {presets.RAG_CANDIDATES} ближайших, этап 2 — реранкер `{info['rerank_model']}` "
        f"({device}), порог {threshold_text(presets.RAG_RERANK_THRESHOLD)}, в запрос — не больше {presets.RAG_TOP_K}; "
        "переписывание запроса включено",
        f"- Стратегия контекста: {strategies}",
        "- Агенты: на каждый сценарий и режим — свежий; пресет, стратегия, модель памяти (`presets.make_memory()`), "
        "поиск, формат ответа и повтор (день 24), инструкция памяти; без хранилища, долговременной памяти, профиля, "
        "инвариантов, задачи и инструментов MCP; сессии на диск не пишутся",
        "- Режимы: " + "; ".join(
            f"`{mode.key}` — {mode.label}" + (
                "" if mode.memory
                else " (слой «рабочая» снят в «Слоях памяти в запросе»: разбор памяти идёт, но память задачи не уходит "
                     "ни в запрос, ни в переписывание, ни в инструкцию)"
            )
            for mode in modes
        ),
        f"- Карта памяти — `presets.MEMORY_MAP` (рабочая часть: {sum(1 for slot in presets.MEMORY_MAP if slot.layer == memory.LAYER_WORKING)} "
        "ключей, разделы — пункты задания), промпт разбора — `presets.MEMORY_PROMPT`, промпт переписывания — "
        "`presets.RAG_REWRITE_PROMPT`, инструкция к выдержкам — `presets.RAG_INSTRUCTION`, инструкция памяти — "
        "`presets.RAG_MEMORY_INSTRUCTION`",
        "- В отчёте нет оценок: «найден» относится только к выдаче, «верная ссылка» — к тому, была ли запись в запросе; "
        "что такое потерянная цель и совпадает ли смысл ответа с источниками, решает автор",
        "",
    ]

    out += ["## 2. Сводка", ""]
    for scenario in scenarios:
        mine = [dialog for dialog in dialogs if dialog.scenario["id"] == scenario["id"]]
        if mine:
            out += summary_table(scenario, mine)

    out += ["## 3. По сценарию — таблица ходов", ""]
    for scenario in scenarios:
        mine = [dialog for dialog in dialogs if dialog.scenario["id"] == scenario["id"]]
        if not mine:
            continue
        out += [f"### Сценарий {scenario['id']} — {scenario['title']}", "", f"_{scenario['checks']}_", ""]
        out += steps_table(scenario, mine)

    out += ["## 4. Память задачи по ходам", ""]
    for scenario in scenarios:
        mine = [dialog for dialog in dialogs if dialog.scenario["id"] == scenario["id"]]
        if mine:
            out += memory_tables(scenario, mine)

    out += ["## 5. По ходам", ""]
    for scenario in scenarios:
        mine = [dialog for dialog in dialogs if dialog.scenario["id"] == scenario["id"]]
        if mine:
            out += [f"### Сценарий {scenario['id']} — {scenario['title']}", "", *step_sections(scenario, mine)]
    return "\n".join(out).rstrip() + "\n"


# --- Прогон ----------------------------------------------------------------------

def make_agent(
    config: agent.AgentConfig, name: str, index: rag_search.RulesIndex, strategy: str, mode: DialogMode,
) -> agent.Agent:
    """Свежий агент на сценарий и режим (§5)."""
    fresh = agent.Agent(
        config, session_id=name, strategies=presets.make_strategies(), strategy=strategy,
        memory=presets.make_memory(), retriever=index, rag_instruction=presets.RAG_INSTRUCTION,
        rag_rewrite_prompt=presets.RAG_REWRITE_PROMPT, rag_rewrite_max_tokens=presets.RAG_REWRITE_MAX_TOKENS,
        rag_empty_instruction=presets.RAG_EMPTY_INSTRUCTION,
        rag_answer_format=presets.RAG_ANSWER_FORMAT, rag_repair_instruction=presets.RAG_REPAIR_INSTRUCTION,
        rag_memory_instruction=presets.RAG_MEMORY_INSTRUCTION,
    )
    if not mode.memory:
        # Долговременной памяти у агента нет, поэтому снимается рабочий слой — и
        # это единственный переключатель памяти задачи (§1.1).
        fresh.set_request_layers([memory.LAYER_LONG_TERM])
    return fresh


def log_turn(turn: Turn, mode: DialogMode) -> None:
    reply = turn.reply
    line = f"{turn.step['id']} · {mode.label}: "
    if not reply.ok:
        logger.info("%s⚠️ ход не удался: %s", line, reply.error)
        return
    record = reply.rag
    if record is not None:
        query = record.query or record.question
        same = " ".join(query.split()) == " ".join(record.question.split())
        line += f"запрос «{query}»{' (без изменений)' if same else ''}"
        line += "".join(f" + «{other}»" for other in record.queries[1:])
        if record.ok:
            line += f" · выдержек {len(record.hits)}"
            if turn.step["sources"]:
                after = found_text(rag_eval.places_after(run_of(turn), turn.step))
                if record.rerank_ok:
                    line += f" · до → после {found_text(rag_eval.places_before(run_of(turn), turn.step))} → {after}"
                else:
                    line += f" · источники {after}"
            if record.empty:
                line += " · пусто по порогу"
        else:
            line += f" · ⚠️ поиск не удался: {record.error}"
    line += f" · ответ: {kind_of(turn)}"
    answer = answer_of(turn)
    if answer is not None:
        if answer.sources:
            line += " · источники [" + ", ".join(str(n) for n in answer.sources) + "]"
        if answer.memory_refs:
            line += " · память [" + ", ".join(answer.memory_refs) + "]"
        if answer.memory_missing:
            line += " · ⚠️ ключи без записи [" + ", ".join(answer.memory_missing) + "]"
        if rag_eval.repair_rounds(run_of(turn)):
            line += " · повтор был"
        if answer.violations:
            line += f" · ⚠️ нарушений {len(answer.violations)}"
    line += f" · память задачи: {memory_update(turn)}"
    logger.info("%s", line)


def run_dialog(
    scenario: dict, mode: DialogMode, strategy: str, config: agent.AgentConfig, index: rag_search.RulesIndex,
    dialog: Dialog,
) -> None:
    """Все ходы сценария одним агентом; ходы дописываются в `dialog` по мере
    прохождения — при Ctrl+C отчёт получает пройденное."""
    fresh = make_agent(config, f"dialog-{scenario['id']}-{mode.key}", index, strategy, mode)
    for step in scenario["steps"]:
        reply = fresh.ask(step["message"])
        state = fresh.debug_state()["task_memory"] or {"entries": []}
        turn = Turn(step, reply, [dict(entry) for entry in state["entries"]])
        dialog.turns.append(turn)
        log_turn(turn, mode)


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="./run.sh rag-dialog",
        description=(
            "Прогон двух длинных сценариев (А — 14 сообщений, Б — 13) в режимах «с памятью задачи» и «без памяти "
            "задачи»: свежий агент на сценарий и режим, стратегия «Скользящее окно»; отчёт Markdown — виды ответов, "
            "источники (выдержки и записи памяти), память задачи по ходам и две колонки автора. Нужен ключ DeepSeek "
            "(src/.env), собранный индекс (./run.sh index) и реранкер в кэше (./run.sh index --probe)."
        ),
    )
    parser.add_argument(
        "--scenarios", metavar="А,Б", default=",".join(s["id"] for s in presets.RAG_DIALOG_SCENARIOS),
        help="сценарии через запятую (по умолчанию оба; латинские A и B — за А и Б)",
    )
    parser.add_argument(
        "--modes", metavar="память,без", default=",".join(DEFAULT_MODES),
        help="режимы через запятую из: " + ", ".join(mode.key for mode in MODES) + " (по умолчанию оба)",
    )
    parser.add_argument(
        "--strategy", metavar="ИМЯ", default=None,
        help="стратегия контекста вместо стратегии сценария (имя из presets.STRATEGIES)",
    )
    parser.add_argument("--preset", default=presets.RAG_EVAL_PRESET, metavar="ИМЯ",
                        help=f"пресет агента (по умолчанию «{presets.RAG_EVAL_PRESET}»)")
    parser.add_argument("--db", type=Path, default=presets.RAG_INDEX_DB, metavar="ПУТЬ",
                        help=f"файл индекса (по умолчанию {rag_search.shown(presets.RAG_INDEX_DB)})")
    parser.add_argument("--out", type=Path, default=None, metavar="ФАЙЛ.md",
                        help=f"куда писать отчёт (по умолчанию {rag_search.shown(presets.RAG_DIALOG_DIR)}/<дата-время>.md)")
    return parser.parse_args(argv)


def main() -> None:
    args = _args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[RAG-диалог] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    config = presets.PRESETS.get(args.preset)
    if config is None:
        logger.error("нет пресета «%s»; есть: %s", args.preset, ", ".join(f"«{name}»" for name in presets.PRESETS))
        sys.exit(2)

    by_id = {scenario["id"]: scenario for scenario in presets.RAG_DIALOG_SCENARIOS}
    wanted = [item.strip().translate(_LATIN_ID) for item in args.scenarios.split(",") if item.strip()]
    unknown = [item for item in wanted if item not in by_id]
    if unknown or not wanted:
        logger.error("нет сценариев: %s; есть: %s", ", ".join(unknown) or "(пусто)", ", ".join(by_id))
        sys.exit(2)
    scenarios = [by_id[item] for item in dict.fromkeys(wanted)]

    keys = [item.strip() for item in args.modes.split(",") if item.strip()]
    unknown_modes = [key for key in keys if key not in MODE_BY_KEY]
    if unknown_modes or not keys:
        logger.error("нет режимов: %s; есть: %s", ", ".join(unknown_modes) or "(пусто)",
                     ", ".join(mode.key for mode in MODES))
        sys.exit(2)
    modes = [MODE_BY_KEY[key] for key in dict.fromkeys(keys)]

    if args.strategy is not None and args.strategy not in presets.make_strategies():
        logger.error("нет стратегии «%s»; есть: %s", args.strategy,
                     ", ".join(f"«{name}»" for name in presets.make_strategies()))
        sys.exit(2)

    index = rag_search.RulesIndex(
        args.db.expanduser(), presets.RAG_SEARCH_STRATEGY, presets.RAG_SEARCH_LANG, presets.RAG_TOP_K,
        candidates_k=presets.RAG_CANDIDATES, rerank_model=presets.RAG_RERANK_MODEL,
        threshold=presets.RAG_RERANK_THRESHOLD, rerank_max_length=presets.RAG_RERANK_MAX_LENGTH,
        rerank_batch=presets.RAG_RERANK_BATCH, allow_download=False, expected_model=presets.RAG_EMBED_MODEL,
    )
    info = index.info()
    if not info["ok"]:
        # Сценарии без индекса бессмысленны: отказ до первого вызова модели.
        logger.error("%s — прогон отменён, модель не вызывалась", info["error"])
        sys.exit(2)
    if not info["rerank_cached"]:
        logger.error(
            "реранкера %s нет в кэше Hugging Face — ./run.sh index --probe скачает ≈1,5 ГБ; прогон отменён, "
            "модель не вызывалась", presets.RAG_RERANK_MODEL,
        )
        sys.exit(2)
    steps = sum(len(scenario["steps"]) for scenario in scenarios) * len(modes)
    logger.info(
        "индекс %s · собран %s · %s/%s: %d кусков из %d · этап 1: %d → этап 2: %s ≥ %s → ≤%d · пресет «%s» (%s) · "
        "сценарии: %s · режимы: %s · ходов: %d",
        info["path"], info["built_at"], info["strategy"], info["lang"], info["chunks"], info["total"],
        presets.RAG_CANDIDATES, presets.RAG_RERANK_MODEL, threshold_text(presets.RAG_RERANK_THRESHOLD),
        presets.RAG_TOP_K, config.name, config.model, ", ".join(scenario["id"] for scenario in scenarios),
        ", ".join(mode.key for mode in modes), steps,
    )

    started = datetime.now()
    out_path = (args.out.expanduser() if args.out is not None
                else presets.RAG_DIALOG_DIR / f"{started:%Y-%m-%d-%H%M%S}.md")
    dialogs: list[Dialog] = []
    interrupted: str | None = None
    code = 0
    try:
        for scenario in scenarios:
            strategy = args.strategy or scenario["strategy"]
            for mode in modes:
                dialog = Dialog(scenario, mode, strategy)
                dialogs.append(dialog)
                logger.info("сценарий %s — %s · %s · стратегия «%s»", scenario["id"], scenario["title"],
                            mode.label, strategy)
                run_dialog(scenario, mode, strategy, config, index, dialog)
    except KeyboardInterrupt:
        interrupted, code = passed(dialogs), 130
        logger.error("прервано %s — пишу отчёт по пройденному", interrupted)
    except Exception as exc:  # noqa: BLE001 — оплаченные ходы важнее трейсбека (правка по ревью)
        interrupted, code = f"{passed(dialogs)}: сбой — {exc}", 1
        logger.exception("сбой %s — пишу отчёт по пройденному", passed(dialogs))

    # Условия индекса — последние известные: если к концу прогона индекс не
    # читается (пересобран, удалён), в отчёт идут условия на старте.
    report = build_report(dialogs, scenarios, config, {**info, **index.info()}, modes, started, interrupted)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("отчёт: %s (%d ходов из %d)", rag_search.shown(out_path), sum(len(d.turns) for d in dialogs), steps)
    print()
    print("Сводка" + report.partition("## 2. Сводка")[2].partition("## 3. ")[0].rstrip())
    sys.exit(code)


def passed(dialogs: list[Dialog]) -> str:
    """Докуда дошёл прогон: «после А5 (с памятью задачи)» / «до первого хода»."""
    last = next((dialog for dialog in reversed(dialogs) if dialog.turns), None)
    return f"после {last.turns[-1].step['id']} ({last.mode.label})" if last else "до первого хода"


if __name__ == "__main__":
    main()
