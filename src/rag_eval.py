# TooManyRules — прогон контрольных вопросов: без RAG и с RAG (день 22, неделя 5).
#
# **Отдельная программа**, как `rag_index.py`: её запускает человек
# (`./run.sh rag-eval`), и работает она до конца, а не живёт. Приложение её не
# импортирует. Вопросы, пресет и параметры поиска берёт из `presets.py`, к модели
# ходит только через `Agent.ask()` (LLM API вызывается только в `agent.py`),
# выдержки ищет тот же `rag_search.RulesIndex`, что и приложение.
#
# Агент **голый** (спецификация дня 22, §2.7): пресет и его системный промпт, без
# хранилища, стратегий кроме «Всей истории», памяти, профиля, инвариантов,
# задачи и инструментов — режимы отличаются только последним сообщением, это
# чистое A/B. На каждый вопрос — два свежих агента одного конфига: у первого
# переключатель RAG снят, у второго включён; по очереди, без параллельности.
# Сессий на диск программа не пишет — данные автора не трогает.
#
# Отчёт — Markdown, готовый к переносу в документ сравнения. **Оценок в нём нет**:
# «найден» относится только к выдаче (`rag_search.sources_found()`), а совпадение
# ответа с ожиданием, верность ссылок и выдумки оценивает автор.
#
# Логи — stdout: строки агентов (как в приложении) и свои с префиксом
# `[RAG-прогон]`.

import argparse
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import agent
import presets
import rag_search

logger = logging.getLogger("rag_eval")

# Латинские K/k в номере вопроса принимаем за кириллические: на клавиатуре
# «К1» легко набрать не той раскладкой.
_LATIN_K = str.maketrans({"K": "К", "k": "К", "к": "К"})


@dataclass
class Run:
    """Один ход одного режима."""

    reply: agent.AgentReply


@dataclass
class Result:
    question: dict
    plain: Run
    rag: Run
    places: list[int | None]


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


def source_line(source: tuple[str, str, int], place: int | None) -> str:
    doc, section, page = source
    where = f"место {place}" if place is not None else "не в выдаче"
    return f"`{doc}` · «{section}» · стр. {page} — {where}"


def pages(hit: dict) -> str:
    first, last = hit["page_from"], hit["page_to"]
    return str(first) if first == last else f"{first}–{last}"


def cell(text: str) -> str:
    return " ".join(str(text).split()).replace("|", "\\|")


def quote(text: str) -> str:
    return "\n".join(f"> {line}" if line.strip() else ">" for line in text.strip().splitlines()) or ">"


def metrics_line(run: Run, with_rag: bool) -> str:
    reply = run.reply
    parts = [
        f"время модели {reply.elapsed:.2f} с",
        f"prompt {num(reply.prompt_tokens)} · completion {num(reply.completion_tokens)}",
        f"стоимость {money(reply.cost_usd)}",
        f"finish_reason `{reply.finish_reason or 'н/д'}`",
    ]
    if with_rag and reply.rag is not None:
        rag = reply.rag
        found = f"поиск {rag.elapsed:.2f} с"
        if rag.load_s:
            found += f" (загрузка модели {rag.load_s:.1f} с)"
        parts.append(found)
        parts.append(f"выдержки ≈{num(rag.tokens)} ток. (оценка)")
    return " · ".join(parts)


def answer_block(run: Run, with_rag: bool) -> list[str]:
    reply = run.reply
    if not reply.ok:
        return [f"⚠️ ход не удался: {reply.error}", ""]
    return [quote(reply.text), "", f"_{metrics_line(run, with_rag)}_", ""]


def build_report(
    results: list[Result], config: agent.AgentConfig, info: dict,
    started: datetime, interrupted_after: str | None, total_questions: int,
) -> str:
    out: list[str] = ["# Прогон контрольных вопросов дня 22: без RAG и с RAG", ""]
    if interrupted_after is not None:
        out += [f"**Прервано после {interrupted_after}** — в отчёте пройденные вопросы ({len(results)} из {total_questions}).", ""]

    out += [
        "## 1. Условия", "",
        f"- Дата и время: {started:%Y-%m-%d %H:%M:%S}",
        f"- Пресет: «{config.name}» — модель `{config.model}`, thinking {'включён' if config.thinking else 'выключен'}",
        f"- Индекс: `{info['path']}` · собран {info['built_at']} · модель эмбеддингов `{info['model']}` · "
        f"нарезка `{info['strategy']}` · язык `{info['lang']}` · top-{info['top_k']} · "
        f"кусков нарезки и языка: {info['chunks']} (всего в индексе {info['total']})",
        "- Агенты: голые — пресет и системный промпт, без хранилища, стратегий, памяти, профиля, инвариантов, "
        "задачи и инструментов MCP; на каждый вопрос — два свежих агента, у одного переключатель RAG снят",
        "- Инструкция к выдержкам — `presets.RAG_INSTRUCTION`; близость в запрос не уходит",
        "- В отчёте нет оценок: «найден» относится только к выдаче, ответы оценивает автор",
        "",
    ]

    out += [
        "## 2. Сводная таблица", "",
        "| № | ожидаемые источники в выдаче | prompt_tokens без → с | completion_tokens без → с | время модели, с без → с | стоимость без → с |",
        "| --- | --- | --- | --- | --- | --- |",
    ]
    sums = {"pp": 0, "pr": 0, "cp": 0, "cr": 0, "tp": 0.0, "tr": 0.0, "kp": 0.0, "kr": 0.0}
    for result in results:
        plain, rag = result.plain.reply, result.rag.reply
        out.append(
            f"| {result.question['id']} | {found_text(result.places)} | "
            f"{num(plain.prompt_tokens)} → {num(rag.prompt_tokens)} | "
            f"{num(plain.completion_tokens)} → {num(rag.completion_tokens)} | "
            f"{plain.elapsed:.2f} → {rag.elapsed:.2f} | {money(plain.cost_usd)} → {money(rag.cost_usd)} |"
        )
        if plain.ok and rag.ok:
            sums["pp"] += plain.prompt_tokens or 0
            sums["pr"] += rag.prompt_tokens or 0
            sums["cp"] += plain.completion_tokens or 0
            sums["cr"] += rag.completion_tokens or 0
            sums["tp"] += plain.elapsed
            sums["tr"] += rag.elapsed
            sums["kp"] += plain.cost_usd or 0.0
            sums["kr"] += rag.cost_usd or 0.0
    out.append(
        f"| **итого** | | {num(sums['pp'])} → {num(sums['pr'])} | {num(sums['cp'])} → {num(sums['cr'])} | "
        f"{sums['tp']:.2f} → {sums['tr']:.2f} | {money(sums['kp'])} → {money(sums['kr'])} |"
    )
    out += ["", "Итоги — по вопросам, где удались оба хода. Время — модель без поиска: поиск и загрузка модели эмбеддингов "
            "указаны у каждого ответа с RAG.", ""]

    out += ["## 3. По вопросам", ""]
    for result in results:
        question = result.question
        rag_record = result.rag.reply.rag
        out += [
            f"### {question['id']}", "",
            f"**Вопрос:** {question['question']}", "",
            f"**Что проверяет:** {question['checks']}", "",
            f"**Ожидание:** {question['expected']}", "",
        ]
        if question["sources"]:
            out.append("**Ожидаемые источники:**")
            out.append("")
            out += [f"- {source_line(src, place)}" for src, place in zip(question["sources"], result.places, strict=True)]
        else:
            out.append("**Ожидаемые источники:** — (вопрос вне корпуса)")
        out.append("")
        if rag_record is None or not rag_record.ok:
            reason = rag_record.error if rag_record is not None else "запись поиска не пришла"
            out += [f"⚠️ **Поиск не удался:** {reason} — запрос ушёл без выдержек.", ""]
        else:
            out += [
                f"**Выдача** (top-{len(rag_record.hits)} из {rag_record.total}):", "",
                "| место | близость | документ | раздел | часть | страницы | токенов |",
                "| --- | --- | --- | --- | --- | --- | --- |",
            ]
            for place, hit in enumerate(rag_record.hits, 1):
                out.append(
                    f"| {place} | {hit['score']:.3f} | {cell(hit['doc'])} | {cell(hit['section'])} | "
                    f"{cell(hit.get('part') or '—')} | {pages(hit)} | {hit['tokens']} |"
                )
            out.append("")
        out += ["**Ответ без RAG:**", "", *answer_block(result.plain, False)]
        out += ["**Ответ с RAG:**", "", *answer_block(result.rag, True)]
    return "\n".join(out).rstrip() + "\n"


def make_agent(config: agent.AgentConfig, name: str, index: rag_search.RulesIndex, rag: bool) -> agent.Agent:
    bare = agent.Agent(config, session_id=name, retriever=index, rag_instruction=presets.RAG_INSTRUCTION)
    if not rag:
        bare.set_rag_in_request(False)
    return bare


def run_question(
    question: dict, config: agent.AgentConfig, index: rag_search.RulesIndex,
) -> Result:
    runs: dict[bool, Run] = {}
    for rag in (False, True):
        mode = "с RAG" if rag else "без RAG"
        bare = make_agent(config, f"eval-{question['id']}-{'с' if rag else 'без'}", index, rag)
        reply = bare.ask(question["question"])
        runs[rag] = Run(reply)
        line = f"{question['id']} {mode}: "
        if reply.ok:
            line += (
                f"{reply.elapsed:.2f} с · prompt {num(reply.prompt_tokens)} / completion "
                f"{num(reply.completion_tokens)} · {money(reply.cost_usd)}"
            )
        else:
            line += f"⚠️ ход не удался: {reply.error}"
        if rag and reply.rag is not None:
            record = reply.rag
            if record.ok:
                places = rag_search.sources_found(record.hits, question["sources"])
                line += (
                    f" · поиск {record.elapsed:.2f} с"
                    f"{f' (загрузка модели {record.load_s:.1f} с)' if record.load_s else ''}"
                    f" · выдержки ≈{num(record.tokens)} ток. · ожидаемые источники в выдаче: {found_text(places)}"
                )
            else:
                line += f" · ⚠️ поиск: {record.error}"
        logger.info("%s", line)
    record = runs[True].reply.rag
    places = (
        rag_search.sources_found(record.hits, question["sources"])
        if record is not None and record.ok else [None] * len(question["sources"])
    )
    return Result(question, runs[False], runs[True], places)


def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="./run.sh rag-eval",
        description=(
            "Прогон контрольных вопросов дня 22: на каждый вопрос два свежих голых агента — без RAG и "
            "с RAG; отчёт Markdown. Нужен ключ DeepSeek (src/.env) и собранный индекс (./run.sh index)."
        ),
    )
    parser.add_argument("--only", metavar="К1,К8", help="часть вопросов (номера из RAG_CONTROL_QUESTIONS)")
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

    questions = list(presets.RAG_CONTROL_QUESTIONS)
    if args.only:
        wanted = [item.strip().translate(_LATIN_K) for item in args.only.split(",") if item.strip()]
        known = {q["id"] for q in questions}
        unknown = [item for item in wanted if item not in known]
        if unknown:
            logger.error("нет вопросов: %s; есть: %s", ", ".join(unknown), ", ".join(q["id"] for q in questions))
            sys.exit(2)
        questions = [q for q in questions if q["id"] in wanted]

    index = rag_search.RulesIndex(
        args.db.expanduser(), presets.RAG_SEARCH_STRATEGY, presets.RAG_SEARCH_LANG, presets.RAG_TOP_K,
        allow_download=False, expected_model=presets.RAG_EMBED_MODEL,
    )
    info = index.info()
    if not info["ok"]:
        # Сравнение без индекса бессмысленно: отказ до первого вызова модели.
        logger.error("%s — прогон отменён, модель не вызывалась", info["error"])
        sys.exit(2)
    logger.info(
        "индекс %s · собран %s · %s/%s: %d кусков из %d · top-%d · пресет «%s» (%s) · вопросов: %d",
        info["path"], info["built_at"], info["strategy"], info["lang"], info["chunks"], info["total"],
        info["top_k"], config.name, config.model, len(questions),
    )

    started = datetime.now()
    out_path = (args.out.expanduser() if args.out is not None
                else presets.RAG_EVAL_DIR / f"{started:%Y-%m-%d-%H%M%S}.md")
    results: list[Result] = []
    interrupted_after: str | None = None
    try:
        for question in questions:
            results.append(run_question(question, config, index))
    except KeyboardInterrupt:
        interrupted_after = results[-1].question["id"] if results else "старта"
        logger.error("прервано после %s — пишу отчёт по пройденным вопросам", interrupted_after)

    report = build_report(results, config, info, started, interrupted_after, len(questions))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("отчёт: %s (%d из %d вопросов)", rag_search.shown(out_path), len(results), len(questions))
    print()
    print("Сводная таблица" + report.partition("## 2. Сводная таблица")[2].partition("## 3. ")[0].rstrip())
    sys.exit(130 if interrupted_after is not None else 0)


if __name__ == "__main__":
    main()
