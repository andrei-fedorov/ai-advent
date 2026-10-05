# TooManyRules — три запроса к локальной модели (день 26, неделя 6).
#
# **Отдельная программа**, как `rag_eval.py`: её запускает человек
# (`./run.sh local`), и работает она до конца, а не живёт. Приложение её не
# импортирует. Три запроса разной сложности — данные `presets.LOCAL_QUESTIONS`:
# Л1 (простой, без игры), Л2 (вопрос К5 из общих знаний, без RAG) и Л3 (тот же
# вопрос с полным RAG дней 22-24: переписывание, поиск, реранкер, ответ с
# цитатами, проверка формата). Все три идут через `Agent.ask()` (LLM API
# вызывается только в `agent.py`) к серверу LM Studio по адресу
# `presets.LOCAL_BASE_URL`; на каждый вопрос — свежий голый агент пресета
# «Локальный» (`rag_eval.make_agent()`).
#
# **Модель программа не выбирает и не знает**: какая загружена в LM Studio, та и
# отвечает; её id приходит в ответе API (`AgentReply.answered_model`) и идёт в
# лог и отчёт по каждому вопросу — модель могут сменить посреди прогона.
# `local_server.check()` до вопросов спрашивает у сервера, что загружено, и
# отказывает, если сервер молчит или загруженной LLM нет. Сервером LM Studio
# программа не управляет: не запускает его, не грузит и не выгружает модели —
# только подсказывает команду.
#
# Отчёт — Markdown, данные для видео. **Оценок в нём нет**: «найден» относится
# только к выдаче Л3 (`rag_search.sources_found()`), а верно ли ответила модель,
# оценивает автор. Мелкие помощники отчёта и разбор выдачи — импортом из
# `rag_eval.py`, без копий: ребро между программами, как у `rag_dialog.py`.
# Сессий на диск программа не пишет, индекс только читает, модели поиска из
# сети не качает (`allow_download=False`).
#
# Логи — stdout: строки агентов (как в приложении) и свои с префиксом
# `[Локальная модель]`. Коды выхода: 0 — все вопросы прошли, 1 — сбой посреди
# прогона (ход `ok=False` или исключение), 2 — отказ до вызова модели, 130 —
# Ctrl+C; при 1 и 130 отчёт пишется по пройденному (правила `rag_dialog.py`).

import argparse
import dataclasses
import logging
import sys
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

import openai

import agent
import embedder
import local_server
import presets
import rag_eval
import rag_search
from rag_eval import Run, cell, found_text, num

logger = logging.getLogger("local_llm")

# Латинская L в номере вопроса — за кириллическую Л (правило `rag_eval`).
_LATIN_ID = str.maketrans({"L": "Л", "l": "Л", "л": "Л"})


@dataclass
class Answer:
    """Один вопрос и его ход."""

    question: dict
    mode: rag_eval.Mode
    run: Run


def speed(reply: agent.AgentReply) -> float | None:
    """≈ток/с: `completion_tokens / elapsed` — скорость с префиллом и раундами
    (повтор формата входит), а не чистая генерация."""
    if not reply.ok or not reply.completion_tokens or reply.elapsed <= 0:
        return None
    return reply.completion_tokens / reply.elapsed


def speed_text(reply: agent.AgentReply) -> str:
    value = speed(reply)
    return "н/д" if value is None else f"{value:.0f}"


def model_text(reply: agent.AgentReply) -> str:
    """Ответившая модель по ответу API; `local` — заглушка из запроса."""
    return f"`{reply.answered_model}`" if reply.answered_model else "н/д"


def model_line(model: local_server.ModelInfo) -> str:
    parts = [part for part in (model.format.upper(), model.quantization, model.arch) if part]
    return f"`{model.id}`" + (f" ({', '.join(parts)})" if parts else "")


def context_line(model: local_server.ModelInfo) -> str:
    if model.loaded_context and model.max_context:
        return f"контекст {num(model.loaded_context)} из {num(model.max_context)}"
    if model.max_context:
        return f"контекст до {num(model.max_context)}"
    return ""


# --- Лог ------------------------------------------------------------------------

def log_answer(answer: Answer) -> None:
    question, mode, run = answer.question, answer.mode, answer.run
    reply = run.reply
    if not reply.ok:
        logger.info("%s · %s: ⚠️ ход не удался: %s", question["id"], mode.label, reply.error)
        return
    line = (
        f"{question['id']} · {question['complexity']} · режим «{mode.key}»: ответила "
        f"{reply.answered_model or 'н/д'} · {reply.elapsed:.2f} с · prompt {num(reply.prompt_tokens)} / "
        f"completion {num(reply.completion_tokens)} · ≈{speed_text(reply)} ток/с (с префиллом и раундами)"
    )
    call = reply.rewrite_call
    if call is not None:
        line += f" · переписывание {call.elapsed:.2f} с: {rag_eval.search_line(run)}"
    record = reply.rag
    if record is not None:
        if record.ok:
            line += (
                f" · поиск {record.elapsed:.2f} с"
                f"{f' (загрузка модели {record.load_s:.1f} с)' if record.load_s else ''}"
                f"{f' (загрузка реранкера {record.rerank_load_s:.1f} с)' if record.rerank_load_s else ''}"
                f" · выдержек {len(record.hits)} ≈{num(record.tokens)} ток."
            )
            if question["sources"]:
                line += f" · ожидаемые источники в выдаче: {found_text(rag_eval.places_after(run, question))}"
            if record.rerank and not record.rerank_ok:
                line += f" · ⚠️ второй этап: {record.rerank_error}"
        else:
            line += f" · ⚠️ поиск: {record.error}"
    verdict = rag_eval.answer_line(run)
    if verdict:
        line += f" · {verdict}"
    logger.info("%s", line)


# --- Отчёт ----------------------------------------------------------------------

def summary_table(answers: list[Answer]) -> list[str]:
    out = [
        "| № | сложность | режим | ответившая модель | время модели, с | prompt / completion | ≈ток/с |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for answer in answers:
        reply = answer.run.reply
        if not reply.ok:
            out.append(
                f"| {answer.question['id']} | {answer.question['complexity']} | `{answer.mode.key}` | "
                f"⚠️ ход не удался | — | — | — |"
            )
            continue
        out.append(
            f"| {answer.question['id']} | {answer.question['complexity']} | `{answer.mode.key}` | "
            f"{model_text(reply)} | {reply.elapsed:.2f} | "
            f"{num(reply.prompt_tokens)} / {num(reply.completion_tokens)} | {speed_text(reply)} |"
        )
    return out


def rag_table(answers: list[Answer]) -> list[str]:
    """Строки по вопросам с RAG: переписывание, выдача, источники, вид ответа,
    нарушения, повтор. Пусто — таких вопросов в прогоне нет."""
    rows = [answer for answer in answers if answer.mode.rag and answer.run.reply.ok]
    if not rows:
        return []
    out = [
        "| № | переписывание | выдача (оценка реранкера) | ожидаемые источники в выдаче | вид ответа | "
        "цитаты дословно | нарушения | повтор формата |",
        "| --- | --- | --- | --- | --- | --- | --- | --- |",
    ]
    for answer in rows:
        run, question = answer.run, answer.question
        record = rag_eval.record_of(run)
        check = rag_eval.answer_of(run)
        if record is None:
            reason = run.reply.rag.error if run.reply.rag is not None else "запись поиска не пришла"
            out.append(f"| {question['id']} | — | ⚠️ поиск не удался: {cell(reason)} | — | — | — | — | — |")
            continue
        call = run.reply.rewrite_call
        rewrite = (
            f"{call.elapsed:.2f} с: {rag_eval.search_line(run)}" if call is not None else "—"
        )
        found = found_text(rag_eval.places_after(run, question)) if question["sources"] else "—"
        if check is None:
            kind = quotes = violations = "—"
        else:
            kind = check.kind
            quotes = f"{check.verbatim} из {len(check.quotes)}"
            violations = str(len(check.violations))
        repaired = rag_eval.repair_rounds(run)
        if repaired:
            repair = "был" + ("" if check is None or not check.violations else ", нарушения остались")
        else:
            repair = f"нет ({record.repair_note})" if record.repair_note else "нет"
        out.append(
            f"| {question['id']} | {cell(rewrite)} | {cell(rag_eval.hits_list(record))} | {found} | {kind} | "
            f"{quotes} | {violations} | {cell(repair)} |"
        )
    return out


def build_report(
    answers: list[Answer], config: agent.AgentConfig, check: local_server.ServerCheck, info: dict | None,
    started: datetime, interrupted: str | None, total_questions: int,
) -> str:
    out: list[str] = ["# Локальная модель: три запроса разной сложности (день 26)", ""]
    if interrupted is not None:
        out += [f"**Прервано: {interrupted}** — в отчёте пройденные вопросы ({len(answers)} из {total_questions}).", ""]

    out += [
        "## 1. Условия", "",
        f"- Дата и время: {started:%Y-%m-%d %H:%M:%S}",
        f"- Сервер: `{config.base_url}` · пресет «{config.name}» · в запросе `model: {config.model}` "
        "(заглушка: какая модель ответит, решает LM Studio) · ключ не нужен",
        f"- Версия SDK `openai`: {openai.__version__} · температура {config.temperature} · "
        f"температура служебных работ {config.service_temperature} · thinking в запросе выключен",
        "- Инструкции — общие с пресетами DeepSeek: системный промпт проекта, RAG-инструкции дней 22-25, промпты "
        "служебных работ; под локальную модель ничего не менялось",
        "- Агенты: голые — свежий агент на каждый вопрос, без хранилища, памяти, профиля, инвариантов, задачи и "
        "инструментов MCP",
        (
            f"- Индекс: `{info['path']}` · собран {info['built_at']} · модель эмбеддингов `{info['model']}` · "
            f"нарезка `{info['strategy']}` · язык `{info['lang']}` · кусков: {info['chunks']} (всего в индексе "
            f"{info['total']}) · этап 1: {presets.RAG_CANDIDATES} → этап 2: `{presets.RAG_RERANK_MODEL}` ≥ "
            f"{rag_eval.threshold_text(presets.RAG_RERANK_THRESHOLD)} → ≤{presets.RAG_TOP_K}"
            if info is not None else "- Индекс: в прогоне не участвует (вопросов с RAG нет)"
        ),
        "- В отчёте нет оценок: «найден» относится только к выдаче, «дословно» — проверка формы, а не смысла; "
        "верно ли ответила модель, оценивает автор",
        "- «≈ток/с» — `completion_tokens` / время модели хода: скорость с префиллом и раундами (повтор формата "
        "входит), а не чистая генерация",
        "",
        "Модели сервера на момент старта прогона:", "",
    ]
    if check.models:
        out += [
            "| id | тип | состояние | формат | квантование | архитектура | контекст загруженный / максимальный |",
            "| --- | --- | --- | --- | --- | --- | --- |",
        ]
        for model in check.models:
            state = model.state or "не сообщается"
            loaded = (
                f"{num(model.loaded_context) if model.loaded_context else '—'} / "
                f"{num(model.max_context) if model.max_context else '—'}"
            )
            out.append(
                f"| `{model.id}` | {model.type or '—'} | {state} | {model.format or '—'} | "
                f"{model.quantization or '—'} | {model.arch or '—'} | {loaded} |"
            )
    else:
        out.append("сервер не назвал ни одной модели")
    if not check.extended:
        out += ["", "Сервер не отдаёт состояние моделей (нет REST LM Studio): какая загружена, неизвестно — "
                "ответившая модель взята из ответов API.", ""]
    else:
        out.append("")

    out += ["## 2. Сводка", "", *summary_table(answers), ""]
    table = rag_table(answers)
    if table:
        out += ["Вопросы с RAG — поиск, проверка формата и повтор (код проверяет форму, смысл — автор):", "", *table, ""]

    out += ["## 3. Ответы", ""]
    for answer in answers:
        question, mode, run = answer.question, answer.mode, answer.run
        out += [
            f"### {question['id']} · {question['complexity']}", "",
            f"**Вопрос:** {question['question']}", "",
            f"**Режим:** `{mode.key}` — {mode.label}", "",
            f"**Что проверяет:** {question['checks']}", "",
        ]
        if run.reply.ok:
            out += [f"**Ответила модель:** {model_text(run.reply)} (в запросе `{run.reply.model}`)", ""]
        if mode.rag:
            record = rag_eval.record_of(run)
            if record is None:
                reason = run.reply.rag.error if run.reply.rag is not None else "запись поиска не пришла"
                out += [f"⚠️ Поиск не удался: {reason} — запрос ушёл без выдержек.", ""]
            else:
                if question["sources"]:
                    out += ["**Ожидаемые источники:**", ""]
                    for source in question["sources"]:
                        out.append(
                            f"- `{source[0]}` · «{source[1]}» · стр. {source[2]}: "
                            f"{rag_eval.source_status(source, run)}"
                        )
                    out.append("")
                if run.reply.rewrite_call is not None:
                    out += [f"**Запрос поиска:** {rag_eval.search_line(run)}", ""]
                notes = f" · ⚠️ второй этап не удался: {record.rerank_error}" if record.rerank and not record.rerank_ok else ""
                out += [f"Выдача: {rag_eval.hits_list(record)}{notes}", ""]
        out += rag_eval.answer_block(run, mode, question)
    return "\n".join(out).rstrip() + "\n"


# --- Прогон ---------------------------------------------------------------------

def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="./run.sh local",
        description=(
            "Три запроса разной сложности к локальной модели через ассистента (день 26): Л1 — простой, Л2 — вопрос "
            "по правилам из общих знаний, Л3 — тот же вопрос с полным RAG. Отвечает модель, загруженная в LM "
            "Studio (lms load <id>, lms server start); ключ не нужен. Для Л3 нужны индекс (./run.sh index) и "
            "реранкер в кэше (./run.sh index --probe)."
        ),
    )
    parser.add_argument("--only", metavar="Л1,Л3", help="часть вопросов (латинская L принимается за Л)")
    parser.add_argument("--url", metavar="АДРЕС", default=None,
                        help=f"адрес сервера вместо {presets.LOCAL_BASE_URL}")
    parser.add_argument("--db", type=Path, default=presets.RAG_INDEX_DB, metavar="ПУТЬ",
                        help=f"файл индекса (по умолчанию {rag_search.shown(presets.RAG_INDEX_DB)})")
    parser.add_argument("--out", type=Path, default=None, metavar="ФАЙЛ.md",
                        help=f"куда писать отчёт (по умолчанию {rag_search.shown(presets.LOCAL_REPORT_DIR)}/<дата-время>.md)")
    return parser.parse_args(argv)


def check_server(base_url: str) -> local_server.ServerCheck:
    """Проверка сервера до вопросов (§5, п. 2): отказ — `sys.exit(2)`, остальное —
    предупреждения."""
    check = local_server.check(base_url, presets.LOCAL_CHECK_TIMEOUT_S)
    if not check.ok:
        logger.error("%s — прогон отменён, модель не вызывалась. Запустите сервер: lms server start", check.error)
        sys.exit(2)
    llms = [model for model in check.models if model.type != "embeddings"]
    if check.extended:
        loaded = check.loaded
        if not loaded:
            ids = ", ".join(f"`{model.id}`" for model in llms) or "(сервер не назвал ни одной)"
            logger.error(
                "сервер отвечает, но модель не загружена — lms load <id>; модели сервера: %s — прогон отменён, "
                "модель не вызывалась", ids,
            )
            sys.exit(2)
        if len(loaded) > 1:
            logger.warning(
                "загружено несколько моделей (%s): что LM Studio берёт для запроса без id, не проверялось — "
                "держите загруженной одну LLM", ", ".join(model.id for model in loaded),
            )
        else:
            model = loaded[0]
            logger.info(
                "сервер отвечает за %.2f с · загружена %s%s", check.elapsed, model_line(model),
                f" · {context_line(model)}" if context_line(model) else "",
            )
    else:
        logger.warning("сервер отвечает, но не сообщает, какая модель загружена — ответившая модель будет "
                       "взята из ответов API")
    return check


def main() -> None:
    args = _args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s", stream=sys.stdout)
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[Локальная модель] %(message)s"))
    logger.addHandler(handler)
    logger.propagate = False

    config = presets.PRESETS[presets.LOCAL_PRESET]
    if args.url:
        config = dataclasses.replace(config, base_url=args.url)

    questions = list(presets.LOCAL_QUESTIONS)
    if args.only:
        wanted = [item.strip().translate(_LATIN_ID) for item in args.only.split(",") if item.strip()]
        known = {q["id"] for q in questions}
        unknown = [item for item in wanted if item not in known]
        if unknown or not wanted:
            logger.error("нет вопросов: %s; есть: %s", ", ".join(unknown) or "(пусто)", ", ".join(sorted(known)))
            sys.exit(2)
        questions = [q for q in questions if q["id"] in wanted]
    modes = {q["id"]: rag_eval.MODE_BY_KEY[q["mode"]] for q in questions}

    check = check_server(config.base_url)

    index = rag_search.RulesIndex(
        args.db.expanduser(), presets.RAG_SEARCH_STRATEGY, presets.RAG_SEARCH_LANG, presets.RAG_TOP_K,
        candidates_k=presets.RAG_CANDIDATES, rerank_model=presets.RAG_RERANK_MODEL,
        threshold=presets.RAG_RERANK_THRESHOLD, rerank_max_length=presets.RAG_RERANK_MAX_LENGTH,
        rerank_batch=presets.RAG_RERANK_BATCH, allow_download=False, expected_model=presets.RAG_EMBED_MODEL,
    )
    info: dict | None = None
    if any(mode.rag for mode in modes.values()):
        info = index.info()
        if not info["ok"]:
            logger.error("%s — прогон отменён, модель не вызывалась", info["error"])
            sys.exit(2)
        if not embedder.in_cache(presets.RAG_RERANK_MODEL):
            logger.error(
                "реранкера %s нет в кэше Hugging Face — ./run.sh index --probe скачает ≈1,5 ГБ; прогон отменён, "
                "модель не вызывалась", presets.RAG_RERANK_MODEL,
            )
            sys.exit(2)
        logger.info(
            "индекс %s · собран %s · %s/%s: %d кусков из %d", info["path"], info["built_at"], info["strategy"],
            info["lang"], info["chunks"], info["total"],
        )
    logger.info(
        "пресет «%s» · %s · температура %s · вопросов: %d (%s)", config.name, config.base_url, config.temperature,
        len(questions), ", ".join(q["id"] for q in questions),
    )

    started = datetime.now()
    out_path = (args.out.expanduser() if args.out is not None
                else presets.LOCAL_REPORT_DIR / f"{started:%Y-%m-%d-%H%M%S}.md")
    answers: list[Answer] = []
    interrupted: str | None = None
    code = 0
    try:
        for question in questions:
            mode = modes[question["id"]]
            bare = rag_eval.make_agent(config, f"local-{question['id']}", index, mode)
            answer = Answer(question, mode, Run(bare.ask(question["question"])))
            answers.append(answer)
            log_answer(answer)
            if not answer.run.reply.ok:
                interrupted, code = f"{question['id']}: ход не удался — {answer.run.reply.error}", 1
                logger.error("сбой на %s — пишу отчёт по пройденному", question["id"])
                break
    except KeyboardInterrupt:
        interrupted, code = _passed(answers), 130
        logger.error("прервано %s — пишу отчёт по пройденному", interrupted)
    except Exception as exc:  # noqa: BLE001 — оплаченные ходы важнее трейсбека (правило rag_dialog)
        interrupted, code = f"{_passed(answers)}: сбой — {exc}", 1
        logger.exception("сбой %s — пишу отчёт по пройденному", _passed(answers))

    report = build_report(
        answers, config, check, {**info, **index.info()} if info is not None else None, started, interrupted,
        len(questions),
    )
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(report, encoding="utf-8")
    logger.info("отчёт: %s (%d из %d вопросов)", rag_search.shown(out_path), len(answers), len(questions))
    print()
    print("Сводка" + report.partition("## 2. Сводка")[2].partition("## 3. ")[0].rstrip())
    sys.exit(code)


def _passed(answers: list[Answer]) -> str:
    return f"после {answers[-1].question['id']}" if answers else "до первого вопроса"


if __name__ == "__main__":
    main()
