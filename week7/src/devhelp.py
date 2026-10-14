"""Ассистент разработчика: /help, RAG, контекст git и ссылки на источники."""

import argparse
import asyncio
import os
from pathlib import Path
import re
import readline  # История строк ввода; история разговора в модель не передаётся.

from dotenv import load_dotenv

from llm import LlmParams, answer
from mcp_client import McpSession
from search import DocsIndex
import settings as cfg


def log(text):
    print(f"[devhelp] {text}", flush=True)


def show_help():
    print("Ассистент ищет документацию проекта и читает контекст git. Каждый вопрос независим.\n"
          "Команды: /help <вопрос>, /help — справка, /exit — выход.\n"
          "Ctrl+D — выход, Ctrl+C во время ответа — отмена ответа.\nПримеры:")
    for question in cfg.EXAMPLE_QUESTIONS:
        print("  /help " + question)


def show_answer(result, search, branch, calls):
    if not result.ok:
        print(f"Ошибка LLM: {result.error}")
        return
    print("\n" + result.text)
    references = sorted({int(number) for number in re.findall(r"\[(\d+)\]", result.text)})
    if references:
        print("\nИсточники:")
        for number in references:
            if 1 <= number <= len(search.hits):
                print(f"  [{number}] {search.hits[number - 1].label()}")
            else:
                print(f"  ⚠️ [{number}] — такой выдержки не было")
    else:
        print("\n⚠️ ответ не ссылается на выдержки — сверить его с документацией нельзя")
    print(f"Git: ветка {branch} (git_branch) · вызвано моделью: {', '.join(calls) or 'нет'}")
    cost = "стоимость неизвестна" if result.cost_usd is None else f"≈ ${result.cost_usd:.6f}"
    print(f"Поиск {search.elapsed:.2f} с · модель {sum(round.elapsed for round in result.rounds):.2f} с · "
          f"раундов {len(result.rounds)} · {result.prompt_tokens:,} → {result.completion_tokens:,} токенов · {cost}")
    if result.finish_reason == "length":
        print("⚠️ ответ достиг ограничения длины")


async def ask(question, repo, index, params):
    # Выполняется в текущем потоке: Ctrl+C прерывает также CPU-поиск.
    search = index.search(question)
    log(f"поиск {search.elapsed:.2f} с, реранк {search.rerank_elapsed:.2f} с")
    if search.rerank_error:
        log("⚠️ реранкер недоступен, взяты ближайшие векторы: " + search.rerank_error)
    for number, hit in enumerate(search.hits, 1):
        score = "—" if hit.score is None else f"{hit.score:.3f}"
        log(f"[{number}] sim={hit.sim:.3f} score={score} до реранка #{hit.rank} · {hit.label()}")
    async with McpSession(cfg.GIT_SERVER[0], [*cfg.GIT_SERVER[1:], "--repo", str(repo)],
                          env=None, timeout_s=cfg.MCP_TIMEOUT_S) as session:
        log(f"MCP подключение {session.elapsed:.3f} с · {session.error or 'готово'}")
        branch_call = await session.call("git_branch", {})
        branch = branch_call.text if branch_call.ok else cfg.UNKNOWN_BRANCH.format(reason=branch_call.error)
        log(f"MCP git_branch {branch_call.elapsed:.3f} с · {branch}")
        catalog = await session.tools()
        log(f"MCP каталог {session.tools_elapsed:.3f} с: "
            f"{', '.join(tool['name'] for tool in catalog) or 'пуст'} · {session.error or 'готово'}")
        tools = [{"type": "function", "function": {"name": tool["name"],
                  "description": tool["description"], "parameters": tool["inputSchema"]}}
                 for tool in catalog if tool["name"] != "git_branch"]
        excerpts = "\n\n".join(f"[{number}] {hit.label()}\n{hit.text}" for number, hit in enumerate(search.hits, 1))
        messages = [{"role": "system", "content": cfg.SYSTEM_PROMPT},
                    {"role": "user", "content": cfg.USER_TEMPLATE.format(repo=repo.name, branch=branch,
                                                                           excerpts=excerpts, question=question)}]
        calls = []
        async def call_tool(name, arguments):
            calls.append(name)
            result = await session.call(name, arguments)
            log(f"MCP {name} {arguments} · {result.elapsed:.3f} с · {result.text if result.ok else result.error}")
            return result
        result = await answer(messages, tools, call_tool, params)
    show_answer(result, search, branch, list(dict.fromkeys(calls)))
    return 0 if result.ok else 1


def run_question(question, repo, index, params):
    try:
        return asyncio.run(ask(question, repo, index, params))
    except KeyboardInterrupt:
        print("\nОтвет отменён")
        return 130
    except Exception as exc:
        log(f"сбой ответа: {exc}")
        return 1


def main():
    parser = argparse.ArgumentParser(description="Ассистент разработчика, /help")
    parser.add_argument("--repo", type=Path, default=cfg.DEFAULT_REPO)
    parser.add_argument("--db", type=Path, default=cfg.INDEX_DB)
    parser.add_argument("question", nargs="?", help="разовый вопрос")
    args = parser.parse_args()
    repo, db = args.repo.expanduser().resolve(), args.db.expanduser().resolve()
    load_dotenv(cfg.WEEK7_DIR / ".env", override=False)
    key = os.environ.get("LLM_API_KEY", "").strip()
    if not key:
        log("отказ: нет LLM_API_KEY в week7/.env")
        return 2
    params = LlmParams(os.environ.get("LLM_BASE_URL", cfg.LLM_BASE_URL), key,
                       os.environ.get("LLM_MODEL", cfg.LLM_MODEL), cfg.LLM_TEMPERATURE,
                       cfg.LLM_MAX_TOKENS, cfg.LLM_MAX_TOOL_ROUNDS, cfg.TOOL_RESULT_MAX_CHARS,
                       cfg.PRICES, cfg.LLM_TIMEOUT_S)
    try:
        if not repo.is_dir():
            raise ValueError("--repo не существует или не каталог")
        index = DocsIndex(db, cfg.MODELS_DIR, rerank_model=cfg.RERANK_MODEL,
                          candidates=cfg.SEARCH_CANDIDATES, top_k=cfg.SEARCH_TOP_K)
        for warning in index.warnings(repo, cfg.DOCS_GLOBS, cfg.DOCS_EXCLUDE):
            log("⚠️ " + warning)
        index.load_models(allow_download=False)
    except Exception as exc:
        log(f"отказ: {exc}")
        return 2
    log(f"репозиторий {repo} · индекс {db}, {len(index.chunks)} кусков · "
        f"HEAD {index.meta['head']} · {index.meta['built_at']} · LLM {params.model} · адрес {params.base_url}")
    if args.question is not None:
        if not args.question.strip():
            log("отказ: пустой вопрос")
            return 2
        return run_question(args.question, repo, index, params)
    show_help()
    while True:
        try:
            line = input("devhelp> ").strip()
        except EOFError:
            print()
            return 0
        except KeyboardInterrupt:
            print()
            continue
        if line == "/exit":
            return 0
        if line == "/help":
            show_help()
        elif line.startswith("/help ") or line.startswith("/help\t"):
            question = line[5:].strip()
            run_question(question, repo, index, params) if question else show_help()
        else:
            print("вопрос задаётся командой /help <вопрос>")


if __name__ == "__main__":
    raise SystemExit(main())
