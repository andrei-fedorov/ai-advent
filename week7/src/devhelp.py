"""Ассистент разработчика: RAG, файлы через MCP, черновики Markdown, CLI и страница."""

import argparse
import asyncio
from collections import OrderedDict
from dataclasses import dataclass
import json
import os
from pathlib import Path
import readline  # История ввода; история разговора в модель не передаётся.
import time

from dotenv import load_dotenv

from citations import source_lines
import file_changes as changes
from llm import LlmParams, LlmResult, answer
from local_web import make_app, serve
from mcp_client import McpSession, ToolCall
from search import DocsIndex
import settings as cfg


def log(text):
    print(f"[devhelp] {text}", flush=True)


def git_session(repo):
    return McpSession(cfg.GIT_SERVER[0], [*cfg.GIT_SERVER[1:], "--repo", str(repo)],
                      env=None, timeout_s=cfg.MCP_TIMEOUT_S)


async def mcp_call(session, name, arguments, calls, service=False):
    # service — вызов делает код, а не модель: на странице он помечен, в шапку .diff не идёт.
    result = await session.call(name, arguments)
    calls.append({"name": name, "arguments": arguments, "elapsed": result.elapsed,
                  "size": len(result.text), "error": result.error, "service": service})
    text = result.text
    if name in ("project_read", "project_grep", "project_files"):
        text = text.splitlines()[0] if text else ""
    log(f"MCP {name} {json.dumps(arguments, ensure_ascii=False)} · {result.elapsed:.3f} с · "
        f"{len(result.text)} симв. · {text if result.ok else result.error}")
    return result


async def path_status(session, path, calls):
    result = await mcp_call(session, "project_path_status", {"path": path}, calls, service=True)
    if not result.ok:
        raise ValueError(result.error or "статус пути недоступен")
    return json.loads(result.text)


@dataclass
class Answer:
    text: str
    sources: list[str]
    branch: str
    calls: list[dict]
    result: LlmResult
    draft: changes.Draft
    diff_path: Path | None
    search: object
    goal: str

    def payload(self):
        return {"answer": self.text, "sources": self.sources, "branch": self.branch,
                "change": {"id": self.draft.id, "diff_path": str(self.diff_path),
                           "files": [changes.file_summary(path, item) for path, item in self.draft.files.items()]}
                          if self.draft.files else None,
                "details": {"query": self.goal,
                            "hits": [{"label": hit.label(), "sim": hit.sim, "score": hit.score}
                                     for hit in self.search.hits],
                            "calls": self.calls, "rounds": len(self.result.rounds),
                            "prompt_tokens": self.result.prompt_tokens,
                            "completion_tokens": self.result.completion_tokens,
                            "cost_usd": self.result.cost_usd, "finish_reason": self.result.finish_reason}}


async def ask(goal, repo, index, params, out_dir) -> Answer:
    # CPU-поиск не блокирует HTTP health и второй POST (409).
    search = await asyncio.to_thread(index.search, goal, kinds=("current", "docs"))
    log(f"поиск {search.elapsed:.2f} с, реранк {search.rerank_elapsed:.2f} с")
    if search.rerank_error:
        log("⚠️ реранкер недоступен, взяты ближайшие векторы: " + search.rerank_error)
    for number, hit in enumerate(search.hits, 1):
        score = "—" if hit.score is None else f"{hit.score:.3f}"
        log(f"[{number}] sim={hit.sim:.3f} score={score} до реранка #{hit.rank} · {hit.label()}")
    draft, calls = changes.Draft(goal), []
    async with git_session(repo) as session:
        log(f"MCP подключение {session.elapsed:.3f} с · {session.error or 'готово'}")
        branch_call = await mcp_call(session, "git_branch", {}, calls, service=True)
        branch = branch_call.text if branch_call.ok else cfg.UNKNOWN_BRANCH.format(reason=branch_call.error)
        catalog = await session.tools()
        log(f"MCP каталог {session.tools_elapsed:.3f} с: "
            f"{', '.join(tool['name'] for tool in catalog if tool['name'] in cfg.DEVHELP_TOOLS) or 'пуст'} · "
            f"{session.error or 'готово'}")
        tools = [{"type": "function", "function": {"name": tool["name"],
                  "description": tool["description"], "parameters": tool["inputSchema"]}}
                 for tool in catalog if tool["name"] in cfg.DEVHELP_TOOLS] + list(cfg.DEVHELP_LOCAL_TOOLS)
        excerpts = "\n\n".join(f"[{number}] {hit.label()}\n{hit.text}" for number, hit in enumerate(search.hits, 1))
        messages = [{"role": "system", "content": cfg.SYSTEM_PROMPT},
                    {"role": "user", "content": cfg.USER_TEMPLATE.format(repo=repo.name, branch=branch,
                                                                           excerpts=excerpts, goal=goal)}]

        async def call_tool(name, arguments):
            if name in cfg.DEVHELP_TOOLS:
                return await mcp_call(session, name, arguments, calls)
            started = time.monotonic()
            path = arguments.get("path")
            try:
                required = {"path", "old", "new"} if name == "propose_edit" else {"path", "content"}
                if name not in ("propose_edit", "propose_file") or set(arguments) != required:
                    raise ValueError("недопустимые аргументы инструмента")
                if not all(isinstance(value, str) for value in arguments.values()):
                    raise ValueError("аргументы должны быть строками")
                status = await path_status(session, path, calls)
                if name == "propose_edit":
                    path = changes.propose_edit(draft, repo, status=status, **arguments,
                                                max_files=cfg.DEVHELP_MAX_CHANGED_FILES)
                else:
                    path = changes.propose_file(draft, repo, status=status, **arguments,
                                                max_files=cfg.DEVHELP_MAX_CHANGED_FILES,
                                                max_chars=cfg.DEVHELP_NEW_FILE_MAX_CHARS)
                if path in draft.files:
                    item = changes.file_summary(path, draft.files[path])
                    label = "новый" if item["kind"] == "new" else "правка"
                    detail = f"{path} · {label} · +{item['added']} −{item['removed']}"
                    text = f"правка принята: {detail} (на диске не изменён)"
                else:
                    detail = f"{path} · правки отменены: текст совпадает с исходным"
                    text = detail + " (на диске не изменён)"
                log(f"черновик {draft.id}: {detail}")
                result = ToolCall(True, text, time.monotonic() - started)
            except (ValueError, OSError, UnicodeError, TypeError) as exc:
                log(f"черновик {draft.id}: {path} · отказ: {exc}")
                result = ToolCall(False, "", time.monotonic() - started, str(exc))
            details = arguments
            if name == "propose_file":
                content = arguments.get("content")
                details = {"path": path, "length": len(content) if isinstance(content, str) else 0}
            calls.append({"name": name, "arguments": details, "elapsed": result.elapsed,
                          "size": len(result.text), "error": result.error, "service": False})
            return result

        result = await answer(messages, tools, call_tool, params,
                              final_tools=cfg.DEVHELP_LOCAL_TOOLS,
                              final_tool_rounds=cfg.DEVHELP_FINAL_TOOL_ROUNDS,
                              tool_text_retries=cfg.DEVHELP_TOOL_TEXT_RETRIES)
        patch = None
        if draft.files:
            head = await mcp_call(session, "git_log", {"limit": 1}, calls, service=True)
            patch = changes.write_patch(draft, out_dir, {"репозиторий": repo,
                "HEAD": head.text.split(" | ", 1)[0] if head.ok else "неизвестен",
                "модель": result.answered_model or params.model, "раунды": len(result.rounds),
                "инструменты": ", ".join(call["name"] for call in calls if not call["service"])})
            log(f"diff {patch} · файлов {len(draft.files)}")
    sources = source_lines(result.text, search.hits) if result.ok else []
    return Answer(result.text, sources, branch, calls, result, draft, patch, search, goal)


async def apply_file(draft, path, repo):
    item = draft.files.get(path)
    if item is None:
        return {"path": path, "status": "отказ", "message": "файла нет в черновике"}
    if item.applied:
        return {"path": path, "status": "уже применён", "message": "уже применён"}
    try:
        async with git_session(repo) as session:
            status = await path_status(session, path, [])
            state, message = changes.apply(draft, repo, path, status)
    except (ValueError, OSError) as exc:
        state, message = "отказ", str(exc)
    if state == "записан":
        log(f"применено {draft.id} {path} · sha {(item.original_sha256 or 'нет')[:8]}→{changes.sha(item.new_text)[:8]}")
    else:
        log(f"применение {draft.id} {path} · отказ: {message}")
    return {"path": path, "status": state, "message": message}


def show_help():
    print("Ассистент ищет документацию и файлы проекта, предлагает правки Markdown. Каждая цель независима.\n"
          "Команды: /help <цель>, /help — справка, /exit — выход.\n"
          "Ctrl+D — выход, Ctrl+C во время ответа — отмена ответа.\nПримеры:")
    for goal in cfg.DEVHELP_GOALS:
        print("  /help " + goal)


def show_answer(response):
    result = response.result
    if not result.ok:
        print(f"Ошибка LLM: {result.error}")
    else:
        print("\n" + response.text)
        print("\n" + "\n".join(response.sources))
    if response.draft.files:
        print("\nПравки")
        for path, item in response.draft.files.items():
            entry = changes.file_summary(path, item)
            print(f"{path} · {'новый' if item.kind == 'new' else 'правка'} · +{entry['added']} −{entry['removed']}")
            print(entry["diff"])
        print(f"Diff: {response.diff_path}")
    names = list(dict.fromkeys(call["name"] for round in result.rounds for call in round.calls))
    print(f"Git: ветка {response.branch} (git_branch) · вызвано моделью: {', '.join(names) or 'нет'}")
    cost = "стоимость неизвестна" if result.cost_usd is None else f"≈ ${result.cost_usd:.6f}"
    print(f"Поиск {response.search.elapsed:.2f} с · модель {sum(r.elapsed for r in result.rounds):.2f} с · "
          f"раундов {len(result.rounds)} · {result.prompt_tokens:,} → {result.completion_tokens:,} токенов · {cost}")
    if result.finish_reason == "length":
        print("⚠️ ответ достиг ограничения длины")


async def cli_answer(goal, repo, index, params, out_dir, apply):
    response = await ask(goal, repo, index, params, out_dir)
    show_answer(response)
    code = 0 if response.result.ok else 1
    if response.draft.files:
        if apply and response.result.ok:
            for path in response.draft.files:
                result = await apply_file(response.draft, path, repo)
                print(f"{path}: {result['status']}" +
                      (f" · {result['message']}" if result["status"] == "отказ" else ""))
                if result["status"] == "отказ":
                    code = 1
        else:
            print(f"файлы не изменены; --apply или git apply '{response.diff_path}'")
    return code


def run_question(goal, repo, index, params, out_dir, apply):
    try:
        return asyncio.run(cli_answer(goal, repo, index, params, out_dir, apply))
    except KeyboardInterrupt:
        print("\nОтвет отменён")
        return 130
    except Exception as exc:
        log(f"сбой ответа: {exc}")
        return 1


def web_app(repo, db, index, params, out_dir, port):
    drafts = OrderedDict()

    def valid_goal(data):
        if (not isinstance(data, dict) or set(data) != {"goal"}
                or not isinstance(data["goal"], str) or not data["goal"].strip()):
            return "нужна goal (непустая строка)"
        return None

    def valid_apply(data):
        if (not isinstance(data, dict) or set(data) != {"change", "path"}
                or not all(isinstance(value, str) and value for value in data.values())):
            return "нужны change и path (непустые строки)"
        return None

    async def chat(data):
        response = await ask(data["goal"].strip(), repo, index, params, out_dir)
        if response.draft.files:
            drafts[response.draft.id] = response.draft
            while len(drafts) > cfg.DEVHELP_KEPT_DRAFTS:
                drafts.popitem(last=False)
        if not response.result.ok:
            # Даже при сбое финального ответа подготовленный diff остаётся доступен.
            payload = response.payload()
            payload["answer"] = "Ошибка LLM: " + str(response.result.error)
            return payload, 200
        return response.payload(), 200

    async def apply(data):
        draft = drafts.get(data["change"])
        if draft is None:
            return {"error": "черновик не найден; примените .diff вручную"}, 404
        return await apply_file(draft, data["path"], repo), 200

    return make_app(page=cfg.DEVHELP_PAGE, port=port, max_body_bytes=cfg.DEVHELP_MAX_BODY_BYTES,
                    log_prefix="[devhelp]", posts={"/api/chat": (valid_goal, chat), "/api/apply": (valid_apply, apply)},
                    health=lambda: {"repo": str(repo), "db": str(db), "model": params.model,
                                    "chunks": len(index.chunks), "goals": cfg.DEVHELP_GOALS})


def main():
    parser = argparse.ArgumentParser(description="Ассистент разработчика, /help")
    parser.add_argument("--repo", type=Path, default=cfg.DEFAULT_REPO)
    parser.add_argument("--db", type=Path, default=cfg.INDEX_DB)
    parser.add_argument("--out", type=Path, default=cfg.CHANGES_DIR)
    parser.add_argument("--apply", action="store_true")
    parser.add_argument("--web", action="store_true")
    parser.add_argument("--port", type=int)
    parser.add_argument("question", nargs="?", help="разовая цель или вопрос")
    args = parser.parse_args()
    if args.web and (args.question is not None or args.apply):
        parser.error("--web нельзя сочетать с целью или --apply")
    if args.port is not None and not args.web:
        parser.error("--port требует --web")
    port = cfg.DEVHELP_PORT if args.port is None else args.port
    if not 1 <= port <= 65535:
        parser.error("--port должен быть от 1 до 65535")
    if args.question is not None and not args.question.strip():
        parser.error("пустая цель")
    repo, db, out_dir = (path.expanduser().resolve() for path in (args.repo, args.db, args.out))
    load_dotenv(cfg.WEEK7_DIR / ".env", override=False)
    key = os.environ.get("LLM_API_KEY", "").strip()
    if not key:
        log("отказ: нет LLM_API_KEY в week7/.env")
        return 2
    params = LlmParams(os.environ.get("LLM_BASE_URL", cfg.LLM_BASE_URL), key,
                       os.environ.get("LLM_MODEL", cfg.LLM_MODEL), cfg.LLM_TEMPERATURE,
                       cfg.DEVHELP_MAX_TOKENS, cfg.DEVHELP_MAX_TOOL_ROUNDS, cfg.TOOL_RESULT_MAX_CHARS,
                       cfg.PRICES, cfg.LLM_TIMEOUT_S, log_prefix="[devhelp]")
    try:
        if not repo.is_dir():
            raise ValueError("--repo не существует или не каталог")
        out_dir.mkdir(parents=True, exist_ok=True)
        # Проверяется реальная возможность записи, в том числе в существующий каталог.
        probe = out_dir / (".probe-" + changes.Draft("").id)
        with probe.open("x"):
            pass
        probe.unlink()
        index = DocsIndex(db, cfg.MODELS_DIR, rerank_model=cfg.RERANK_MODEL,
                          candidates=cfg.SEARCH_CANDIDATES, top_k=cfg.SEARCH_TOP_K)
        for warning in index.warnings(repo, cfg.DOCS_GLOBS + cfg.CODE_GLOBS, cfg.DOCS_EXCLUDE + cfg.CODE_EXCLUDE):
            log("⚠️ " + warning)
        index.load_models(allow_download=False)
        ready = (f"репозиторий {repo} · индекс {db}, {len(index.chunks)} кусков · "
                 f"HEAD {index.meta['head']} · {index.meta['built_at']} · LLM {params.model} · адрес {params.base_url}")
        if args.web:
            app = web_app(repo, db, index, params, out_dir, port)
            serve(app, port, lambda: log(ready + f" · страница http://127.0.0.1:{port}"))
            return 0
    except KeyboardInterrupt:
        return 0
    except (ValueError, OSError, RuntimeError) as exc:
        log(f"отказ: {exc}")
        return 2
    log(ready)
    if args.question is not None:
        return run_question(args.question, repo, index, params, out_dir, args.apply)
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
            goal = line[5:].strip()
            run_question(goal, repo, index, params, out_dir, args.apply) if goal else show_help()
        else:
            print("цель задаётся командой /help <цель>")


if __name__ == "__main__":
    raise SystemExit(main())
