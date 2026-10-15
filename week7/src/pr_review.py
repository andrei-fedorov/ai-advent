"""Ревью ветки: контекст MCP, RAG документации и кода, проверяемый Markdown."""

import argparse
import asyncio
from collections import Counter
import fnmatch
import json
import os
from pathlib import Path, PurePosixPath
import re
import tempfile
import time

from dotenv import load_dotenv

from llm import LlmParams, answer
from mcp_client import McpSession, ToolCall
from review_report import numbered_diff, numbered_text, render_review, search_query
from search import DocsIndex
import settings as cfg


class StartRefused(ValueError):
    pass


def log(text):
    print(f"[review] {text}", flush=True)


def changed_files(text):
    lines = text.splitlines()
    revisions = json.loads(lines[1].removeprefix("ревизии: "))
    files, warnings = [], []
    decoder = json.JSONDecoder()
    for line in lines[2:]:
        if line.startswith("и ещё "):
            warnings.append("список файлов ограничен сервером: " + line)
            continue
        status, names, count = line.split("\t")
        first, end = decoder.raw_decode(names)
        path = json.loads(names[end:].removeprefix(" → ")) if names[end:] else first
        binary = count == "двоичный"
        numbers = re.findall(r"\d+", count)
        files.append({"status": status, "path": path, "binary": binary,
                      "changes": sum(map(int, numbers)), "note": count})
    return revisions, files, warnings


def merge_hits(results, limit):
    combined = {}
    for result in results:
        for hit in result.hits:
            key = (hit.doc, hit.section, hit.part, hit.line)
            value = hit.score if hit.score is not None else hit.sim
            old = combined.get(key)
            if old is None or value > (old.score if old.score is not None else old.sim):
                combined[key] = hit
    return sorted(combined.values(), key=lambda hit: hit.score if hit.score is not None else hit.sim,
                  reverse=True)[:limit]


async def review(args, index, params):
    started = time.monotonic()
    async with McpSession(cfg.GIT_SERVER[0], [*cfg.GIT_SERVER[1:], "--repo", str(args.repo)],
                          env=None, timeout_s=cfg.MCP_TIMEOUT_S) as session:
        log(f"MCP подключение {session.elapsed:.3f} с · {session.error or 'готово'}")
        if session.error:
            raise StartRefused(session.error)

        async def call(name, arguments, *, optional=False, startup=False):
            result = await session.call(name, arguments)
            log(f"MCP {name} {json.dumps(arguments, ensure_ascii=False)} · {result.elapsed:.3f} с · "
                + (f"{len(result.text)} символов" if result.ok else str(result.error)))
            if not result.ok and not optional:
                raise (StartRefused if startup else RuntimeError)(result.error)
            return result

        initial = await call("git_changed_files", {"base": args.base, "head": args.head}, startup=True)
        revisions, files, warnings = changed_files(initial.text)
        base, head = revisions["base"], revisions["head"]
        log(f"репозиторий {args.repo} · base {args.base}={base} · head {args.head}={head} · "
            f"индекс {args.db} · модель {params.model} · адрес {params.base_url}")
        warnings.extend(index.warnings(args.repo, cfg.DOCS_GLOBS + cfg.CODE_GLOBS,
                                       cfg.DOCS_EXCLUDE + cfg.CODE_EXCLUDE))
        if index.meta["head"] != head:
            warnings.append("индекс собран на другой ревизии: " + index.meta["head"])
        if not files:
            for warning in warnings:
                log("⚠️ " + warning)
            report, stats = render_review(json.dumps({"summary": "Изменений между base и head нет.", "findings": []}),
                                          changed=set(), ranges={}, source_labels=[],
                                          context_lines=[f"base `{base[:12]}` · head `{head[:12]}` · файлов 0"],
                                          warnings=warnings)
        else:
            try:
                index.load_models(allow_download=False)
            except Exception as exc:
                raise StartRefused(str(exc)) from exc
            selected = []
            for file in sorted(files, key=lambda file: -file["changes"]):
                if file["binary"] or file["status"] == "D" or any(
                        fnmatch.fnmatch(file["path"], pattern) for pattern in cfg.REVIEW_SKIP_GLOBS):
                    file["note"] += " · пропущен (двоичный, удалённый или исключённый)"
                elif len(selected) >= cfg.REVIEW_MAX_FILES:
                    file["note"] += " · не рассматривался"
                    warnings.append(file["path"] + ": не рассматривался (предел файлов)")
                else:
                    selected.append(file)
            log(f"отбор: {len(files)} файлов, рассматривается {len(selected)}")
            diffs, full_files, ranges, queries = [], [], {}, []
            diff_chars = full_chars = 0
            reviewed_paths = set()

            async def read_file(path, limit, *, optional=False):
                parts, next_line = [], 1
                while True:
                    result = await call("git_show_file", {"ref": head, "path": path, "start_line": next_line},
                                        optional=optional)
                    if not result.ok:
                        # Отсутствующий файл правил — штатный случай, остальные сбои не скрываем.
                        detail = result.error or ""
                        if optional and any(marker in detail for marker in
                                            ("файл отсутствует в ревизии:",)):
                            return None
                        raise RuntimeError(detail)
                    header, _, body = result.text.partition("\n")
                    match = re.search(r"строки (\d+)–(\d+) из (\d+)$", header)
                    if not match:
                        raise RuntimeError("неожиданный заголовок git_show_file")
                    begin, end, total = map(int, match.groups())
                    body = body.removesuffix("\nобрезано: продолжение не поместилось")
                    parts.append(body)
                    if sum(map(len, parts)) > limit:
                        if optional:
                            warnings.append(path + ": правила превышают лимит, не включены")
                            return ""
                        return None
                    if end >= total:
                        return "".join(parts)
                    if end < begin:
                        if optional:
                            warnings.append(path + ": строка правил превышает лимит, не включены")
                            return ""
                        return None
                    next_line = end + 1

            for file in selected:
                path = file["path"]
                raw = (await call("git_range_diff", {"base": base, "head": head, "path": path,
                                                       "context": cfg.REVIEW_DIFF_CONTEXT})).text
                if len(queries) < cfg.REVIEW_MAX_QUERIES:
                    queries.append(search_query(path, raw, cfg.REVIEW_QUERY_MAX_CHARS))
                numbered, ranges[path] = numbered_diff(raw)
                if diff_chars + len(raw) > cfg.REVIEW_DIFF_MAX_CHARS:
                    file["note"] += " · diff не поместился"
                    warnings.append(path + ": diff не поместился")
                else:
                    diff_chars += len(raw)
                    diffs.append(f"Файл: {path}\n{numbered}")
                    reviewed_paths.add(path)
                if any(line.startswith("обрезано: показано") for line in raw.splitlines()):
                    warnings.append(path + ": diff обрезан сервером")
                full = await read_file(path, min(cfg.REVIEW_FILE_MAX_CHARS,
                                                 cfg.REVIEW_FILES_TOTAL_MAX_CHARS - full_chars))
                if full is None:
                    file["note"] += " · только diff, полный файл не поместился"
                    warnings.append(path + ": полный файл не поместился, доступен инструментом")
                else:
                    reviewed_paths.add(path)
                    full_chars += len(full)
                    full_files.append(f"Файл: {path}\n{numbered_text(full)}")
            rules, rule_cache = {}, {}

            async def rule(path):
                if path not in rule_cache:
                    rule_cache[path] = await read_file(path, cfg.REVIEW_RULES_MAX_CHARS, optional=True)
                    # Отличаем отсутствующий файл от существующего слишком большого.
                    # Отсутствие уже отражено ошибкой MCP, предупреждение не требуется.
                if rule_cache[path] is not None:
                    rules[path] = rule_cache[path]
                    return True
                return False

            for name in cfg.REVIEW_RULES_FILES:
                await rule(name)
                for file in files:
                    for parent in PurePosixPath(file["path"]).parents:
                        if str(parent) == ".":
                            break
                        if await rule(str(parent / name)):
                            break
            search_started = time.monotonic()
            results = {"docs": [], "code": []}
            changed = {file["path"] for file in files}
            for query in queries:
                for kind, kinds in (("docs", ("current", "docs")), ("code", ("code",))):
                    found = index.search(query, kinds=kinds, exclude_docs=changed)
                    results[kind].append(found)
                    log(f"поиск {kind} {query[:80]!r} · {found.elapsed:.2f} с · реранк {found.rerank_elapsed:.2f} с")
                    if found.rerank_error:
                        warning = "реранкер: ближайшие векторы вместо реранка: " + found.rerank_error
                        log("⚠️ " + warning)
                        warnings.append(warning)
                    for hit in found.hits:
                        score = "—" if hit.score is None else f"{hit.score:.3f}"
                        log(f"sim={hit.sim:.3f} score={score} до реранка #{hit.rank} · {hit.label()}")
            docs = merge_hits(results["docs"], cfg.REVIEW_DOCS_TOP_K)
            code = merge_hits(results["code"], cfg.REVIEW_CODE_TOP_K)
            hits = docs + code
            search_elapsed = time.monotonic() - search_started
            for number, hit in enumerate(hits, 1):
                log(f"объединение [{number}] {hit.label()}")
            if len(args.body) > cfg.REVIEW_BODY_MAX_CHARS:
                warnings.append("описание PR обрезано")
            for warning in warnings:
                log("⚠️ " + warning)
            user = cfg.REVIEW_USER_TEMPLATE.format(
                repo=args.repo.name, base_short=base[:12], head_short=head[:12], head=head,
                title=args.title, body=args.body[:cfg.REVIEW_BODY_MAX_CHARS],
                files="\n".join(f"{f['status']} {f['path']} · {f['note']}" for f in files),
                rules="\n\n".join(f"{path}:\n{body}" for path, body in rules.items()) or "Не найдены.",
                excerpts="\n\n".join(f"[{n}] {hit.label()}\n{hit.text}" for n, hit in enumerate(hits, 1)),
                full_files="\n\n".join(full_files), diffs="\n\n".join(diffs))
            log(f"контекст: diff {diff_chars} · полные файлы {full_chars} · сообщение {len(user)} символов")
            catalog = await session.tools()
            if session.error:
                raise RuntimeError(session.error)
            tools = [{"type": "function", "function": {"name": tool["name"],
                      "description": tool["description"], "parameters": tool["inputSchema"]}}
                     for tool in catalog if tool["name"] in cfg.REVIEW_MODEL_TOOLS]
            if {tool["function"]["name"] for tool in tools} != set(cfg.REVIEW_MODEL_TOOLS):
                raise RuntimeError("MCP: git_show_file отсутствует в каталоге")
            log("каталог модели: " + ", ".join(tool["function"]["name"] for tool in tools))
            calls = []

            async def model_tool(name, arguments):
                calls.append(name)
                if name not in cfg.REVIEW_MODEL_TOOLS:
                    return ToolCall(False, "", 0, "инструмент не разрешён")
                result = await call(name, arguments, optional=True)
                if not result.ok and not result.text:
                    raise RuntimeError(result.error)
                if result.ok:
                    header, _, body = result.text.partition("\n")
                    match = re.search(r"строки (\d+)–", header)
                    result.text = header + "\n" + numbered_text(body, int(match[1]) if match else 1)
                return result

            result = await answer([{"role": "system", "content": cfg.REVIEW_SYSTEM_PROMPT},
                                   {"role": "user", "content": user}], tools, model_tool, params)
            if not result.ok:
                raise RuntimeError("LLM: " + str(result.error))
            if result.finish_reason == "length":
                warnings.append("ответ достиг ограничения длины")
            model_elapsed = sum(round.elapsed for round in result.rounds)
            cost = "стоимость неизвестна" if result.cost_usd is None else f"≈ ${result.cost_usd:.6f}"
            call_label = ", ".join(f"{name} ×{count}" for name, count in Counter(calls).items()) or "нет"
            context = [f"base `{base[:12]}` · head `{head[:12]}` · файлов {len(files)} "
                       f"(рассмотрено {len(reviewed_paths)}) · diff {diff_chars} символов · полные файлы {full_chars}",
                       "правила: " + (", ".join(f"`{path}`" for path in rules) or "не найдены"),
                       f"выдержки: документация {len(docs)}, код {len(code)}",
                       f"модель {result.answered_model} · раундов {len(result.rounds)} · вызвано: {call_label} · "
                       f"{result.prompt_tokens} → {result.completion_tokens} токенов · {cost} · "
                       f"{time.monotonic() - started:.2f} с (поиск {search_elapsed:.2f} с, модель {model_elapsed:.2f} с)"]
            context.extend(f"{file['status']} `{file['path']}` · {file['note']}" for file in files)
            report, stats = render_review(result.text, changed=changed, ranges=ranges,
                                          source_labels=[hit.label() for hit in hits], context_lines=context,
                                          warnings=warnings, link_base=args.link_base,
                                          max_findings=cfg.REVIEW_MAX_FINDINGS)
        log(f"разбор: {json.dumps(stats, ensure_ascii=False)}")
    if session.error:
        raise RuntimeError(session.error)
    return report


def write_report(path, report):
    path = path.expanduser().resolve()
    with tempfile.NamedTemporaryFile(mode="w", encoding="utf-8", dir=path.parent,
                                     prefix=path.name + ".", suffix=".tmp", delete=False) as stream:
        temporary = Path(stream.name)
        try:
            stream.write(report)
            stream.close()
            os.replace(temporary, path)
        finally:
            temporary.unlink(missing_ok=True)


def main():
    parser = argparse.ArgumentParser(description="AI-ревью ветки от общего предка base...head")
    parser.add_argument("--repo", type=Path, default=cfg.DEFAULT_REPO)
    parser.add_argument("--db", type=Path, default=cfg.INDEX_DB)
    parser.add_argument("--base", default="main")
    parser.add_argument("--head", default="HEAD")
    parser.add_argument("--title", default="")
    parser.add_argument("--body-file", type=Path)
    parser.add_argument("--link-base", default="")
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    args.repo, args.db = args.repo.expanduser().resolve(), args.db.expanduser().resolve()
    load_dotenv(cfg.WEEK7_DIR / ".env", override=False)
    key = os.environ.get("LLM_API_KEY", "").strip()
    if not key:
        log("отказ: нет LLM_API_KEY")
        return 2
    params = LlmParams(os.environ.get("LLM_BASE_URL", "").strip() or cfg.LLM_BASE_URL, key,
                       os.environ.get("LLM_MODEL", "").strip() or cfg.LLM_MODEL,
                       cfg.LLM_TEMPERATURE, cfg.REVIEW_MAX_TOKENS, cfg.LLM_MAX_TOOL_ROUNDS,
                       cfg.TOOL_RESULT_MAX_CHARS, cfg.PRICES, cfg.LLM_TIMEOUT_S, log_prefix="[review]")
    try:
        if not args.repo.is_dir():
            raise ValueError("--repo не существует или не каталог")
        args.body = args.body_file.read_text(encoding="utf-8") if args.body_file else ""
        index = DocsIndex(args.db, cfg.MODELS_DIR, rerank_model=cfg.RERANK_MODEL,
                          candidates=cfg.SEARCH_CANDIDATES, top_k=cfg.SEARCH_TOP_K)
    except Exception as exc:
        log(f"отказ: {exc}")
        return 2
    try:
        report = asyncio.run(review(args, index, params))
        if args.out:
            write_report(args.out, report)
            log(f"ревью записано: {args.out}")
        else:
            print("\n" + report)
        return 0
    except StartRefused as exc:
        log(f"отказ: {exc}")
        return 2
    except KeyboardInterrupt:
        log("прервано; прежний отчёт сохранён")
        return 130
    except Exception as exc:
        log(f"сбой: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
