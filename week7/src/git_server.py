"""Отдельный MCP-сервер git: только чтение, stdout занят JSON-RPC."""

import argparse
import json
import logging
import os
from pathlib import Path, PureWindowsPath
import subprocess
import sys
import time
from typing import Annotated

from mcp.server.mcpserver import MCPServer
from mcp.server.mcpserver.exceptions import ToolError
from pydantic import Field

CHANGED_MAX_FILES = 200
RANGE_DIFF_MAX_CHARS = 40_000
SHOW_MAX_LINES = 400
SHOW_MAX_CHARS = 30_000
STATUS_MAX_LINES = 100
DIFF_MAX_CHARS = 12000
GIT_TIMEOUT_S = 10
BRANCH_DESCRIPTION = "Узнать текущую git-ветку, upstream и число коммитов впереди/позади."
STATUS_DESCRIPTION = "Что изменено сейчас: файлы в индексе, рабочем дереве и неотслеживаемые. Вызывай при вопросах о текущих изменениях."
LOG_DESCRIPTION = "Узнать, над чем работали в последних коммитах: дата, автор, тема и теги. Вызывай при вопросах об истории разработки."
DIFF_DESCRIPTION = "Показать изменения: статистика и diff всего рабочего дерева или конкретного относительного файла. staged=true — изменения в индексе."


def make_server(repo: Path):
    server = MCPServer("git-сервер", log_level="WARNING")

    def git(*args, allow_failure=False):
        started = time.monotonic()
        try:
            result = subprocess.run(
                ["git", "-C", str(repo), "--no-pager", "--literal-pathspecs", *args],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=GIT_TIMEOUT_S,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"},
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ToolError(f"git: {exc}") from exc
        finally:
            print(f"[git-сервер] {args[0]} · {time.monotonic() - started:.3f} с", file=sys.stderr, flush=True)
        if result.returncode and not allow_failure:
            raise ToolError(result.stderr.strip() or f"git завершился с кодом {result.returncode}")
        return result.stdout if not result.returncode else None

    def validate_path(path):
        relative = Path(path)
        if (not path or "\0" in path or relative.is_absolute() or PureWindowsPath(path).is_absolute()
                or ".." in relative.parts or ".." in PureWindowsPath(path).parts
                or not (repo / relative).resolve().is_relative_to(repo)):
            raise ToolError("path должен быть относительным, без '..' и внутри репозитория")

    def revision(ref):
        if not ref or ref.startswith("-") or "\0" in ref or any(char.isspace() for char in ref):
            raise ToolError("недопустимая ревизия")
        sha = git("rev-parse", "--verify", "--end-of-options", ref + "^{commit}", allow_failure=True)
        if not sha:
            raise ToolError(f"ревизия не найдена: {ref}")
        return sha.strip()

    def revisions(base, head):
        root = git("rev-parse", "--show-toplevel").strip()
        if Path(root).resolve() != repo:
            raise ToolError("--repo должен быть корнем git-репозитория")
        return revision(base), revision(head)

    @server.tool(description="Список файлов PR от общего предка: статус, пути, число добавленных и удалённых строк.")
    def git_changed_files(base: str, head: str) -> str:
        base, head = revisions(base, head)
        common = git("merge-base", base, head).strip()
        options = ("--no-ext-diff", "--no-textconv", "-M")
        names = iter(git("diff", *options, "--name-status", "-z", f"{base}...{head}", "--").split("\0"))
        files = []
        for status in names:
            if not status:
                continue
            path = next(names)
            old = None
            if status[0] in "RC":
                old, path = path, next(names)
            files.append((status[0], old, path))
        stats = {}
        entries = iter(git("diff", *options, "--numstat", "-z", f"{base}...{head}", "--").split("\0"))
        for entry in entries:
            if not entry:
                continue
            added, deleted, path = entry.split("\t", 2)
            if not path:
                next(entries)  # Старое имя при переименовании.
                path = next(entries)
            stats[path] = (added, deleted)
        lines = [f"base {base[:12]} · head {head[:12]} · merge-base {common[:12]}",
                 "ревизии: " + json.dumps({"base": base, "head": head, "merge_base": common})]
        for status, old, path in files[:CHANGED_MAX_FILES]:
            added, deleted = stats.get(path, ("0", "0"))
            name = json.dumps(path, ensure_ascii=False)
            if old is not None:
                name = json.dumps(old, ensure_ascii=False) + " → " + name
            count = "двоичный" if added == "-" else f"+{added} −{deleted}"
            lines.append(f"{status}\t{name}\t{count}")
        if len(files) > CHANGED_MAX_FILES:
            lines.append(f"и ещё {len(files) - CHANGED_MAX_FILES}")
        return "\n".join(lines)

    @server.tool(description="Diff одного файла PR от общего предка, с ограниченным контекстом и объёмом.")
    def git_range_diff(base: str, head: str, path: str,
                       context: Annotated[int, Field(ge=0, le=50)] = 10) -> str:
        validate_path(path)
        base, head = revisions(base, head)
        # Оба имени нужны, чтобы diff переименования содержал удалённые строки старого пути.
        old_paths = []
        names = iter(git("diff", "--no-ext-diff", "--no-textconv", "-M", "--name-status", "-z",
                         f"{base}...{head}", "--").split("\0"))
        for status in names:
            if not status:
                continue
            old = next(names)
            if status[0] in "RC":
                new = next(names)
                if new == path:
                    validate_path(old)
                    old_paths.append(old)
        text = git("diff", "--no-ext-diff", "--no-textconv", "-M", f"-U{context}",
                   f"{base}...{head}", "--", *old_paths, path)
        if len(text) > RANGE_DIFF_MAX_CHARS:
            reserve = 100
            cut = max(0, text.rfind("\n", 0, RANGE_DIFF_MAX_CHARS - reserve))
            return text[:cut] + f"\nобрезано: показано {cut} из {len(text)} символов"
        return text

    @server.tool(description="Прочитать файл проекта в ревизии PR целиком или диапазоном строк. "
                 "Вызывай, когда для замечания нужно увидеть код вне diff: вызывающий код, определение "
                 "функции, константы. ref — sha головы PR из сообщения. Ответ ограничен 400 строками "
                 "и 30000 символами; продолжение читай следующим диапазоном.")
    def git_show_file(ref: str, path: str,
                      start_line: Annotated[int, Field(ge=1)] = 1,
                      end_line: Annotated[int | None, Field(ge=1)] = None) -> str:
        validate_path(path)
        sha = revision(ref)
        if end_line is not None and end_line < start_line:
            raise ToolError("end_line должен быть не меньше start_line")
        kind = git("cat-file", "-t", f"{sha}:{path}", allow_failure=True)
        if kind is None:
            raise ToolError(f"файл отсутствует в ревизии: {path}")
        if kind.strip() != "blob":
            raise ToolError("path должен указывать на файл")
        text = git("show", "--no-ext-diff", "--no-textconv", f"{sha}:{path}", "--")
        if "\0" in text:
            raise ToolError("двоичный файл")
        lines = text.splitlines(keepends=True)
        if start_line > len(lines) and lines:
            raise ToolError("start_line за концом файла")
        end = min(len(lines), end_line or len(lines), start_line + SHOW_MAX_LINES - 1)
        chosen = []
        size = 0
        for line in lines[start_line - 1:end]:
            if size + len(line) > max(0, SHOW_MAX_CHARS - len(path) - 150):
                break
            chosen.append(line)
            size += len(line)
        shown_end = start_line + len(chosen) - 1
        header = f"{path} @ {sha[:12]}, строки {start_line}–{shown_end} из {len(lines)}"
        # Даже одна строка длиннее потолка возвращается с явной пометкой, без потери границы.
        suffix = "\nобрезано: продолжение не поместилось" if shown_end < end else ""
        return header + "\n" + "".join(chosen) + suffix

    @server.tool(description=BRANCH_DESCRIPTION)
    def git_branch() -> str:
        sha = git("rev-parse", "--verify", "HEAD").strip()
        branch = git("symbolic-ref", "--quiet", "--short", "HEAD", allow_failure=True)
        if not branch:
            return f"detached HEAD at {sha[:12]}"
        upstream = git("rev-parse", "--abbrev-ref", "--symbolic-full-name", "@{upstream}", allow_failure=True)
        if not upstream:
            return branch.strip()
        ahead, behind = git("rev-list", "--left-right", "--count", "HEAD...@{upstream}").split()
        return f"{branch.strip()} · upstream {upstream.strip()} · впереди/позади на {ahead}/{behind}"

    @server.tool(description=STATUS_DESCRIPTION)
    def git_status() -> str:
        raw = git("status", "--porcelain=v1", "-z", "--untracked-files=all")
        entries = iter(raw.split("\0"))
        lines = []
        for entry in entries:
            if not entry:
                continue
            status, path = entry[:2], entry[3:]
            if "R" in status or "C" in status:
                original = next(entries, "")
                path = f"{original} → {path}"
            if any(ord(char) < 32 for char in path):
                path = json.dumps(path, ensure_ascii=False)
            if status == "??":
                lines.append(f"неотслеживаемый: {path}")
            else:
                labels = []
                if status[0] != " ":
                    labels.append(f"индекс {status[0]}")
                if status[1] != " ":
                    labels.append(f"рабочее дерево {status[1]}")
                lines.append(f"{', '.join(labels)}: {path}")
        shown = STATUS_MAX_LINES - 1 if len(lines) > STATUS_MAX_LINES else STATUS_MAX_LINES
        remaining = len(lines) - shown
        return "\n".join(lines[:shown] + ([f"и ещё {remaining}"] if remaining > 0 else [])) or "Рабочее дерево чистое"

    @server.tool(description=LOG_DESCRIPTION)
    def git_log(limit: Annotated[int, Field(ge=1, le=30)] = 10) -> str:
        return git("log", f"-{limit}", "--date=short", "--format=%h | %ad | %an | %s | %d").strip()

    @server.tool(description=DIFF_DESCRIPTION)
    def git_diff(staged: bool = False, path: str | None = None) -> str:
        if path is not None:
            validate_path(path)
        options = ["--cached"] if staged else []
        files = [path] if path is not None else []
        stat = git("diff", "--no-ext-diff", "--no-textconv", *options, "--stat", "--", *files)
        diff = git("diff", "--no-ext-diff", "--no-textconv", *options, "--", *files)
        text = stat + "\n" + diff if stat or diff else "Изменений нет"
        if len(text) > DIFF_MAX_CHARS:
            marker = f"\nобрезано: показано {DIFF_MAX_CHARS} из {len(text)} символов"
            cut = text.rfind("\n", 0, DIFF_MAX_CHARS - len(marker) + 1)
            cut = max(cut, 0)
            return text[:cut] + f"\nобрезано: показано {cut} из {len(text)} символов"
        return text.strip()

    return server


def main():
    parser = argparse.ArgumentParser(description="MCP-сервер чтения git")
    parser.add_argument("--repo", type=Path, required=True)
    args = parser.parse_args()
    logging.basicConfig(stream=sys.stderr, format="[git-сервер] %(message)s", force=True)
    make_server(args.repo.expanduser().resolve()).run(transport="stdio")


if __name__ == "__main__":
    main()
