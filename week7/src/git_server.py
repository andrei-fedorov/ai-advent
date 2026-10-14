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
            relative = Path(path)
            if (not path or "\0" in path or relative.is_absolute() or PureWindowsPath(path).is_absolute()
                    or ".." in relative.parts or ".." in PureWindowsPath(path).parts
                    or not (repo / relative).resolve().is_relative_to(repo)):
                raise ToolError("path должен быть относительным, без '..' и внутри репозитория")
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
