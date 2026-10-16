"""Отдельный MCP-сервер git: только чтение, stdout занят JSON-RPC."""

import argparse
import fnmatch
import json
import logging
import os
from pathlib import Path, PureWindowsPath
import subprocess
import stat
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
FILES_MAX = 300
GREP_MAX_MATCHES = 100
GREP_LINE_MAX_CHARS = 200
READ_MAX_LINES = 300
READ_MAX_CHARS = 12000
GIT_TIMEOUT_S = 10
BRANCH_DESCRIPTION = "Узнать текущую git-ветку, upstream и число коммитов впереди/позади."
STATUS_DESCRIPTION = "Что изменено сейчас: файлы в индексе, рабочем дереве и неотслеживаемые. Вызывай при вопросах о текущих изменениях."
LOG_DESCRIPTION = "Узнать, над чем работали в последних коммитах: дата, автор, тема и теги. Вызывай при вопросах об истории разработки."
DIFF_DESCRIPTION = "Показать изменения: статистика и diff всего рабочего дерева или конкретного относительного файла. staged=true — изменения в индексе."


def make_server(repo: Path):
    server = MCPServer("git-сервер", log_level="WARNING")

    def git(*args, allow_failure=False, ok_codes=(0,)):
        started = time.monotonic()
        try:
            # check-ignore сам принимает имена; глобальная literal-magic ему не подходит.
            result = subprocess.run(
                ["git", "-C", str(repo), "--no-pager",
                 *([] if args[0] == "check-ignore" else ["--literal-pathspecs"]), *args],
                capture_output=True, text=True, encoding="utf-8", errors="replace",
                timeout=GIT_TIMEOUT_S,
                env={**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"},
            )
        except (OSError, subprocess.TimeoutExpired) as exc:
            raise ToolError(f"git: {exc}") from exc
        finally:
            print(f"[git-сервер] {args[0]} · {time.monotonic() - started:.3f} с", file=sys.stderr, flush=True)
        if result.returncode not in ok_codes and not allow_failure:
            raise ToolError(result.stderr.strip() or f"git завершился с кодом {result.returncode}")
        return result.stdout if result.returncode in ok_codes else None

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

    def project_list(glob=None):
        paths = sorted(set(git("ls-files", "-co", "--exclude-standard", "-z").split("\0")) - {""})
        return [path for path in paths if ((repo / path).is_file() or (repo / path).is_symlink())
                and (glob is None or fnmatch.fnmatchcase(path, glob))]

    @server.tool(description="Список файлов проекта: отслеживаемые git и новые, без игнорируемых. "
                 "glob — шаблон пути, например week7/src/*.py или *.md; без него — все файлы. "
                 "Вызывай, чтобы узнать, какие файлы есть.")
    def project_files(glob: str | None = None) -> str:
        paths = project_list(glob)
        lines = [path + (" (символьная ссылка)" if (repo / path).is_symlink() else "")
                 for path in paths[:FILES_MAX]]
        if len(paths) > FILES_MAX:
            lines.append(f"и ещё {len(paths) - FILES_MAX}")
        return "\n".join([*lines, f"всего {len(paths)}"])

    @server.tool(description="Найти строки в файлах проекта: путь, номер строки, текст. pattern — регулярное "
                 "выражение, fixed=true — точная строка; glob — в каких файлах искать. Вызывай, чтобы найти "
                 "все места использования имени, импорта, константы. Показывает до 100 совпадений.")
    def project_grep(pattern: str, glob: str | None = None, fixed: bool = False) -> str:
        paths = project_list(glob)
        if not paths:
            return "совпадений нет"
        if "\0" in pattern:
            raise ToolError("pattern содержит нулевой байт")
        # Символьные ссылки git grep сам не читает; код 1 означает отсутствие совпадений.
        raw = git("grep", "-n", "-I", "--null", "--no-color", "--untracked",
                  "-F" if fixed else "-E", "-e", pattern, "--", *paths, ok_codes=(0, 1))
        matches = []
        for line in raw.splitlines():
            path, number, text = line.split("\0", 2)
            matches.append(f"{path}:{number}: {text[:GREP_LINE_MAX_CHARS]}" +
                           ("…" if len(text) > GREP_LINE_MAX_CHARS else ""))
        suffix = [f"обрезано: и ещё {len(matches) - GREP_MAX_MATCHES}"] if len(matches) > GREP_MAX_MATCHES else []
        return "\n".join(matches[:GREP_MAX_MATCHES] + suffix) or "совпадений нет"

    @server.tool(description="Прочитать файл проекта из рабочего дерева целиком или диапазоном строк, с номерами "
                 "строк. Ответ ограничен 300 строками и 12000 символами; продолжение читай следующим диапазоном. "
                 "Номера строк — не часть файла.")
    def project_read(path: str, start_line: Annotated[int, Field(ge=1)] = 1,
                     end_line: Annotated[int | None, Field(ge=1)] = None) -> str:
        validate_path(path)
        paths = project_list()
        if path not in paths:
            raise ToolError("файл отсутствует в списке проекта (возможно, игнорируется git)")
        # Ссылка внутри проекта не должна открывать игнорируемые файлы через другое имя.
        resolved = (repo / path).resolve().relative_to(repo).as_posix()
        if resolved not in paths:
            raise ToolError("цель ссылки отсутствует в списке проекта")
        if end_line is not None and end_line < start_line:
            raise ToolError("end_line должен быть не меньше start_line")
        try:
            with (repo / path).open(encoding="utf-8", newline="") as stream:
                text = stream.read()
        except (OSError, UnicodeError) as exc:
            raise ToolError(f"нельзя прочитать UTF-8: {exc}") from exc
        if "\0" in text:
            raise ToolError("двоичный файл")
        lines = text.splitlines()
        if start_line > len(lines) and lines:
            raise ToolError("start_line за концом файла")
        end = min(len(lines), end_line or len(lines), start_line + READ_MAX_LINES - 1)
        chosen, size = [], 0
        for n in range(start_line, end + 1):
            line = f"{n}│{lines[n - 1]}\n"
            if size + len(line) > max(0, READ_MAX_CHARS - len(path) - 200):
                break
            chosen.append(line)
            size += len(line)
        shown = start_line + len(chosen) - 1
        suffix = f"\nобрезано: продолжение со строки {shown + 1}" if shown < (end_line or len(lines)) and shown < len(lines) else ""
        return f"{path}, строки {start_line}–{shown} из {len(lines)}\n" + "".join(chosen) + suffix

    @server.tool(description="Статус пути для проверки предложения и применения правок; служебный инструмент.")
    def project_path_status(path: str) -> str:
        validate_path(path)
        current, symlink = repo, False
        for part in Path(path).parts:
            current /= part
            try:
                symlink = symlink or stat.S_ISLNK(os.lstat(current).st_mode)
            except FileNotFoundError:
                pass
        tracked = git("ls-files", "--error-unmatch", "--", path, allow_failure=True) is not None
        ignored = git("check-ignore", "-q", "--", path, allow_failure=True) is not None
        status = "ignored" if ignored else "tracked" if tracked else "untracked" if (repo / path).exists() else "absent"
        return json.dumps({"status": status, "symlink": symlink})

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
