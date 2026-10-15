"""Чистые функции: номера строк, разбор JSON, проверки и Markdown ревью."""

import json
import re
from urllib.parse import quote

MARKER = "<!-- week7-ai-review -->"
CATEGORIES = {"bug": "Потенциальные баги", "architecture": "Архитектурные проблемы",
              "recommendation": "Рекомендации"}
HUNK_RE = re.compile(r"^@@ -\d+(?:,\d+)? \+(\d+)(?:,\d+)? @@(.*)")
SHOW_HEADER_RE = re.compile(r"^(.* @ \S+, строки )(\d+)–(\d+)( из \d+)$")
SHOW_TRUNCATED = "\nобрезано: продолжение не поместилось"
RENDERED_SUFFIXES = (".md", ".markdown")


def numbered_text(text: str, start_line: int = 1) -> str:
    return "\n".join(f"{number:5} | {line}" for number, line in enumerate(text.splitlines(), start_line))


def numbered_show_result(text: str, max_chars: int) -> str:
    """Нумерует ответ git_show_file и укладывает его в max_chars целыми строками с верным заголовком."""
    header, _, body = text.partition("\n")
    match = SHOW_HEADER_RE.match(header)
    if not match:
        return header + "\n" + numbered_text(body)
    first, last = int(match[2]), int(match[3])
    cut = body.endswith(SHOW_TRUNCATED)
    lines = numbered_text(body.removesuffix(SHOW_TRUNCATED), first).splitlines()
    kept, size = [], len(header) + 100
    for line in lines:
        # Хотя бы одна строка: сверхдлинную обрежет уже llm.py.
        if kept and size + len(line) + 1 > max_chars:
            break
        kept.append(line)
        size += len(line) + 1
    if len(kept) < len(lines):
        last, cut = first + len(kept) - 1, True
    note = f"\nобрезано: продолжение читай со строки {last + 1}" if cut else ""
    return f"{match[1]}{first}–{last}{match[4]}\n" + "\n".join(kept) + note


def numbered_diff(text: str) -> tuple[str, list[tuple[int, int]]]:
    """Возвращает diff и диапазоны именно добавленных строк новой версии."""
    result, ranges = [], []
    number = None
    for line in text.splitlines():
        hunk = HUNK_RE.match(line)
        if hunk:
            number = int(hunk[1])
            result.append(line)
        elif number is not None and line.startswith(("+", " ")):
            result.append(f"{number:5} |{line}")
            if line.startswith("+"):
                if ranges and ranges[-1][1] == number - 1:
                    ranges[-1] = (ranges[-1][0], number)
                else:
                    ranges.append((number, number))
            number += 1
        elif number is not None and line.startswith("-"):
            result.append(f"      |{line}")
        else:
            result.append(line)
    return "\n".join(result), ranges


def search_query(path: str, diff: str, max_chars: int) -> str:
    parts = [path]
    for line in diff.splitlines():
        hunk = HUNK_RE.match(line)
        if hunk and hunk[2].strip():
            parts.append(hunk[2].strip())
        elif line.startswith("+") and not line.startswith("+++"):
            parts.append(line[1:])
    return "\n".join(parts)[:max_chars]


def parse_response(text: str) -> dict | None:
    def integer(value):
        # Номер строкой («41») модель пишет часто; это не повод терять всё ревью.
        if isinstance(value, str) and re.fullmatch(r"\d+", value.strip(), re.ASCII):
            return int(value)
        return value

    def normalized(value):
        if not isinstance(value, dict) or not isinstance(value.get("summary"), str) or not isinstance(value.get("findings"), list):
            return None
        findings = []
        for finding in value["findings"]:
            if not isinstance(finding, dict):
                return None
            if not all(isinstance(finding.get(key), str) for key in
                       ("category", "severity", "file", "title", "details")):
                return None
            # null — замечание ко всему файлу; показывается с пометкой.
            line = integer(finding.get("line"))
            if line is not None and type(line) is not int:
                return None
            sources = finding.get("sources")
            sources = [] if sources is None else sources
            if not isinstance(sources, list):
                return None
            sources = [integer(source) for source in sources]
            if any(type(source) is not int for source in sources):
                return None
            findings.append({**finding, "line": line, "sources": sources})
        return {**value, "findings": findings}
    try:
        value = json.loads(text)
    except (ValueError, TypeError):
        start = text.find("{")
        if start < 0:
            return None
        try:
            value, _ = json.JSONDecoder().raw_decode(text[start:])
        except ValueError:
            return None
    return normalized(value)


def finding_warnings(finding: dict, changed: set[str], ranges: dict, source_count: int) -> list[str]:
    warnings = []
    if finding["category"] not in CATEGORIES:
        warnings.append("категория: " + finding["category"])
    if finding["file"] not in changed:
        warnings.append("файла нет в PR")
    elif finding["line"] is None:
        warnings.append("строка не указана")
    elif not any(low <= finding["line"] <= high for low, high in ranges.get(finding["file"], ())):
        warnings.append("строка вне изменений PR")
    warnings.extend(f"[{number}] — такой выдержки не было" for number in finding["sources"]
                    if not 1 <= number <= source_count)
    return warnings


def safe_text(text: str) -> str:
    # Модель не должна закрывать служебный details или создавать вторую метку бота.
    text = re.sub(r"</details\s*>", "&lt;/details&gt;", text, flags=re.IGNORECASE)
    return text.replace(MARKER, "&lt;!-- week7-ai-review --&gt;")


def render_review(raw: str, *, changed: set[str], ranges: dict, source_labels: list[str],
                  context_lines: list[str], warnings=(), link_base: str = "", max_findings: int = 15) -> tuple[str, dict]:
    response = parse_response(raw)
    stats = {key: 0 for key in CATEGORIES}
    stats.update(warnings=0, json=response is not None)
    lines = [MARKER, "## 🤖 AI-ревью", ""]
    if response is None:
        lines.extend(["⚠️ ответ модели не в формате JSON — показан как есть", "", safe_text(raw), ""])
        stats["warnings"] += 1
        findings = []
    else:
        lines.extend([safe_text(response["summary"]), ""])
        findings = response["findings"][:max_findings]
    for category, heading in CATEGORIES.items():
        lines.extend([f"### {heading}", ""])
        group = [finding for finding in findings
                 if (finding["category"] if finding["category"] in CATEGORIES else "recommendation") == category]
        group.sort(key=lambda finding: {"высокая": 0, "средняя": 1, "низкая": 2}.get(finding["severity"], 3))
        stats[category] = len(group)
        if not group:
            lines.extend(["Не найдено.", ""])
        for index, finding in enumerate(group, 1):
            line = finding["line"]
            label = finding["file"] if line is None else f"{finding['file']}:{line}"
            location = f"`{label}`"
            if link_base:
                # Markdown GitHub показывает отрисованным, якорь строки работает только в ?plain=1.
                plain = "?plain=1" if finding["file"].lower().endswith(RENDERED_SUFFIXES) else ""
                anchor = "" if line is None else f"{plain}#L{line}"
                location = f"[{location}]({link_base.rstrip('/')}/{quote(finding['file'], safe='/')}{anchor})"
            lines.extend([safe_text(f"{index}. **{finding['severity']}** · {location} — {finding['title']}"),
                          "   " + safe_text(finding["details"]).replace("\n", "\n   ")])
            for warning in finding_warnings(finding, changed, ranges, len(source_labels)):
                lines.append("   ⚠️ " + safe_text(warning))
                stats["warnings"] += 1
            sources = [f"[{number}] `{source_labels[number - 1]}`" for number in finding["sources"]
                       if 1 <= number <= len(source_labels)]
            if sources:
                lines.append("   Источники: " + safe_text("; ".join(sources)))
            lines.append("")
    if response and len(response["findings"]) > max_findings:
        lines.extend([f"Ещё {len(response['findings']) - max_findings} замечаний не показано.", ""])
    lines.extend(["<details><summary>Контекст ревью</summary>", ""])
    lines.extend("- " + safe_text(line) for line in context_lines)
    lines.extend(f"- [{number}] {safe_text(label)}" for number, label in enumerate(source_labels, 1))
    lines.extend("- ⚠️ " + safe_text(warning) for warning in warnings)
    stats["warnings"] += len(warnings)
    lines.extend(["", "</details>", "", "_Ревью сделано моделью и не заменяет ревью человека._", ""])
    return "\n".join(lines), stats
