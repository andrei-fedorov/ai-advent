"""Нарезка Markdown по заголовкам и абзацам, без диска и моделей."""

from dataclasses import dataclass
import re


@dataclass(frozen=True)
class Chunk:
    doc: str
    section: str
    part: int
    parts: int
    line: int
    text: str

    @property
    def embedding_text(self) -> str:
        return f"{self.doc} · {self.section}\n{self.text}"


def _fence(line: str, opened: str | None) -> str | None:
    match = re.match(r"^ {0,3}(`{3,}|~{3,})(.*)$", line.rstrip("\r\n"))
    if not match:
        return opened
    marker, tail = match.groups()
    if opened is None:
        if marker[0] == "`" and "`" in tail:
            return None
        return marker
    if marker[0] == opened[0] and len(marker) >= len(opened) and not tail.strip():
        return None
    return opened


def _parts(lines: list[tuple[int, str]], limit: int) -> list[tuple[int, str]]:
    # Пустые строки внутри ограждения не разделяют блок кода.
    paragraphs = []
    paragraph = []
    opened = None
    for number, line in lines:
        next_opened = _fence(line, opened)
        if opened is None and next_opened is not None and paragraph:
            paragraphs.append(paragraph)
            paragraph = []
        if not line.strip() and opened is None:
            if paragraph:
                paragraphs.append(paragraph)
                paragraph = []
        else:
            paragraph.append((number, line))
        if opened is not None and next_opened is None:
            paragraphs.append(paragraph)
            paragraph = []
        opened = next_opened
    if paragraph:
        paragraphs.append(paragraph)
    atoms = []
    for paragraph in paragraphs:
        body = "".join(line for _, line in paragraph).rstrip()
        if len(body) <= limit:
            atoms.append((paragraph[0][0], body))
            continue
        for number, line in paragraph:
            rest = line.rstrip("\r\n")
            if not rest:
                atoms.append((number, ""))
            while rest:
                cut = len(rest)
                if cut > limit:
                    cut = rest.rfind(" ", 0, limit + 1)
                    if cut <= 0:
                        cut = limit
                piece = rest[:cut]
                if piece.strip():
                    atoms.append((number, piece))
                rest = rest[cut:].lstrip(" ")
    result = []
    body = ""
    start = 0
    previous_end = 0
    for number, atom in atoms:
        separator = "\n\n" if number > previous_end + 1 else "\n"
        joined = body + separator + atom if body else atom
        if body and len(joined) > limit:
            result.append((start, body.rstrip()))
            body = ""
        if not body:
            start, body = number, atom
        else:
            body = joined
        previous_end = number + atom.count("\n")
    if body.strip():
        result.append((start, body.rstrip()))
    return result


def chunk_markdown(text: str, doc: str, max_chars: int = 1200) -> list[Chunk]:
    if max_chars < 1:
        raise ValueError("Предел куска должен быть положительным")
    sections = []
    headings = []
    section = ""
    body = []
    opened = None
    for number, line in enumerate(text.splitlines(keepends=True), 1):
        heading = re.match(r"^ {0,3}(#{1,6})[ \t]+(.+?)\s*$", line) if opened is None else None
        if heading:
            sections.append((section, body))
            body = []
            level = len(heading[1])
            title = re.sub(r"[ \t]+#+[ \t]*$", "", heading[2]).strip()
            while headings and headings[-1][0] >= level:
                headings.pop()
            headings.append((level, title))
            section = " › ".join(title for _, title in headings)
        else:
            body.append((number, line))
            opened = _fence(line, opened)
    sections.append((section, body))
    result = []
    for section, lines in sections:
        pieces = _parts(lines, max_chars)
        for part, (line, body) in enumerate(pieces, 1):
            result.append(Chunk(doc, section, part, len(pieces), line, body))
    return result
