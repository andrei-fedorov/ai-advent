"""Нарезка кода блоками и пустыми строками; без парсера, диска и моделей."""

import re

from md_chunks import Chunk


def chunk_code(text: str, doc: str, max_chars: int = 1200,
               def_re: str = r"^\s+(?:async\s+def|def|class|fun|func|struct|enum|interface|object)\s+") -> list[Chunk]:
    if max_chars < 1:
        raise ValueError("Предел куска должен быть положительным")
    blocks, block = [], []
    previous_blank = True
    for number, line in enumerate(text.splitlines(), 1):
        if line and not line[0].isspace() and line[0] not in ")]}" and previous_blank and block:
            blocks.append(block)
            block = []
        block.append((number, line))
        previous_blank = not line.strip()
    if block:
        blocks.append(block)
    result = []
    for block in blocks:
        meaningful = [line.strip() for _, line in block if line.strip()
                      and not line.lstrip().startswith(("@", "#", "//", "/*", "*"))]
        section = (meaningful or [line.strip() for _, line in block if line.strip()] or [""])[0][:120]
        # Сначала абзацы, затем строки; отступы и фактические номера сохраняются.
        paragraphs, paragraph = [], []
        for pair in block:
            if not pair[1].strip():
                if paragraph:
                    paragraphs.append(paragraph)
                    paragraph = []
            else:
                paragraph.append(pair)
        if paragraph:
            paragraphs.append(paragraph)
        atoms = []
        for paragraph in paragraphs:
            body = "\n".join(line for _, line in paragraph)
            if len(body) <= max_chars:
                atoms.append((paragraph[0][0], body))
            else:
                for number, line in paragraph:
                    for offset in range(0, len(line), max_chars):
                        atoms.append((number, line[offset:offset + max_chars]))
        pieces = []
        body, start, last_end = "", 0, 0
        for number, atom in atoms:
            separator = "\n" * max(1, number - last_end)
            joined = body + separator + atom if body else atom
            if body and len(joined) > max_chars:
                pieces.append((start, body))
                body = ""
            if not body:
                start, body = number, atom
            else:
                body = joined
            last_end = number + atom.count("\n")
        if body:
            pieces.append((start, body))
        for part, (line, body) in enumerate(pieces, 1):
            first = body.splitlines()[0]
            label = section + " › " + first.strip()[:120] if re.match(def_re, first) else section
            result.append(Chunk(doc, label, part, len(pieces), line, body))
    return result
