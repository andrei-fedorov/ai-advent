# TooManyRules — очистка и нарезка документов для индекса правил (день 21,
# неделя 5).
#
# Лист графа: чистые функции и типы. Ни `pymupdf`, ни модели эмбеддингов, ни
# диска, ни логов, ни Too Many Bones. Строки PDF собирает `rag_index.py`
# (единственное место, где импортируется `pymupdf`) и отдаёт сюда списком
# `Line`; какие шрифты что значат в каком документе, приходит описанием
# `SourceDoc` из `presets.py` — ветвлений «для книги» и «для листа» в коде нет.
#
# Что здесь (спецификация дня 21, §2.3-§2.6, §6.1, §7):
# - описание документа (`SourceDoc`, `CleanRules`) и его проверка;
# - `clean()` — одна очищенная копия документа на обе стратегии: мусор
#   долой по правилам документа, строки → абзацы, заголовки и записи
#   глоссария помечены, выброшенное посчитано по правилам;
# - `units()` — разделы документа (у книги и путеводителя — по заголовкам,
#   у глоссария — записи «Название — действие»);
# - две стратегии нарезки: `fixed_chunks()` — окна фиксированного размера с
#   перекрытием, `structural_chunks()` — по разделам и записям;
# - `compare()` — числа для отчёта о сравнении стратегий, без суждений.
#
# Языкового здесь нет ничего (§2.6, п. 5): конец предложения — знаки
# препинания, перенос — `str.isalpha()`/`str.islower()`, размеры — в символах.
# Язык документа — поле описания, а список допустимых языков приходит
# параметром из `presets.py`.

import bisect
import re
import statistics
from dataclasses import dataclass, field

# Как документ режется по структуре (§2.5): книга и путеводитель — по
# разделам, лист навыков — по записям глоссария.
STRUCTURES = ("sections", "glossary")
STRATEGIES = ("fixed", "structural")

# Кегль в PDF дробный: у заголовков одной книги 13.0, 13.8 и 13.9. Пара
# «шрифт, кегль» из правил документа совпадает со строкой, если кегли
# расходятся не больше чем на столько пунктов.
HEADING_SIZE_TOLERANCE = 0.5

BULLETS = ("•",)
# Управляющие символы из PDF (§2.3): `\x07`/`\x08` — остатки табуляции и
# отточия, `�` — отточие оглавления; мягкий перенос и невидимые пробелы
# — туда же. Неразрывные и прочие пробелы не выбрасываются, а становятся
# обычным пробелом.
REMOVED_CHARS = "\x07\x08�­​﻿"
SENTENCE_END = ".!?…"
# Отчёт (§7): кусок «не оборван», если кончается на конец предложения или на
# двоеточие (дальше список). Хвостовые кавычки и скобки не мешают.
CLAUSE_END = SENTENCE_END + ":"
CLOSERS = "\"'»”’)]"
ENTRY_SEP = " — "
SECTION_JOIN = " · "
GROUP_JOIN = " › "

# Правила очистки — ключи счёта выброшенного (§2.3). Ключ правила «страницы»
# собирается из списка страниц документа («стр. 3–5, 31»).
DROP_SMALL = "мелкий кегль"
DROP_DIGITS = "строки из цифр"
DROP_MARKERS = "одинокие маркеры"
DROP_CONTROL = "управляющие символы"
DROP_HYPHEN = "переносы"

_DOC_KEY = re.compile(r"[a-z0-9]+(?:-[a-z0-9]+)*")
_LANG = re.compile(r"[a-z]{2}")
_SPACES = re.compile(r"\s+")
# Конец предложения внутри абзаца: знак, возможно закрывающие кавычки и
# скобки, затем пробел.
_SENTENCE = re.compile(r"[.!?…]+[\"'»”’)\]]*(?=\s)")


# --- Описание документа ------------------------------------------------------

@dataclass(frozen=True)
class CleanRules:
    """Правила очистки и структуры одного документа — данные, а не код
    (§2.6, п. 4): у английского издания другие шрифты и кегли.

    - `min_size` — строки мельче (по самому крупному фрагменту) выбрасываются;
    - `drop_pages` — страницы целиком (оглавление, схемы состава, указатель);
    - `heading_fonts` — пары «шрифт, кегль» строк-заголовков (кегль — с допуском
      `HEADING_SIZE_TOLERANCE`), шрифт строки — тот, которым набрано больше всего
      её символов;
    - `entry_font` — только у глоссария: строка, чей первый фрагмент набран
      этим шрифтом и в которой есть « — », начинает запись;
    - `default_group` — имя раздела (группы) до первого заголовка;
    - `columns` — пары «страница, x границы колонок»: на этих страницах блоки
      идут по колонкам (сначала левее границы, потом правее), в колонке —
      сверху вниз; на остальных страницах — в порядке PyMuPDF.
    """

    min_size: float
    drop_pages: tuple[int, ...] = ()
    heading_fonts: tuple[tuple[str, float], ...] = ()
    entry_font: str = ""
    default_group: str = ""
    columns: tuple[tuple[int, float], ...] = ()


@dataclass(frozen=True)
class SourceDoc:
    """Документ корпуса. Ключ — пара (`doc`, `lang`): одна книга в двух
    изданиях — два документа с одним `doc` (§2.6, п. 2). `path` — имя файла в
    каталоге источников; `publisher` → `game` → `edition` — три уровня из §4
    спецификации проекта."""

    doc: str
    lang: str
    path: str
    title: str
    publisher: str
    game: str
    edition: str
    structure: str
    rules: CleanRules

    def __post_init__(self) -> None:
        if not _DOC_KEY.fullmatch(self.doc):
            raise ValueError(f"{self.doc!r}: ключ документа — латиница, цифры и дефис (он входит в chunk_id)")
        if not _LANG.fullmatch(self.lang):
            raise ValueError(f"{self.doc}: язык {self.lang!r} — код ISO 639-1 из двух строчных букв")
        if self.structure not in STRUCTURES:
            raise ValueError(f"{self.doc}: структура {self.structure!r}, допустимо: {', '.join(STRUCTURES)}")
        if (self.structure == "glossary") != bool(self.rules.entry_font):
            raise ValueError(f"{self.doc}: шрифт записей (`entry_font`) задаётся у глоссария и только у него")
        for field_name in ("path", "title", "publisher", "game", "edition"):
            if not getattr(self, field_name).strip():
                raise ValueError(f"{self.doc}: пустое поле {field_name}")

    @property
    def key(self) -> str:
        return f"{self.doc}/{self.lang}"


def check_documents(docs: list[SourceDoc] | tuple[SourceDoc, ...], langs: tuple[str, ...]) -> None:
    """Язык каждого документа — из списка допустимых, пары (`doc`, `lang`) не
    повторяются. Зовётся при импорте `presets.py`: ошибка описания — ошибка
    импорта."""
    seen: set[tuple[str, str]] = set()
    for source in docs:
        if source.lang not in langs:
            raise ValueError(f"{source.doc}: язык {source.lang!r} вне списка {', '.join(langs)}")
        if (source.doc, source.lang) in seen:
            raise ValueError(f"документ {source.key} описан дважды")
        seen.add((source.doc, source.lang))


# --- Вход очистки ------------------------------------------------------------

@dataclass(frozen=True)
class Line:
    """Строка PDF, как её собрал `rag_index.py`. `font` — шрифт большинства
    символов строки, `size` — кегль самого крупного фрагмента, `first_font` —
    шрифт первого непустого фрагмента (по нему узнаётся запись глоссария:
    «Провокация» жирным, дальше обычным), `x`/`y` — левый верхний угол
    строки."""

    page: int
    block: int
    text: str
    font: str
    size: float
    first_font: str
    x: float
    y: float


# --- Очищенный документ ------------------------------------------------------

@dataclass(frozen=True)
class Paragraph:
    """Абзац очищенного текста: `kind` — `heading`, `entry` (запись глоссария,
    `name` — её название) или `text`; `start`/`end` — смещения в
    `CleanDoc.text`."""

    kind: str
    text: str
    start: int
    end: int
    page_from: int
    page_to: int
    name: str = ""


@dataclass(frozen=True)
class CleanDoc:
    source: SourceDoc
    text: str
    paragraphs: tuple[Paragraph, ...]
    # (смещение, страница): с этого места очищенного текста идёт эта
    # страница. По возрастанию смещений.
    page_marks: tuple[tuple[int, int], ...]
    pages: int
    chars_raw: int
    dropped: dict[str, int]
    # (правило, страница, текст) — для выгрузки `--dump`.
    dropped_items: tuple[tuple[str, int, str], ...]

    def page_at(self, offset: int) -> int:
        index = bisect.bisect_right(self.page_marks, (offset, float("inf"))) - 1
        return self.page_marks[max(index, 0)][1] if self.page_marks else 0


def pages_label(pages: tuple[int, ...]) -> str:
    """«стр. 3–5, 31» — ключ правила «страницы» в счёте выброшенного."""
    parts: list[str] = []
    ordered = sorted(set(pages))
    i = 0
    while i < len(ordered):
        j = i
        while j + 1 < len(ordered) and ordered[j + 1] == ordered[j] + 1:
            j += 1
        parts.append(str(ordered[i]) if i == j else f"{ordered[i]}–{ordered[j]}")
        i = j + 1
    return "стр. " + ", ".join(parts)


def _is_heading(line: Line, rules: CleanRules) -> bool:
    return any(
        line.font == font and abs(line.size - size) <= HEADING_SIZE_TOLERANCE
        for font, size in rules.heading_fonts
    )


def _starts_lower(text: str) -> bool:
    return bool(text) and text[0].isalpha() and text[0].islower()


def _ends_clause(text: str) -> bool:
    """Текст кончается концом предложения или двоеточием; хвостовые
    пробелы, кавычки и скобки не мешают."""
    tail = text.rstrip().rstrip(CLOSERS).rstrip()
    return bool(tail) and tail[-1] in CLAUSE_END


def _continues(previous: str, text: str) -> bool:
    """Новый блок PDF продолжает прежний абзац, если начинается посреди
    фразы: со строчной буквы — или не с буквы и не с маркера списка, когда
    прежний абзац не кончился концом предложения («…сбросьте 1 трофей» +
    «(простой или ценный).»). Блоки PDF рвут абзацы на переходе колонки и
    у картинок."""
    if _starts_lower(text):
        return True
    if not text or text[0].isalpha() or text.startswith(BULLETS) or text[0] in "—–-":
        return False
    return not _ends_clause(previous)


def _reading_order(lines: list[Line], columns: dict[int, float]) -> list[Line]:
    """Строки блоками. Порядок блоков — как у PyMuPDF, кроме страниц из
    `columns`: там сначала блоки левее границы, потом правее, в колонке —
    сверху вниз. Положение блока — по его оставшимся после очистки строкам:
    номер страницы в одном блоке с текстом правой колонки иначе тянул бы
    блок в левую."""
    blocks: dict[tuple[int, int], list[Line]] = {}
    for line in lines:
        blocks.setdefault((line.page, line.block), []).append(line)
    keys = list(blocks)
    ordered: list[Line] = []
    i = 0
    while i < len(keys):
        page = keys[i][0]
        j = i
        while j < len(keys) and keys[j][0] == page:
            j += 1
        page_keys = keys[i:j]
        if page in columns:
            split = columns[page]

            def position(key: tuple[int, int]) -> tuple[bool, float, float]:
                x = min(line.x for line in blocks[key])
                y = min(line.y for line in blocks[key])
                return (x >= split, y, x)

            page_keys = sorted(page_keys, key=position)
        for key in page_keys:
            ordered.extend(blocks[key])
        i = j
    return ordered


# Строки заголовка одного блока — один заголовок, если набраны тем же шрифтом
# того же кегля («8. Навыки / запасного плана / (ЗП) злодеев»); другой кегль —
# уже следующий заголовок («4. Область навыков» 13.9, «Кубики навыков» 12.9).
_SAME_SIZE = 0.2


@dataclass
class _Draft:
    kind: str
    text: str
    pages: list[tuple[int, int]]  # (смещение в абзаце, страница)
    name: str = ""
    font: str = ""
    size: float = 0.0


def clean(source: SourceDoc, lines: list[Line], pages: int) -> CleanDoc:
    """Очищенная копия документа по его правилам (§2.3). Выброшенное
    считается в символах по правилам; абзац — строки одного блока через
    пробел, кроме трёх случаев: строка-заголовок — свой абзац (строки одного
    заголовка — одного шрифта и кегля); строка с маркером «•» — новый абзац;
    блок, который начинается посреди фразы, — продолжение прежнего абзаца
    (`_continues()`: блок PDF порвал фразу). У глоссария строки после записи
    дописываются к ней до следующей записи или заголовка."""
    rules = source.rules
    drop_pages = set(rules.drop_pages)
    pages_key = pages_label(rules.drop_pages) if rules.drop_pages else ""
    dropped: dict[str, int] = {}
    if pages_key:
        dropped[pages_key] = 0
    for key in (DROP_SMALL, DROP_DIGITS, DROP_MARKERS, DROP_CONTROL, DROP_HYPHEN):
        dropped[key] = 0
    items: list[tuple[str, int, str]] = []

    def drop(rule: str, line: Line, chars: int, text: str) -> None:
        dropped[rule] += chars
        items.append((rule, line.page, text))

    kept: list[Line] = []
    for line in lines:
        stripped = line.text.strip()
        if not stripped:
            continue
        if line.page in drop_pages:
            drop(pages_key, line, len(stripped), stripped)
            continue
        if line.size < rules.min_size:
            drop(DROP_SMALL, line, len(stripped), stripped)
            continue
        if stripped.isdigit():
            drop(DROP_DIGITS, line, len(stripped), stripped)
            continue
        if stripped in BULLETS:
            drop(DROP_MARKERS, line, len(stripped), stripped)
            continue
        removed = sum(line.text.count(char) for char in REMOVED_CHARS)
        text = line.text
        if removed:
            drop(DROP_CONTROL, line, removed, repr(stripped))
            text = text.translate({ord(char): None for char in REMOVED_CHARS})
        text = _SPACES.sub(" ", text).strip()
        if text:
            kept.append(Line(line.page, line.block, text, line.font, line.size, line.first_font, line.x, line.y))

    drafts: list[_Draft] = []
    current: _Draft | None = None
    previous_block: tuple[int, int] | None = None

    def start(kind: str, line: Line, name: str = "") -> _Draft:
        draft = _Draft(kind, line.text, [(0, line.page)], name, line.font, line.size)
        drafts.append(draft)
        return draft

    def append(draft: _Draft, line: Line) -> None:
        text = line.text
        head = draft.text
        if len(head) >= 2 and head.endswith("-") and head[-2].isalpha() and _starts_lower(text):
            dropped[DROP_HYPHEN] += 1
            items.append((DROP_HYPHEN, line.page, f"{head[-20:]} + {text[:20]}"))
            draft.text = head[:-1]
        else:
            draft.text = head + " "
        if draft.pages[-1][1] != line.page:
            draft.pages.append((len(draft.text), line.page))
        draft.text += text

    for line in _reading_order(kept, dict(rules.columns)):
        block = (line.page, line.block)
        new_block = block != previous_block
        previous_block = block
        if _is_heading(line, rules):
            if (
                current is not None and current.kind == "heading" and not new_block
                and line.font == current.font and abs(line.size - current.size) <= _SAME_SIZE
            ):
                append(current, line)
            else:
                current = start("heading", line)
        elif rules.entry_font and line.first_font == rules.entry_font and ENTRY_SEP in line.text:
            current = start("entry", line, line.text.split(ENTRY_SEP, 1)[0].strip())
        elif line.text.startswith(BULLETS):
            current = start("text", line)
        elif current is None or current.kind == "heading":
            current = start("text", line)
        elif current.kind == "entry":
            append(current, line)
        elif new_block and not _continues(current.text, line.text):
            current = start("text", line)
        else:
            append(current, line)

    paragraphs: list[Paragraph] = []
    marks: list[tuple[int, int]] = []
    pieces: list[str] = []
    offset = 0
    for draft in drafts:
        if pieces:
            pieces.append("\n\n")
            offset += 2
        for local, page in draft.pages:
            if not marks or marks[-1][1] != page:
                marks.append((offset + local, page))
        paragraphs.append(Paragraph(
            kind=draft.kind, text=draft.text, start=offset, end=offset + len(draft.text),
            page_from=draft.pages[0][1], page_to=draft.pages[-1][1], name=draft.name,
        ))
        pieces.append(draft.text)
        offset += len(draft.text)
    return CleanDoc(
        source=source, text="".join(pieces), paragraphs=tuple(paragraphs), page_marks=tuple(marks),
        pages=pages, chars_raw=sum(len(line.text) for line in lines), dropped=dropped,
        dropped_items=tuple(items),
    )


# --- Разделы -----------------------------------------------------------------

@dataclass(frozen=True)
class Unit:
    """Структурная единица документа (§2.5): `section` — раздел книги или
    путеводителя, `entry` — запись глоссария, `group` — текст группы
    глоссария без записей. Заголовки входят в единицу, которая за ними идёт."""

    kind: str
    name: str
    start: int
    end: int


def units(clean_doc: CleanDoc) -> list[Unit]:
    """Разделы документа по порядку; вместе с заголовками покрывают весь
    очищенный текст. Два заголовка подряд — одно имя через « · »; заголовки в
    самом конце документа без текста дописываются к прежней единице."""
    source = clean_doc.source
    default = source.rules.default_group or source.title
    result: list[Unit] = []
    names: list[str] = []
    kind = ""
    start = end = 0
    has_text = False
    open_unit = False

    def close() -> None:
        nonlocal open_unit
        if not open_unit:
            return
        open_unit = False
        if not has_text and result:
            last = result[-1]
            result[-1] = Unit(last.kind, last.name, last.start, end)
            return
        result.append(Unit(kind, SECTION_JOIN.join(names), start, end))

    if source.structure == "sections":
        for paragraph in clean_doc.paragraphs:
            if paragraph.kind == "heading":
                if open_unit and not has_text:
                    names.append(paragraph.text)
                    end = paragraph.end
                    continue
                close()
                kind, names, start, end, has_text, open_unit = "section", [paragraph.text], paragraph.start, paragraph.end, False, True
            else:
                if not open_unit:
                    kind, names, start, has_text, open_unit = "section", [default], paragraph.start, False, True
                end = paragraph.end
                has_text = True
        close()
        return result

    group = [default]
    heading_start: int | None = None
    previous_heading = False
    for paragraph in clean_doc.paragraphs:
        if paragraph.kind == "heading":
            close()
            if previous_heading:
                group.append(paragraph.text)
            else:
                group = [paragraph.text]
                heading_start = paragraph.start
            end = paragraph.end
            previous_heading = True
            continue
        group_name = SECTION_JOIN.join(group)
        if paragraph.kind == "entry":
            close()
            kind, names = "entry", [f"{group_name}{GROUP_JOIN}{paragraph.name}"]
            start = heading_start if heading_start is not None else paragraph.start
            has_text, open_unit = True, True
        elif not (open_unit and kind == "group"):
            close()
            kind, names = "group", [group_name]
            start = heading_start if heading_start is not None else paragraph.start
            has_text, open_unit = True, True
        end = paragraph.end
        heading_start = None
        previous_heading = False
    if previous_heading and heading_start is not None:
        # Заголовки в конце без текста — к прежней записи.
        kind, names, start, has_text, open_unit = "group", [SECTION_JOIN.join(group)], heading_start, False, True
    close()
    return result


# --- Куски -------------------------------------------------------------------

@dataclass(frozen=True)
class Chunk:
    """Кусок индекса — все поля строки `chunks` (§4.1), кроме `tokens` и
    `embedding`: их считает модель."""

    chunk_id: str
    strategy: str
    doc: str
    lang: str
    ordinal: int
    source: str
    title: str
    section: str
    part: str | None
    sections_spanned: int
    page_from: int
    page_to: int
    char_start: int
    char_end: int
    publisher: str
    game: str
    edition: str
    text: str
    chars: int


def chunk_id(strategy: str, lang: str, doc: str, ordinal: int) -> str:
    """`<strategy>:<lang>:<doc>:<ordinal>` (§4.2), номер — четыре цифры."""
    return f"{strategy}:{lang}:{doc}:{ordinal:04d}"


def _make_chunk(
    clean_doc: CleanDoc, strategy: str, ordinal: int, start: int, end: int,
    section: str, part: str | None, spanned: int,
) -> Chunk:
    source = clean_doc.source
    text = clean_doc.text[start:end]
    return Chunk(
        chunk_id=chunk_id(strategy, source.lang, source.doc, ordinal),
        strategy=strategy, doc=source.doc, lang=source.lang, ordinal=ordinal,
        source=source.path, title=source.title, section=section, part=part,
        sections_spanned=spanned,
        page_from=clean_doc.page_at(start), page_to=clean_doc.page_at(max(start, end - 1)),
        char_start=start, char_end=end,
        publisher=source.publisher, game=source.game, edition=source.edition,
        text=text, chars=len(text),
    )


def _trim(text: str, start: int, end: int) -> tuple[int, int]:
    while start < end and text[start].isspace():
        start += 1
    while end > start and text[end - 1].isspace():
        end -= 1
    return start, end


def _space_before(text: str, low: int, high: int) -> int:
    """Позиция последнего пробельного символа в `text[low:high]`; -1 — нет."""
    for index in range(min(high, len(text)) - 1, low - 1, -1):
        if text[index].isspace():
            return index
    return -1


def fixed_chunks(clean_doc: CleanDoc, size: int, overlap: int) -> list[Chunk]:
    """«Фиксированный размер» (§2.4): окна по `size` символов с перекрытием
    `overlap`; граница сдвигается назад к ближайшему пробелу, чтобы не резать
    слово. Предложений, абзацев и заголовков стратегия не видит. `section` —
    раздел, где кусок начался, `sections_spanned` — сколько разделов он
    задел."""
    if not 0 <= overlap < size:
        raise ValueError(f"перекрытие {overlap} должно быть меньше размера {size}")
    text = clean_doc.text
    doc_units = units(clean_doc)
    starts = [unit.start for unit in doc_units]
    chunks: list[Chunk] = []
    position = 0
    while position < len(text):
        end = min(position + size, len(text))
        if end < len(text) and not text[end].isspace():
            space = _space_before(text, position + 1, end)
            if space > position:
                end = space
        first, last = _trim(text, position, end)
        if first < last:
            index = max(bisect.bisect_right(starts, first) - 1, 0)
            spanned = sum(1 for unit in doc_units if unit.start < last and unit.end > first)
            chunks.append(_make_chunk(
                clean_doc, "fixed", len(chunks) + 1, first, last,
                doc_units[index].name if doc_units else clean_doc.source.title, None, max(spanned, 1),
            ))
        if end >= len(text):
            break
        following = end - overlap
        space = _space_before(text, position + 1, following)
        following = space + 1 if space >= 0 else following
        position = following if following > position else end
    return chunks


def _sentences(text: str, start: int, end: int) -> list[tuple[int, int]]:
    """Предложения абзаца `text[start:end]` — по знакам конца предложения."""
    result: list[tuple[int, int]] = []
    begin = start
    for match in _SENTENCE.finditer(text, start, end):
        result.append(_trim(text, begin, match.end()))
        begin = match.end()
    if begin < end:
        result.append(_trim(text, begin, end))
    return [(first, last) for first, last in result if first < last]


def _hard_split(text: str, start: int, end: int, limit: int) -> list[tuple[int, int]]:
    """Последний довод — предложение длиннее предела: по пробелам."""
    result: list[tuple[int, int]] = []
    while end - start > limit:
        space = _space_before(text, start + 1, start + limit + 1)
        cut = space if space > start else start + limit
        result.append(_trim(text, start, cut))
        start, _ = _trim(text, cut, end)
    result.append((start, end))
    return result


def _pieces(clean_doc: CleanDoc, unit: Unit, limit: int) -> list[tuple[int, int]]:
    """Раздел длиннее предела — части по границам абзацев; абзац длиннее
    предела — по границам предложений (§2.5). Жадно: в часть идёт столько
    абзацев подряд, сколько влезает."""
    text = clean_doc.text
    if unit.end - unit.start <= limit:
        return [(unit.start, unit.end)]
    atoms: list[tuple[int, int]] = []
    for paragraph in clean_doc.paragraphs:
        if paragraph.start < unit.start or paragraph.end > unit.end:
            continue
        if paragraph.end - paragraph.start <= limit:
            atoms.append((paragraph.start, paragraph.end))
            continue
        for first, last in _sentences(text, paragraph.start, paragraph.end):
            if last - first <= limit:
                atoms.append((first, last))
            else:
                atoms.extend(_hard_split(text, first, last, limit))
    pieces: list[tuple[int, int]] = []
    piece_start, piece_end = atoms[0]
    for first, last in atoms[1:]:
        if last - piece_start <= limit:
            piece_end = last
        else:
            pieces.append((piece_start, piece_end))
            piece_start, piece_end = first, last
    pieces.append((piece_start, piece_end))
    return pieces


def structural_chunks(clean_doc: CleanDoc, max_chars: int) -> list[Chunk]:
    """«По структуре» (§2.5): кусок — раздел книги или путеводителя, запись
    или группа глоссария; длинный раздел — части «2/3». Короткие разделы не
    склеиваются."""
    chunks: list[Chunk] = []
    for unit in units(clean_doc):
        pieces = _pieces(clean_doc, unit, max_chars)
        for number, (first, last) in enumerate(pieces, 1):
            part = f"{number}/{len(pieces)}" if len(pieces) > 1 else None
            chunks.append(_make_chunk(clean_doc, "structural", len(chunks) + 1, first, last, unit.name, part, 1))
    return chunks


# --- Сравнение ---------------------------------------------------------------

@dataclass(frozen=True)
class Metrics:
    """Числа отчёта §7 по стратегии и документу (`doc` пуст — все документы).
    Доли считает тот, кто печатает; «лучше» здесь не бывает."""

    strategy: str
    doc: str
    chunks: int
    chars: tuple[int, int, int]
    tokens: tuple[int, int, int]
    over_limit: int
    cut_end: int
    cut_start: int
    multi_section: int
    units: int
    units_split: int
    entries: int
    entries_whole: int
    clean_chars: int
    chunk_chars: int
    token_limit: int = field(default=0)

    @property
    def volume(self) -> float:
        return self.chunk_chars / self.clean_chars if self.clean_chars else 0.0


def _spread(values: list[int]) -> tuple[int, int, int]:
    if not values:
        return (0, 0, 0)
    return (min(values), round(statistics.median(values)), max(values))


def compare(
    chunks: list[Chunk], docs: list[CleanDoc], tokens: dict[str, int], token_limit: int,
) -> list[Metrics]:
    """Метрики §7 по каждой стратегии: сначала все документы, затем каждый.
    `tokens` — число токенов модели по `chunk_id`; `token_limit` — предел
    модели (куски сверх — `over_limit`)."""
    by_key = {(clean_doc.source.doc, clean_doc.source.lang): clean_doc for clean_doc in docs}
    doc_units = {key: units(clean_doc) for key, clean_doc in by_key.items()}
    result: list[Metrics] = []
    for strategy in STRATEGIES:
        selected = [chunk for chunk in chunks if chunk.strategy == strategy]
        scopes: list[tuple[str, list[tuple[str, str]]]] = [("", list(by_key))]
        scopes += [(f"{doc}/{lang}", [(doc, lang)]) for doc, lang in by_key]
        for label, keys in scopes:
            part = [chunk for chunk in selected if (chunk.doc, chunk.lang) in keys]
            cut_start = 0
            units_total = units_split = entries = entries_whole = 0
            for key in keys:
                clean_doc = by_key[key]
                own = [chunk for chunk in part if (chunk.doc, chunk.lang) == key]
                for chunk in own:
                    # Хватает хвоста перед куском: пробелы, кавычки и скобки
                    # между концом предложения и куском — единицы символов.
                    before = clean_doc.text[max(0, chunk.char_start - 64):chunk.char_start]
                    if _starts_lower(chunk.text) or (before.strip() and not _ends_clause(before)):
                        cut_start += 1
                for unit in doc_units[key]:
                    whole = any(chunk.char_start <= unit.start and chunk.char_end >= unit.end for chunk in own)
                    units_total += 1
                    units_split += 0 if whole else 1
                    if unit.kind == "entry":
                        entries += 1
                        entries_whole += 1 if whole else 0
            counts = [tokens.get(chunk.chunk_id, 0) for chunk in part]
            result.append(Metrics(
                strategy=strategy, doc=label, chunks=len(part),
                chars=_spread([chunk.chars for chunk in part]),
                tokens=_spread(counts),
                over_limit=sum(1 for count in counts if count > token_limit),
                cut_end=sum(1 for chunk in part if not _ends_clause(chunk.text)),
                cut_start=cut_start,
                multi_section=sum(1 for chunk in part if chunk.sections_spanned > 1),
                units=units_total, units_split=units_split,
                entries=entries, entries_whole=entries_whole,
                clean_chars=sum(len(by_key[key].text) for key in keys),
                chunk_chars=sum(chunk.chars for chunk in part),
                token_limit=token_limit,
            ))
    return result
