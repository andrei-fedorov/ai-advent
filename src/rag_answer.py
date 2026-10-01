"""Проверка и сборка ответа RAG (день 24, спецификация §3).

Модель отвечает только по выдержкам, ставит после утверждений номер выдержки
`[N]` и в конце пишет раздел «Цитаты:» с дословными фрагментами. Этот модуль
делает то, что формализуемо: проверяет номера и дословность цитат по тексту
кусков, находит «Не знаю» и уточняющий вопрос, собирает под ответом список
источников из метаданных куска. Подтверждает ли цитата утверждение — смысл, его
оценивает человек: модуль форму проверяет, смысл не трогает.

Чистые функции: ни сети, ни диска, ни логов, ни модели, ни Too Many Bones. Лист
графа — из проекта не импортирует ничего. Работает со словарями выдачи
(`chunk_id`, `title`, `section`, `part`, `page_from`, `page_to`, `text`), про
индекс не знает; маркеры формата («Цитаты:», «Не знаю») приходят данными —
`AnswerFormat`, языковых условий в коде нет (правило дня 21).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# Вид ответа (§2.3): что он собой представляет, а не хорош ли он.
KIND_ANSWER = "по выдержкам"
KIND_IDK = "не знаю"
KIND_NO_REFS = "без ссылок"

# Статус цитаты (§2.3). Нарушение — всё, кроме `QUOTE_OK`.
QUOTE_OK = "дословно"
QUOTE_SHORT = "короткая"
QUOTE_OTHER = "из другой выдержки"
QUOTE_LOOSE = "не дословно"
QUOTE_NO_EXCERPT = "нет такой выдержки"

# Первая строка сообщения повтора формата (§3, `repair_message()`).
REPAIR_HEADER = "Ответ не прошёл проверку формата:"


@dataclass(frozen=True)
class AnswerFormat:
    """Маркеры формата ответа. Экземпляр создаёт `presets.py` и собирает из него
    тексты инструкций, чтобы маркер в инструкции и маркер в разборе не
    разошлись (приём `MEMORY_PROMPT` дня 11)."""

    quotes_header: str
    sources_header: str
    idk_openings: tuple[str, ...]
    min_quote_words: int


@dataclass(frozen=True)
class Quote:
    """Цитата и её проверка. `words` — сколько слов в цитате (по `words()`),
    `found_in` — номер выдержки, в которой цитата найдена дословно (`None` —
    нигде), `coverage` — доля слов цитаты, которые встречаются в своей
    выдержке: подсказка для автора у `QUOTE_LOOSE`, а не второй порог."""

    number: int
    text: str
    words: int
    status: str
    found_in: int | None = None
    coverage: float | None = None


@dataclass(frozen=True)
class AnswerCheck:
    """Результат `check()` (§3). `raw` — ответ модели как пришёл, `body` — текст
    до первого служебного раздела, `empty`/`excerpts` — выдача пуста по порогу и
    сколько выдержек было в запросе, `refs` — номера из тела в порядке первого
    упоминания, `claims` — на каждый номер предложения тела с ним, `asks` — в
    теле есть «?», `own_sources` — модель написала свой раздел «Источники:»
    (он выброшен), `violations` — фразы для панели, лога и сообщения повтора."""

    raw: str
    body: str
    empty: bool
    excerpts: int
    kind: str
    refs: tuple[int, ...]
    quotes: tuple[Quote, ...]
    claims: tuple[tuple[int, tuple[str, ...]], ...]
    asks: bool
    own_sources: bool
    violations: tuple[str, ...]

    @property
    def ok(self) -> bool:
        return not self.violations

    @property
    def sources(self) -> tuple[int, ...]:
        """Номера из выдачи, на которые есть ссылка в тексте или цитата, по
        возрастанию. Номера вне выдачи сюда не входят — их показывает `render()`
        отдельной строкой."""
        numbers = {*self.refs, *(quote.number for quote in self.quotes)}
        return tuple(sorted(n for n in numbers if 1 <= n <= self.excerpts))

    @property
    def verbatim(self) -> int:
        """Сколько цитат прошли проверку (`QUOTE_OK`)."""
        return sum(1 for quote in self.quotes if quote.status == QUOTE_OK)


# --- Слова и дословность -------------------------------------------------

# Слово — последовательность букв и цифр; `#` — отдельное слово: в книге правил
# он стоит вместо числа («максимальное # кубиков») и в цитате должен совпасть.
_WORD = re.compile(r"[^\W_]+|#")
# Пропуск внутри цитаты: «…», «...», «[…]», «(…)».
_GAP = re.compile(r"\[\s*(?:…|\.\.\.)\s*\]|\(\s*(?:…|\.\.\.)\s*\)|…|\.\.\.")


def words(text: str) -> list[str]:
    """Нормализация §2.3: слова в нижнем регистре, ё → е. Пунктуация, кавычки,
    маркеры списков, переносы строк и регистр не важны. Одно определение
    «дословно» на проверку и на подсказку `coverage`."""
    return _WORD.findall(text.lower().replace("ё", "е"))


def _find(haystack: Sequence[str], needle: Sequence[str], start: int = 0) -> int:
    """Индекс, с которого `needle` идёт подряд в `haystack` не раньше `start`;
    -1 — нет."""
    size = len(needle)
    for i in range(start, len(haystack) - size + 1):
        if list(haystack[i:i + size]) == list(needle):
            return i
    return -1


def _contains(excerpt: Sequence[str], parts: Sequence[Sequence[str]]) -> bool:
    """Части цитаты (между пропусками) идут в выдержке по очереди, каждая —
    после предыдущей."""
    position = 0
    for part in parts:
        index = _find(excerpt, part, position)
        if index < 0:
            return False
        position = index + len(part)
    return True


# --- Разбор --------------------------------------------------------------

# Что срезается со строки заголовка раздела: разметка markdown и кавычки.
# Кавычки — правка по ревью: в инструкциях заголовок стоит в ёлочках
# («Цитаты:»), и модель на прогоне в приложении так его и написала; без них
# цитаты не разбирались, а повтор писал то же самое.
_HEADER_NOISE = re.compile(r"[*_#>«»\"“”„]")
_REFS = re.compile(r"\[(\d+(?:\s*[,;]\s*\d+)*)\]")
_LEADING_REFS = re.compile(r"^\s*(?:\[\d+(?:\s*[,;]\s*\d+)*\]\s*)+")
_QUOTE_LINE = re.compile(r"^\s*(?:(?:[-*•]|\d+[.)])\s+)?\[(\d+)\]\s*[:—–-]?\s*(.*)$")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")
_QUOTE_PAIRS = (("«", "»"), ('"', '"'), ("“", "”"), ("„", "“"), ("„", "”"))
# Знаки, которые модель ставит после закрывающей кавычки: «…».
_AFTER_QUOTE = ".,;: "


def _norm_header(line: str) -> str:
    """Строка без markdown-разметки, кавычек, регистра и двоеточия — для
    сравнения с заголовком раздела."""
    return _HEADER_NOISE.sub("", line).strip().rstrip(":").strip().casefold()


def _numbers(group: str) -> list[int]:
    return [int(n) for n in re.findall(r"\d+", group)]


def _strip_quotes(text: str) -> str:
    """Цитата без внешних кавычек — и без точки или запятой после закрывающей
    («…».): иначе в ответе игрока она вышла бы в двойных ёлочках (правка по
    ревью). Знак внутри кавычек остаётся частью цитаты."""
    text = text.strip()
    for opening, closing in _QUOTE_PAIRS:
        inner = text.rstrip(_AFTER_QUOTE)
        if len(inner) >= 2 and inner.startswith(opening) and inner.endswith(closing):
            return inner[1:-1].strip()
    return text


def _split(raw: str, fmt: AnswerFormat) -> tuple[str, list[tuple[int, str]], bool, bool]:
    """Тело ответа, цитаты (номер и текст как написала модель), признаки «был
    раздел «Цитаты:»» и «был свой раздел «Источники:»». Раздел начинается
    строкой, которая совпадает с заголовком. Строка без `[N]` сразу за строкой
    цитаты продолжает её (модель переносит длинную цитату); остальной текст
    раздела цитат — после пустой строки или до первой цитаты — продолжение
    тела: модель дописала его после цитат (правка по ревью: иначе он
    приклеивался к последней цитате и делал её «не дословной»)."""
    quotes_header = _norm_header(fmt.quotes_header)
    sources_header = _norm_header(fmt.sources_header)
    body: list[str] = []
    tail: list[str] = []
    quotes: list[list] = []
    section = ""
    had_quotes = own_sources = False
    continues = False               # прошлая строка — цитата или её продолжение
    for line in raw.splitlines():
        header = _norm_header(line)
        if header == quotes_header:
            section, had_quotes, continues = "quotes", True, False
            continue
        if header == sources_header:
            section = "sources"
            own_sources = True
            continue
        if not section:
            body.append(line)
        elif section == "quotes":
            if not line.strip():
                continues = False
                if tail and tail[-1]:
                    tail.append("")
                continue
            match = _QUOTE_LINE.match(line)
            if match:
                quotes.append([int(match.group(1)), match.group(2)])
                continues = True
            elif continues:
                quotes[-1][1] += " " + line.strip()
            else:
                tail.append(line)
    text = "\n\n".join(part for part in ("\n".join(body).strip(), "\n".join(tail).strip()) if part)
    return text, [(n, _strip_quotes(quote)) for n, quote in quotes], had_quotes, own_sources


def _starts_with_idk(body: str, fmt: AnswerFormat) -> bool:
    """Тело без начальной разметки начинается с маркера «не знаю»."""
    start = re.sub(r"^[\s*_#>«\"“„]+", "", body).replace("’", "'").casefold()
    return any(
        start.startswith(opening.replace("’", "'").casefold()) for opening in fmt.idk_openings
    )


def _ref_numbers(text: str) -> set[int]:
    return {n for match in _REFS.finditer(text) for n in _numbers(match.group(1))}


def _claims(body: str, refs: Sequence[int]) -> tuple[tuple[int, tuple[str, ...]], ...]:
    """На каждый номер — предложения тела с ним. Режутся по `.`/`!`/`?` и
    переносам строк; номер в начале предложения относится к предыдущему."""
    sentences: list[tuple[str, set[int]]] = []
    for chunk in _SENTENCE_BREAK.split(body):
        chunk = chunk.strip()
        if not chunk:
            continue
        leading = _LEADING_REFS.match(chunk)
        if leading and sentences:
            sentences[-1][1].update(_ref_numbers(leading.group(0)))
            chunk = chunk[leading.end():].strip()
            if not chunk:
                continue
        sentences.append((chunk, _ref_numbers(chunk)))
    return tuple(
        (number, tuple(text for text, numbers in sentences if number in numbers))
        for number in refs
    )


def _check_quote(
    number: int, text: str, excerpts: Sequence[Sequence[str]], fmt: AnswerFormat,
) -> Quote:
    parts = [p for p in (words(part) for part in _GAP.split(text)) if p]
    size = sum(len(p) for p in parts)
    if not 1 <= number <= len(excerpts):
        return Quote(number, text, size, QUOTE_NO_EXCERPT)
    own = excerpts[number - 1]
    if _contains(own, parts):
        status = QUOTE_OK if size >= fmt.min_quote_words else QUOTE_SHORT
        return Quote(number, text, size, status, found_in=number)
    for other, excerpt in enumerate(excerpts, 1):
        if other != number and _contains(excerpt, parts):
            return Quote(number, text, size, QUOTE_OTHER, found_in=other)
    known = set(own)
    flat = [w for p in parts for w in p]
    coverage = sum(1 for w in flat if w in known) / len(flat) if flat else 0.0
    return Quote(number, text, size, QUOTE_LOOSE, coverage=coverage)


def _quote_violation(quote: Quote, fmt: AnswerFormat) -> str:
    n = quote.number
    if quote.status == QUOTE_SHORT:
        return f"цитата [{n}] короче {fmt.min_quote_words} слов — приведи фрагмент подлиннее"
    if quote.status == QUOTE_OTHER:
        return (
            f"цитата [{n}] найдена не в выдержке [{n}], а в [{quote.found_in}] — "
            "поставь номер той выдержки, откуда она взята"
        )
    if quote.status == QUOTE_NO_EXCERPT:
        return f"цитата [{n}]: выдержки с таким номером в запросе не было"
    return (
        f"цитата [{n}] не найдена дословно в выдержке [{n}] "
        f"(слов из выдержки: {(quote.coverage or 0.0):.0%}) — скопируй фрагмент символ в символ"
    )


def check(
    raw: str, hits: Sequence[Mapping], empty: bool, fmt: AnswerFormat,
) -> AnswerCheck:
    """Разбор и проверка ответа по §2.3: `hits` — выдача хода (с `text`), `empty`
    — выдача пуста по порогу. Смысл не проверяется."""
    hits = tuple(hits)
    excerpts = [words(str(hit.get("text") or "")) for hit in hits]
    body, raw_quotes, had_quotes, own_sources = _split(raw or "", fmt)
    refs: list[int] = []
    for match in _REFS.finditer(body):
        for number in _numbers(match.group(1)):
            if number not in refs:
                refs.append(number)
    quotes = tuple(_check_quote(n, text, excerpts, fmt) for n, text in raw_quotes)
    idk = _starts_with_idk(body, fmt)
    asks = "?" in body
    violations: list[str] = []

    if empty or not hits:
        if idk:
            kind = KIND_IDK
            if not asks:
                violations.append("«Не знаю» без уточняющего вопроса — задай один уточняющий вопрос")
            if refs or quotes:
                violations.append(
                    "номера выдержек или цитаты при пустой выдаче — выдержек в запросе нет"
                )
        else:
            kind = KIND_NO_REFS
            if refs or quotes:
                violations.append(
                    "номера выдержек или цитаты при пустой выдаче — выдержек в запросе нет"
                )
            violations.append(
                "выдача пуста по порогу, а ответ не начинается с «Не знаю»"
            )
    else:
        k = len(hits)
        if idk:
            kind = KIND_IDK
        elif refs or quotes:
            kind = KIND_ANSWER
        else:
            kind = KIND_NO_REFS
        if kind == KIND_NO_REFS:
            violations.append(
                "ответ не ссылается ни на одну выдержку и не начинается с «Не знаю»"
            )
        else:
            for number in refs:
                if not 1 <= number <= k:
                    violations.append(
                        f"в тексте ссылка [{number}] на выдержку, которой в запросе не было "
                        f"(выдержек: {k})"
                    )
            if kind == KIND_IDK:
                if not asks:
                    violations.append(
                        "«Не знаю» без уточняющего вопроса — задай один уточняющий вопрос"
                    )
            else:
                if quotes and not refs:
                    violations.append(
                        "цитаты есть, а ссылок в тексте нет — поставь номер выдержки "
                        "после каждого утверждения"
                    )
                if refs and not quotes:
                    # Раздел есть, а строк `[N] «…»` в нём нет — своим текстом:
                    # «раздела нет» модель поняла бы неверно (правка по ревью).
                    title = f"«{fmt.quotes_header.rstrip(':')}:»"
                    violations.append(
                        f"в разделе {title} нет ни одной строки вида [N] «…» — начни каждую "
                        "цитату с номера выдержки в квадратных скобках"
                        if had_quotes
                        else f"ссылки есть, а раздела {title} нет — добавь цитаты"
                    )
                quoted = {quote.number for quote in quotes}
                for number in refs:
                    if 1 <= number <= k and quotes and number not in quoted:
                        violations.append(f"у источника [{number}] нет цитаты")
            for quote in quotes:
                if quote.status != QUOTE_OK:
                    violations.append(_quote_violation(quote, fmt))

    return AnswerCheck(
        raw=raw or "",
        body=body,
        empty=bool(empty or not hits),
        excerpts=len(hits),
        kind=kind,
        refs=tuple(refs),
        quotes=quotes,
        claims=_claims(body, refs),
        asks=asks,
        own_sources=own_sources,
        violations=tuple(violations),
    )


# --- Сборка ответа для игрока --------------------------------------------

def quote_mark(quote: Quote) -> str:
    """Пометка цитаты, не прошедшей проверку (`⚠️ <статус>`), или `""` у
    прошедшей. Одна для чата, панели и отчёта."""
    if quote.status == QUOTE_OK:
        return ""
    if quote.status == QUOTE_LOOSE:
        return f"⚠️ {quote.status} (слов из выдержки: {(quote.coverage or 0.0):.0%})"
    if quote.status == QUOTE_OTHER:
        return f"⚠️ {quote.status} [{quote.found_in}]"
    return f"⚠️ {quote.status}"


def source_label(hit: Mapping) -> str:
    """«документ · «раздел»[ · часть N/M] · стр. N[–M] · `chunk_id`» — метаданные
    куска, как их собирает код (§2.2)."""
    label = f"{hit['title']} · «{hit['section']}»"
    if hit.get("part"):
        label += f" · часть {hit['part']}"
    first, last = hit["page_from"], hit["page_to"]
    label += f" · стр. {first}" if first == last else f" · стр. {first}–{last}"
    return f"{label} · `{hit['chunk_id']}`"


def render(
    check: AnswerCheck,
    hits: Sequence[Mapping],
    fmt: AnswerFormat,
    threshold: float | None = None,
    best: float | None = None,
) -> str:
    """Ответ для игрока по §2.2 (Markdown): тело как есть, «Источники:» из
    метаданных куска и «Цитаты:» с пометками ⚠️. `threshold` и `best` — для
    строки пустой выдачи. Раздел «Источники:», написанный моделью, уже
    выброшен разбором."""
    hits = tuple(hits)
    sources_title = f"**{fmt.sources_header.rstrip(':')}:**"
    quotes_title = f"**{fmt.quotes_header.rstrip(':')}:**"
    numbers = sorted({*check.refs, *(quote.number for quote in check.quotes)})
    parts: list[str] = []
    if check.body:
        parts.append(check.body)

    if numbers:
        lines = [sources_title]
        for number in numbers:
            if 1 <= number <= len(hits):
                lines.append(f"- [{number}] {source_label(hits[number - 1])}")
            else:
                lines.append(f"- [{number}] ⚠️ выдержки с таким номером в запросе не было")
        parts.append("\n".join(lines))
    elif check.empty:
        reason = "ни один кусок правил не прошёл порог релевантности"
        if threshold is not None:
            reason += f" {threshold:.2f}"
        if best is not None:
            reason += f" (лучший — {best:.3f})"
        parts.append(f"{sources_title} не найдены — {reason}.")
    elif check.kind == KIND_IDK:
        parts.append(
            f"{sources_title} ответ не ссылается на выдержки (в запросе их было {check.excerpts})."
        )
    else:
        parts.append(
            f"{sources_title} ⚠️ ответ не ссылается ни на одну выдержку — "
            "сверить его с правилами нельзя."
        )

    if check.quotes:
        lines = [quotes_title]
        for quote in check.quotes:
            mark = quote_mark(quote)
            lines.append(f"- [{quote.number}] «{quote.text}»" + (f" — {mark}" if mark else ""))
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def repair_message(check: AnswerCheck, instruction: str) -> str:
    """Сообщение повтора формата (§2.4): заголовок, нарушения строками «- …»,
    пустая строка и инструкция."""
    lines = [REPAIR_HEADER, *(f"- {violation}" for violation in check.violations)]
    return "\n".join(lines) + "\n\n" + instruction.strip()
