"""Проверка и сборка ответа RAG (день 24, спецификация §3).

Модель отвечает только по выдержкам, ставит после утверждений номер выдержки
`[N]` и в конце пишет раздел «Цитаты:» с дословными фрагментами. Этот модуль
делает то, что формализуемо: проверяет номера и дословность цитат по тексту
кусков, находит «Не знаю» и уточняющий вопрос, собирает под ответом список
источников из метаданных куска. Подтверждает ли цитата утверждение — смысл, его
оценивает человек: модуль форму проверяет, смысл не трогает.

День 25 (§2.4, §3) добавляет второй вид источника — запись памяти задачи: после
утверждения, опирающегося на неё, модель ставит ключ записи в квадратных
скобках, `[тиран]`. Код проверяет, что запись с таким ключом была в запросе
этого хода, и пишет её значение в «Источники:». Верно ли записано в памяти и
подтверждает ли запись утверждение — смысл, его оценивает человек. Без
`memory_keys` ссылки на память не разбираются вовсе: проверка ровно дня 24.

Чистые функции: ни сети, ни диска, ни логов, ни модели, ни Too Many Bones. Лист
графа — из проекта не импортирует ничего. Работает со словарями выдачи
(`chunk_id`, `title`, `section`, `part`, `page_from`, `page_to`, `text`), про
индекс и про память агента не знает: записи и ключи приходят данными;
маркеры формата («Цитаты:», «Не знаю») — тоже, `AnswerFormat`, языковых условий
в коде нет (правило дня 21).
"""

from __future__ import annotations

import re
from collections.abc import Mapping, Sequence
from dataclasses import dataclass

# Вид ответа (§2.3): что он собой представляет, а не хорош ли он.
KIND_ANSWER = "по выдержкам"
KIND_IDK = "не знаю"
KIND_NO_REFS = "без ссылок"
# День 25 (§2.4): нет ни номеров выдержек, ни цитат, есть хотя бы одна верная
# ссылка на запись памяти задачи, и тело не начинается с «Не знаю».
KIND_MEMORY = "по памяти задачи"

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
class MemoryEntry:
    """Запись памяти, ушедшая в запрос (день 25, §3): ключ, значение и подпись
    слоя — «память задачи», «долговременная память». Подпись — данные от
    агента: модуль про слои не знает."""

    key: str
    value: str
    label: str


@dataclass(frozen=True)
class AnswerCheck:
    """Результат `check()` (§3). `raw` — ответ модели как пришёл, `body` — текст
    до первого служебного раздела, `empty`/`excerpts` — выдача пуста по порогу и
    сколько выдержек было в запросе, `refs` — номера из тела в порядке первого
    упоминания, `claims` — на каждый номер предложения тела с ним, `asks` — в
    теле есть «?», `own_sources` — модель написала свой раздел «Источники:»
    (он выброшен), `violations` — фразы для панели, лога и сообщения повтора.

    День 25 (§3), в конце и с умолчаниями: `memory_refs` — ключи записей памяти
    из тела, которые в запросе были, в порядке первого упоминания;
    `memory_missing` — ключи карты в скобках, записи с которыми в запросе не
    было; `memory_claims` — на каждый упомянутый ключ предложения тела с ним
    (тем же разрезом, что `claims`)."""

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
    memory_refs: tuple[str, ...] = ()
    memory_missing: tuple[str, ...] = ()
    memory_claims: tuple[tuple[str, tuple[str, ...]], ...] = ()

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
_QUOTE_LINE = re.compile(r"^\s*(?:(?:[-*•]|\d+[.)])\s+)?\[(\d+)\]\s*[:—–-]?\s*(.*)$")
_SENTENCE_BREAK = re.compile(r"(?<=[.!?])\s+|\n+")
_QUOTE_PAIRS = (("«", "»"), ('"', '"'), ("“", "”"), ("„", "“"), ("„", "”"))
# Знаки, которые модель ставит после закрывающей кавычки: «…».
_AFTER_QUOTE = ".,;: "


def _norm_header(line: str) -> str:
    """Строка без markdown-разметки, кавычек, регистра и двоеточия — для
    сравнения с заголовком раздела."""
    return _HEADER_NOISE.sub("", line).strip().rstrip(":").strip().casefold()


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


# --- Скобки-ссылки: номера выдержек (день 24) и ключи памяти (день 25, §2.4) ----
# Одна грамматика на номера в тексте, ключи памяти и ссылку в начале
# предложения (правка по ревью: разборов было три, и номер из смешанной скобки
# `[1, тиран]` не попадал в номера ответа, а `[¹]` в начале предложения ронял
# `int()` — `str.isdigit()` верен и для надстрочных цифр, а `\d` нет).
# - Скобка из одних номеров через запятую или `;` — номера, ровно как на дне 24
#   (в том числе перед `(`).
# - С ключами карты (`canon` непуст) — скобка, каждая часть которой номер или
#   ключ карты: `[тиран]`, `[тиран, сложность]`, `[1, состав партии]`. Ссылка
#   markdown `[текст](…)` ключом не считается.
# - Скобка, где есть хоть одна другая часть (`[см. выше]`, `[тиран, см. выше]`),
#   ссылкой не считается вовсе. Без `canon` это ровно ссылки дня 24.
_BRACKET = re.compile(r"\[([^\[\]]+)\]")
_LEADING_BRACKET = re.compile(r"\s*\[([^\[\]]+)\]")
_NUMBERS_ONLY = re.compile(r"\d+(?:\s*[,;]\s*\d+)*")
_NUMBER = re.compile(r"\d+")
_KEY_SEPARATORS = re.compile(r"[,;]")

# Предложение тела: текст, номера выдержек и ключи памяти в нём.
_Sentence = tuple[str, set[int], set[str]]


def _norm_key(text: str) -> str:
    """Ключ карты для сравнения: регистр, ё → е, пробелы. Своя функция, а не
    `context.normalize_key()`: модуль — лист графа и `context.py` не импортирует
    (§3)."""
    return " ".join(text.lower().replace("ё", "е").split())


def _bracket(
    content: str, linked: bool, canon: Mapping[str, str],
) -> tuple[list[int], list[str]] | None:
    """Скобка-ссылка: номера выдержек и ключи памяти в порядке записи; `None` —
    скобка не ссылка. `linked` — сразу за скобкой `(` (ссылка markdown),
    `canon` — «нормализованный ключ → ключ карты»; пуст — только номера дня 24."""
    if _NUMBERS_ONLY.fullmatch(content):
        return [int(n) for n in _NUMBER.findall(content)], []
    if not canon or linked:
        return None
    numbers: list[int] = []
    keys: list[str] = []
    for part in _KEY_SEPARATORS.split(content):
        part = part.strip()
        if _NUMBER.fullmatch(part):
            numbers.append(int(part))
        elif _norm_key(part) in canon:
            keys.append(canon[_norm_key(part)])
        else:
            return None
    return numbers, keys


def _refs_in(text: str, canon: Mapping[str, str]) -> tuple[list[int], list[str]]:
    """Номера выдержек и ключи памяти всех скобок-ссылок текста в порядке
    упоминания, с повторами."""
    numbers: list[int] = []
    keys: list[str] = []
    for match in _BRACKET.finditer(text):
        found = _bracket(match.group(1), text.startswith("(", match.end()), canon)
        if found is not None:
            numbers += found[0]
            keys += found[1]
    return numbers, keys


def _leading_refs(chunk: str, canon: Mapping[str, str]) -> tuple[int, list[int], list[str]]:
    """Скобки-ссылки в самом начале предложения: где они кончаются, номера и
    ключи. Скобка, которая не ссылка, обрывает ряд."""
    position = 0
    numbers: list[int] = []
    keys: list[str] = []
    while True:
        match = _LEADING_BRACKET.match(chunk, position)
        if match is None:
            break
        found = _bracket(match.group(1), chunk.startswith("(", match.end()), canon)
        if found is None:
            break
        numbers += found[0]
        keys += found[1]
        position = match.end()
    return position, numbers, keys


def _sentences(body: str, canon: Mapping[str, str]) -> list[_Sentence]:
    """Предложения тела с номерами выдержек и ключами памяти в них. Режутся по
    `.`/`!`/`?` и переносам строк; ссылка в начале предложения относится к
    предыдущему."""
    sentences: list[_Sentence] = []
    for chunk in _SENTENCE_BREAK.split(body):
        chunk = chunk.strip()
        if not chunk:
            continue
        end, numbers, keys = _leading_refs(chunk, canon)
        if end and sentences:
            sentences[-1][1].update(numbers)
            sentences[-1][2].update(keys)
            chunk = chunk[end:].strip()
            if not chunk:
                continue
        numbers, keys = _refs_in(chunk, canon)
        sentences.append((chunk, set(numbers), set(keys)))
    return sentences


def _claims(
    sentences: Sequence[_Sentence], refs: Sequence[int],
) -> tuple[tuple[int, tuple[str, ...]], ...]:
    """На каждый номер — предложения тела с ним."""
    return tuple(
        (number, tuple(text for text, numbers, _ in sentences if number in numbers))
        for number in refs
    )


def _memory_claims(
    sentences: Sequence[_Sentence], keys: Sequence[str],
) -> tuple[tuple[str, tuple[str, ...]], ...]:
    """На каждый ключ памяти — предложения тела с ним, тем же разрезом."""
    return tuple(
        (key, tuple(text for text, _, mentioned in sentences if key in mentioned))
        for key in keys
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


def _memory_violation(key: str) -> str:
    return (
        f"в тексте [{key}] — записи с таким ключом в памяти этого запроса нет: "
        "ставь в скобки только ключи из блока «Рабочая память»"
    )


def check(
    raw: str, hits: Sequence[Mapping], empty: bool, fmt: AnswerFormat,
    memory: Sequence[MemoryEntry] = (), memory_keys: Sequence[str] = (),
) -> AnswerCheck:
    """Разбор и проверка ответа по §2.3: `hits` — выдача хода (с `text`), `empty`
    — выдача пуста по порогу. Смысл не проверяется.

    День 25 (§2.4): `memory` — записи памяти, ушедшие в запрос этого хода,
    `memory_keys` — все ключи карты памяти. Ключ карты в квадратных скобках —
    ссылка на память: верная, если запись с ним в запросе была, иначе
    нарушение. Без `memory_keys` ссылки на память не разбираются — проверка
    ровно дня 24 (голый агент `rag-eval`)."""
    hits = tuple(hits)
    excerpts = [words(str(hit.get("text") or "")) for hit in hits]
    body, raw_quotes, had_quotes, own_sources = _split(raw or "", fmt)
    # Номера выдержек и ключи памяти — одним разбором скобок (правка по ревью),
    # в порядке первого упоминания.
    canon = {_norm_key(key): key for key in memory_keys if _norm_key(key)}
    numbers, keys = _refs_in(body, canon)
    refs = list(dict.fromkeys(numbers))
    mentioned = list(dict.fromkeys(keys))
    quotes = tuple(_check_quote(n, text, excerpts, fmt) for n, text in raw_quotes)
    idk = _starts_with_idk(body, fmt)
    asks = "?" in body
    violations: list[str] = []

    available = {_norm_key(entry.key) for entry in memory}
    memory_refs = tuple(key for key in mentioned if _norm_key(key) in available)
    memory_missing = tuple(key for key in mentioned if _norm_key(key) not in available)
    # Ответ только по памяти (§2.4): ни номеров выдержек, ни цитат, есть верная
    # ссылка на запись.
    memory_only = bool(memory_refs) and not (refs or quotes)

    if empty or not hits:
        if idk:
            kind = KIND_IDK
            if not asks:
                violations.append("«Не знаю» без уточняющего вопроса — задай один уточняющий вопрос")
            if refs or quotes:
                violations.append(
                    "номера выдержек или цитаты при пустой выдаче — выдержек в запросе нет"
                )
        elif memory_only:
            # «Не начинается с «Не знаю»» здесь не нарушение: ответ опирается на
            # слова игрока, а не на правила (вопрос о разговоре, итог).
            kind = KIND_MEMORY
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
        elif memory_only:
            kind = KIND_MEMORY
        else:
            kind = KIND_NO_REFS
        if kind == KIND_NO_REFS:
            violations.append(
                "ответ не ссылается ни на одну выдержку и не начинается с «Не знаю»"
            )
        elif kind == KIND_MEMORY:
            # Выдержки прошли порог, а ответ на них не ссылается — не нарушение:
            # код не отличает «напомни, что решили» от вопроса о правилах (урок
            # «Спасибо, понятно!» дня 24); панель показывает строку «сверьте».
            pass
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

    # Ссылка на память без записи в запросе — нарушение при любом виде ответа
    # (§2.4): ловится не значение, а сам факт выдуманной записи.
    violations.extend(_memory_violation(key) for key in memory_missing)

    # Предложения тела режутся один раз — на утверждения с номерами и с ключами.
    sentences = _sentences(body, canon)
    return AnswerCheck(
        raw=raw or "",
        body=body,
        empty=bool(empty or not hits),
        excerpts=len(hits),
        kind=kind,
        refs=tuple(refs),
        quotes=quotes,
        claims=_claims(sentences, refs),
        asks=asks,
        own_sources=own_sources,
        violations=tuple(violations),
        memory_refs=memory_refs,
        memory_missing=memory_missing,
        memory_claims=_memory_claims(sentences, mentioned),
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


# Значение записи памяти в «Источниках» — целиком; длину держит промпт разбора
# памяти (`MEMORY_VALUE_WORDS`), а обрезка по 120 символов резала бы почти
# каждое `уточнено` (§2.4). С «…» режется только значение длиннее этого числа —
# страховка от сбоя разбора.
MEMORY_VALUE_MAX_CHARS = 300


def memory_value_text(value: str) -> str:
    """Значение записи памяти для «Источников» и таблиц: пробелы схлопнуты,
    длиннее `MEMORY_VALUE_MAX_CHARS` — с «…»."""
    text = " ".join(str(value).split())
    if len(text) > MEMORY_VALUE_MAX_CHARS:
        text = text[:MEMORY_VALUE_MAX_CHARS].rstrip() + "…"
    return text


# Строка «Источники:» у ответа, который не ссылается ни на одну выдержку (день 24).
_NO_REFS_NOTE = "⚠️ ответ не ссылается ни на одну выдержку — сверить его с правилами нельзя"


def _threshold_tail(threshold: float | None, best: float | None) -> str:
    """« 0.10 (лучший — 0.007)» — порог и лучшая оценка для строк пустой выдачи."""
    tail = f" {threshold:.2f}" if threshold is not None else ""
    if best is not None:
        tail += f" (лучший — {best:.3f})"
    return tail


def empty_reason(threshold: float | None = None, best: float | None = None) -> str:
    """Почему источников нет при пустой выдаче: «ни один кусок правил не прошёл
    порог релевантности 0.10 (лучший — 0.007)» — для `render()` и
    `render_unchecked()`."""
    return f"ни один кусок правил не прошёл порог релевантности{_threshold_tail(threshold, best)}"


def render(
    check: AnswerCheck,
    hits: Sequence[Mapping],
    fmt: AnswerFormat,
    threshold: float | None = None,
    best: float | None = None,
    memory: Sequence[MemoryEntry] = (),
) -> str:
    """Ответ для игрока по §2.2 (Markdown): тело как есть, «Источники:» из
    метаданных куска и «Цитаты:» с пометками ⚠️. `threshold` и `best` — для
    строки пустой выдачи. Раздел «Источники:», написанный моделью, уже
    выброшен разбором.

    День 25 (§2.4): после строк выдержек — строки записей памяти по порядку
    первого упоминания, значение — на момент ответа (память меняется дальше, а
    собранный ответ в истории остаётся каким был). Ключ без записи — строка с
    ⚠️. Ответ только по памяти при пустой выдаче или при выдержках, на которые он
    не ссылается, получает ещё строку про правила."""
    hits = tuple(hits)
    sources_title = f"**{fmt.sources_header.rstrip(':')}:**"
    quotes_title = f"**{fmt.quotes_header.rstrip(':')}:**"
    numbers = sorted({*check.refs, *(quote.number for quote in check.quotes)})
    entries = {_norm_key(entry.key): entry for entry in memory}
    memory_lines: list[str] = []
    for key, _ in check.memory_claims:
        entry = entries.get(_norm_key(key))
        if entry is None:
            memory_lines.append(f"- [{key}] ⚠️ записи с таким ключом в памяти этого запроса не было")
        else:
            memory_lines.append(f"- [{key}] {entry.label} · «{memory_value_text(entry.value)}»")
    parts: list[str] = []
    if check.body:
        parts.append(check.body)

    if numbers or memory_lines:
        lines = [sources_title]
        for number in numbers:
            if 1 <= number <= len(hits):
                lines.append(f"- [{number}] {source_label(hits[number - 1])}")
            else:
                lines.append(f"- [{number}] ⚠️ выдержки с таким номером в запросе не было")
        lines.extend(memory_lines)
        if not numbers:
            if check.empty:
                lines.append(
                    f"- правила: не найдены — ни один кусок не прошёл порог{_threshold_tail(threshold, best)}"
                )
            elif check.kind == KIND_NO_REFS:
                # Ответ без ссылок, в котором только ключи без записи, — нарушение
                # дня 24, и строка у него та же (правка по ревью: иначе он
                # выглядел бы нейтрально, как ответ по памяти).
                lines.append(f"- {_NO_REFS_NOTE}")
            else:
                lines.append(
                    f"- выдержки: в запросе было {check.excerpts}, ответ на них не ссылается"
                )
        parts.append("\n".join(lines))
    elif check.empty:
        parts.append(f"{sources_title} не найдены — {empty_reason(threshold, best)}.")
    elif check.kind == KIND_IDK:
        parts.append(
            f"{sources_title} ответ не ссылается на выдержки (в запросе их было {check.excerpts})."
        )
    else:
        parts.append(f"{sources_title} {_NO_REFS_NOTE}.")

    if check.quotes:
        lines = [quotes_title]
        for quote in check.quotes:
            mark = quote_mark(quote)
            lines.append(f"- [{quote.number}] «{quote.text}»" + (f" — {mark}" if mark else ""))
        parts.append("\n".join(lines))
    return "\n\n".join(parts)


def render_unchecked(raw: str, fmt: AnswerFormat, reason: str) -> str:
    """Ответ для игрока без проверки (день 25, §2.4, §3): части RAG в последнем
    сообщении не было — запрос ушёл чистым вопросом, и инструкций формата модель
    не видела. Текст модели как есть и одна строка «Источники: не найдены —
    <reason>; ответ не сверен с правилами.» — ответ из общих знаний не должен
    быть неотличим от ответа по правилам. Выключенный RAG — режим сравнения, а
    не сбой: эту функцию для него не зовут."""
    title = f"**{fmt.sources_header.rstrip(':')}:**"
    note = f"{title} не найдены — {reason}; ответ не сверен с правилами."
    body = (raw or "").strip()
    return f"{body}\n\n{note}" if body else note


def render_failed(raw: str, fmt: AnswerFormat, reason: str) -> str:
    """Ответ для игрока, когда поиск не удался (день 25, §2.4): строка
    «Источники: не найдены — поиск по правилам не удался (…); ответ не сверен с
    правилами.»."""
    return render_unchecked(
        raw, fmt, f"поиск по правилам не удался ({reason or 'причина не названа'})"
    )


def repair_message(check: AnswerCheck, instruction: str) -> str:
    """Сообщение повтора формата (§2.4): заголовок, нарушения строками «- …»,
    пустая строка и инструкция."""
    lines = [REPAIR_HEADER, *(f"- {violation}" for violation in check.violations)]
    return "\n".join(lines) + "\n\n" + instruction.strip()
