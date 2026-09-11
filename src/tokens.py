# TooManyRules — счёт токенов (день 8, неделя 2).
#
# Модуль знает только про текст и числа: ни Gradio, ни Too Many Bones, ни
# вызовов API здесь нет и быть не может. Из проекта не импортируется ничего —
# это лист графа зависимостей (app.py → presets.py → agent.py → tokens.py),
# см. `docs/TooManyRules — Неделя 2 архитектура.md`, §3, и спецификацию
# дня 8, §4.
#
# Всё, что здесь считается до запроса, — оценка, а не факт. Источник истины
# один: `usage` в ответе API. Точное число до запроса дал бы только тот же
# токенизатор, что стоит на сервере модели; он раздаётся демо-зипом и тянет
# `transformers` — сотни мегабайт в проект с тремя строчками в
# requirements.txt (обоснование — спецификация дня 8, §2.1). Вместо него —
# эвристика по классам символов плюс калибровка по собственному трафику:
# агент копит отношение «факт / оценка» и передаёт его сюда параметром.
#
# Функции здесь чистые и без состояния: их зовут на каждый рендер панели,
# поэтому своих логов у модуля тоже нет — всё, что нужно видеть в терминале,
# логирует агент.

import re
from dataclasses import dataclass

# --- Эвристика оценки ----------------------------------------------------
# Символов на токен по классам символов. Пробельные не считаются вовсе: в BPE
# они прилипают к соседнему токену, и отдельная строка под них только испортила
# бы оценку русского текста, где пробелов около шестой части.
#
# Стартовые значения (спецификация дня 8, §3.1): кириллица — русское слово в
# 5-6 букв обычно режется на 2 токена; латиница — из правила документации
# DeepSeek «1 английский символ ≈ 0.3 токена» (≈3.3 с пробелами, ≈4.0 без);
# прочее (цифры, пунктуация, эмодзи) дробится мельче слов.
#
# ЗАМЕР (обязательная сверка с фактом, спецификация дня 8, §3.1): живой диалог
# из 8 ходов на русском, пресет «Базовый», `deepseek-v4-flash`, 10.09.2026.
# (Эту модель 11.09.2026 сменила `deepseek-flash`; токенизатор у семейства
# общий, но если расхождение в панели поедет — замер стоит повторить.)
# Оценка против фактического `prompt_tokens`: по ходам от +5.0% до +9.7%, по
# всем восьми вызовам суммарно +6.7% (24 877 против 23 325), коэффициент
# калибровки к концу диалога — ×0.938. По тексту ответов, где служебной
# разметки нет вовсе, — +7.0% (7 628 против 7 129).
#
# Стартовые коэффициенты уложились в допуск ±15% и поэтому оставлены как есть:
# подгонять их под один диалог — ровно тот хак, против которого написан §2.1.
# Систематические +7% снимает калибровка, и на видео видно, как она это
# делает за первые же ходы.
CHARS_PER_TOKEN_CYRILLIC = 2.5
CHARS_PER_TOKEN_LATIN = 4.0
CHARS_PER_TOKEN_OTHER = 2.0

# Служебные токены разметки: роль и разделители каждого сообщения плюс
# обвязка запроса целиком.
TOKENS_PER_MESSAGE = 4
TOKENS_PER_REQUEST = 3

_WHITESPACE_RE = re.compile(r"\s+")
# Кириллица целиком, включая ё и расширения: считаем именно её, а не «всё
# нелатинское», — расхождение классов и есть смысл таблицы выше.
_CYRILLIC_RE = re.compile(r"[Ѐ-ԯ]+")
_LATIN_RE = re.compile(r"[A-Za-z]+")

# --- Контекстное окно ----------------------------------------------------
# Размер окна модели — такой же внешний факт, как таблица цен в `agent.py`,
# и ключ здесь тот же: имя модели. Источник: документация DeepSeek, «Models &
# Pricing» (https://api-docs.deepseek.com/quick_start/pricing), сверено
# 11.09.2026: CONTEXT LENGTH 1M у обеих моделей, MAX OUTPUT 384K.
#
# Документация говорит «1M», сервер отвечает точнее. Заведомо переполненный
# запрос вернул 400 с текстом: «This model's maximum context length is
# 1048576 tokens. However, you requested 1146719 tokens». Значит окно —
# 2^20 = 1 048 576, а не круглый миллион; константа стоит по факту от API,
# как и требует спецификация дня 8, §3.3. Проверено 10.09.2026 на прежней
# `deepseek-v4-flash` и переспрошено 11.09.2026 у пришедшей ей на смену
# `deepseek-flash` — ответ тот же.
#
# Модели, которой здесь нет, соответствует `None`: всё, что считает бюджет,
# обязано это пережить и показать «н/д». Тот же задел, что у
# `estimate_cost_usd()` — на неделе 6 через `AgentConfig.base_url` появится
# локальная модель, у которой окно нам неизвестно.
CONTEXT_WINDOW_TOKENS: dict[str, int] = {
    "deepseek-flash": 1_048_576,
    "deepseek-v4-pro": 1_048_576,
}

# Сколько места оставляем под ответ, если `max_tokens` в конфиге не задан.
# Не «максимум модели» (384K), а практический потолок ответа по правилам
# настолки: занятость должна показывать, сколько осталось под нормальный
# ответ, а не под теоретический предел.
DEFAULT_ANSWER_RESERVE = 4_096

# Границы уровней занятости: до 70% — «ok», 70-90% — «warn», 90-100% —
# «danger», 100% и выше — «over».
WARN_RATIO = 0.7
DANGER_RATIO = 0.9

# Верхняя граница заполнителя — предохранитель от лишнего нуля в поле ввода.
# В спецификации дня 8 (§4) стояло «~200 000», но окно моделей V4 оказалось
# больше миллиона: с порогом ниже окна кнопка «Набить контекст» не смогла бы
# показать главное — настоящий отказ API по переполнению. Порог поднят ровно
# настолько, чтобы заведомо переполненный запрос собирался (1.2M по оценке —
# это ≈1.15M по счёту API, то есть на 100 тысяч выше окна), а мегабайты сверх
# этого — уже нет.
FILLER_MAX_TOKENS = 1_200_000

# Строка заполнителя: осмысленный русский текст, явно помеченный и
# пронумерованный, — на видео должно быть видно и что это не настоящий
# вопрос, и сколько его.
#
# Формулировка подобрана замером, а не на слух. Заполнитель здесь — не только
# балласт, но и измерительный прибор: если он токенизируется иначе, чем живой
# диалог, то «набить 8 тысяч» будет означать что угодно, кроме восьми тысяч.
# Три варианта строки, замеренные вызовами API 10.09.2026 на наборе в 4 000
# токенов по оценке:
#   гладкий русский текст без чисел     — оценка выше факта на 21%
#   текст с повторяющимся номером       — оценка выше факта на 2%
#   редкие слова и разнобой             — оценка ниже факта на 14%
# Взят средний вариант: номер строки повторяется трижды, и оценка совпадает
# с фактом с точностью до пары процентов.
_FILLER_LINE = (
    "[заполнитель {index:03d}] Строка {index} из набивки контекста: текст "
    "служебный, к правилам Too Many Bones отношения не имеет и нужен только "
    "чтобы занять место в запросе. Вес строки около 60 токенов, номер "
    "{index} повторяется трижды."
)


def context_window(model: str) -> int | None:
    """Размер контекстного окна модели или `None`, если модели нет в таблице."""
    return CONTEXT_WINDOW_TOKENS.get(model)


def estimate_tokens(text: str | None) -> int:
    """Оценка числа токенов в строке.

    Именно оценка: источник истины — `usage` в ответе API (см. спецификацию
    дня 8, §2.1). Пустая строка и `None` дают 0, отрицательных чисел здесь не
    бывает.
    """
    return round(_raw_tokens(text))


def _raw_tokens(text: str | None) -> float:
    """Та же оценка, но без округления.

    Дробное промежуточное значение нужно заполнителю: суммы по строкам
    складываются точно, и `filler_text()` не промахивается на округлениях.
    Считаем тремя проходами по строке средствами `re` (C-уровень): построчный
    цикл по символам на заполнителе в миллион токенов был бы заметен глазом.
    """
    if not text:
        return 0.0
    stripped = _WHITESPACE_RE.sub("", text)
    without_cyrillic = _CYRILLIC_RE.sub("", stripped)
    cyrillic = len(stripped) - len(without_cyrillic)
    without_latin = _LATIN_RE.sub("", without_cyrillic)
    latin = len(without_cyrillic) - len(without_latin)
    other = len(without_latin)
    return (
        cyrillic / CHARS_PER_TOKEN_CYRILLIC
        + latin / CHARS_PER_TOKEN_LATIN
        + other / CHARS_PER_TOKEN_OTHER
    )


@dataclass(frozen=True)
class RequestTokens:
    """Разложение того, что уходит в модель, по частям."""

    system: int
    history: int
    question: int
    overhead: int              # служебные токены разметки
    total: int
    per_message: list[int]     # оценка по каждому сообщению истории, по порядку


def count_request(
    system_prompt: str,
    history: list[dict],
    question: str = "",
) -> RequestTokens:
    """Оценка запроса по частям: системный промпт + вся история + новый вопрос.

    Считается ровно то, что уходит в API: агент зовёт эту функцию по
    результату `_build_messages()`, а не по стеку отдельно (спецификация
    дня 8, §5.4).

    `question=""` — законный случай: так панель считает бюджет стека без
    нового вопроса, то есть нижнюю границу следующего запроса. Пустой вопрос
    не считается сообщением и служебных токенов не добавляет.
    """
    per_message = [estimate_tokens(_content(message)) for message in history or []]
    system = estimate_tokens(system_prompt)
    history_tokens = sum(per_message)
    question_tokens = estimate_tokens(question)

    messages = 1 + len(per_message) + (1 if question else 0)
    overhead = messages * TOKENS_PER_MESSAGE + TOKENS_PER_REQUEST
    return RequestTokens(
        system=system,
        history=history_tokens,
        question=question_tokens,
        overhead=overhead,
        total=system + history_tokens + question_tokens + overhead,
        per_message=per_message,
    )


def _content(message) -> str:
    """Текст сообщения из стека. Сообщение без `content`, с `content=None` или
    вообще не словарь оценку ронять не должны — счёт токенов идёт на каждый
    рендер панели, и падать ему негде."""
    if isinstance(message, dict):
        return message.get("content") or ""
    return ""


@dataclass(frozen=True)
class ContextUsage:
    """Бюджет контекста: сколько занято от окна модели."""

    model: str
    estimated: int             # сырая оценка
    used: int                  # она же с учётом калибровки
    limit: int | None          # None — модель вне таблицы
    answer_reserve: int
    available: int | None      # limit - answer_reserve
    free: int | None           # available - used
    ratio: float | None        # used / available, может быть > 1
    level: str                 # "ok" | "warn" | "danger" | "over" | "unknown"
    # Разложение, по которому бюджет посчитан. Поля выше — производные от
    # него, и панели нужны оба числа сразу: и «занято столько-то», и «из
    # чего это сложилось». Держим их вместе, чтобы считать один раз и в
    # одном месте — иначе панель считала бы то же самое вторым путём, и
    # числа разъехались бы.
    request: RequestTokens


def context_usage(
    request: RequestTokens,
    model: str,
    max_tokens: int | None = None,
    calibration: float | None = None,
) -> ContextUsage:
    """Бюджет контекста по оценке запроса.

    Доступно не всё окно: `max_tokens` из конфига агента (а если он не задан —
    `DEFAULT_ANSWER_RESERVE`) вычитается из лимита. Иначе бюджет врал бы —
    «занято 100%» означало бы, что ответу места уже не осталось.

    Калибровка приходит параметром и применяется к оценке множителем; своего
    состояния у модуля нет. Модель вне таблицы окон — не ошибка: тогда
    `limit`/`available`/`free`/`ratio` равны `None`, уровень «unknown», и
    панель показывает «н/д».
    """
    estimated = request.total
    used = round(estimated * calibration) if calibration else estimated
    answer_reserve = max_tokens if max_tokens else DEFAULT_ANSWER_RESERVE

    limit = context_window(model)
    if limit is None:
        return ContextUsage(
            model=model,
            estimated=estimated,
            used=used,
            limit=None,
            answer_reserve=answer_reserve,
            available=None,
            free=None,
            ratio=None,
            level="unknown",
            request=request,
        )

    available = max(limit - answer_reserve, 0)
    free = available - used
    # Резерв больше окна — вырожденный случай (в проекте не встречается, но
    # `max_tokens` берётся из конфига и в принципе может быть любым): делить
    # не на что, зато очевидно, что не влезаем.
    ratio = used / available if available > 0 else None
    return ContextUsage(
        model=model,
        estimated=estimated,
        used=used,
        limit=limit,
        answer_reserve=answer_reserve,
        available=available,
        free=free,
        ratio=ratio,
        level=_level(ratio),
        request=request,
    )


def _level(ratio: float | None) -> str:
    if ratio is None:
        return "over"
    if ratio >= 1.0:
        return "over"
    if ratio >= DANGER_RATIO:
        return "danger"
    if ratio >= WARN_RATIO:
        return "warn"
    return "ok"


def filler_text(target_tokens: int) -> str:
    """Текст-заполнитель примерно на указанное число токенов — для проверки
    того, что происходит при переполнении контекста.

    Набирается до целевой оценки собственной же `estimate_tokens()`, то есть
    возвращается то, что реально померено, а не «на глаз». Значение выше
    `FILLER_MAX_TOKENS` обрезается до него, ноль и отрицательное дают пустую
    строку.
    """
    try:
        target = int(target_tokens)
    except (TypeError, ValueError):
        return ""
    target = min(target, FILLER_MAX_TOKENS)
    if target <= 0:
        return ""

    lines: list[str] = []
    raw = 0.0
    index = 1
    while round(raw) < target:
        line = _FILLER_LINE.format(index=index)
        lines.append(line)
        # Пробельные символы не считаются, поэтому сумма по строкам в точности
        # равна оценке склеенного текста — добирать после сборки не приходится.
        raw += _raw_tokens(line)
        index += 1
    return "\n".join(lines)
