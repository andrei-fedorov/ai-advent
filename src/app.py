# TooManyRules — день 1 (чат с DeepSeek), день 2 (формат ответа A vs B),
# день 3 (четыре способа рассуждения на одной и той же задаче).
# Запуск: python app.py (после pip install -r requirements.txt и настройки .env)

import json
import logging
import os
import time

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

# Логирование — на день 2 оба вызова пишутся в stdout, чтобы их можно было
# увидеть в терминале рядом с Gradio-интерфейсом.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("toomanyrules")

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
# The cheap everyday default
DEEPSEEK_MODEL = "deepseek-v4-flash"
# Hardest reasoning and coding
#DEEPSEEK_MODEL = "deepseek-v4-pro"

# Дни 1-4 спроектированы под старую deepseek-chat, которая не рассуждала.
# deepseek-v4-flash/pro — reasoning-модели с thinking, включённым по
# умолчанию: скрытые reasoning-токены тратят тот же бюджет max_tokens, что
# и видимый ответ (в дне 2 это приводило к пустому content и падению
# json.loads при finish_reason=length), и на остальных днях просто удлиняют
# ответ и повышают его стоимость без пользы для задачи. Отключаем thinking
# явно на всех вызовах дней 1-4, чтобы поведение соответствовало старой
# модели. День 5 — исключение: там режим thinking сам является переменной
# эксперимента (см. DAY5_MATRIX) и переключается отдельно.
DEEPSEEK_EXTRA_BODY = {"thinking": {"type": "disabled"}}

# Системный промпт дня 1 — оставлен без изменений. Используется и в дне 1,
# и в варианте A дня 2 (по спецификации дня 2: «тот же системный промпт, что
# и в дне 1, без изменений»).
SYSTEM_PROMPT = """\
Ты — TooManyRules, ассистент по правилам настольной игры Too Many Bones
(издательство Chip Theory Games) и её дополнений (Undertow, Age of Tyranny,
Splice & Dice, Unbreakable и другие).

Отвечай точно и по делу, ориентируясь на официальные правила игры.
Если не уверен в ответе — так и скажи, не выдумывай правила.
Отвечай на том языке, на котором задан вопрос (русский или английский).

Если вопрос не связан с Too Many Bones и настольными играми Chip Theory
Games — вежливо сообщи, что можешь помочь только с вопросами по правилам
этих игр.
"""

# Системный промпт варианта B дня 2: исходный промпт + строгие требования
# к формату (JSON), длине и явному маркеру завершения.
SYSTEM_PROMPT_B = SYSTEM_PROMPT + """

Дополнительно к своей роли: отвечай СТРОГО в формате JSON со следующими
полями:
{
  "rule_summary": "краткое изложение правила в 1-2 предложениях",
  "details": "более подробное пояснение, если нужно",
  "confidence": "high | medium | low"
}

Не добавляй никакого текста до или после JSON-объекта. Сразу после
закрывающей скобки JSON выведи строку "###END###" и ничего больше.
Уложись в 2-3 предложения суммарно в полях rule_summary и details.
"""

# Параметры ограничений для варианта B. Задаются явно в API-запросе
# (не только текстовой инструкцией в промпте) — задание требует обоих подходов.
STOP_SEQUENCE = "###END###"
# 150 (значение-пример из спецификации) слишком мало для кириллицы: она
# токенизируется хуже английского, и на составных вопросах модель обрывается
# посреди JSON, не успевая дойти до "}" и STOP_SEQUENCE. 350 даёт запас,
# но по-прежнему заметно короче и предсказуемее варианта A.
MAX_TOKENS_B = 350

# День 3: системные промпты четырёх способов рассуждения. Базовая роль
# (SYSTEM_PROMPT) используется как есть в способе 1 и как фундамент для
# способов 2 и 4; способ 3 на шаге 1 берёт отдельный «мета-промпт» (модель
# играет роль не ассистента по правилам, а автора идеального запроса).

# Способ 2: SYSTEM_PROMPT + требование сначала расписать рассуждение по шагам,
# а потом дать финальный ответ отдельным абзацем — так его проще отделить от
# рассуждения при визуальном сравнении с другими способами.
SYSTEM_PROMPT_STEP_BY_STEP = SYSTEM_PROMPT + """

Прежде чем дать финальный ответ, распиши рассуждение по шагам:
1. Какие правила потенциально применимы к ситуации.
2. В каком порядке они должны разрешаться.
3. Есть ли в правилах прямое указание на этот случай или это интерпретация.
После рассуждения дай чёткий финальный ответ отдельным абзацем.
"""


def _build_meta_prompt(question: str) -> str:
    """Способ 3, шаг 1: «мета-промптер». Собираем строкой, а не через
    `str.format()` — в вопросе могут встретиться фигурные скобки (куски
    кода, примеры правил), которые `.format` воспримет как поля."""
    return (
        "Ты помогаешь сформулировать идеальный запрос к ассистенту по правилам "
        "Too Many Bones для точного разбора следующей ситуации: «"
        + question
        + "».\n"
        "Составь промпт, который стоит использовать, чтобы получить максимально "
        "точный и обоснованный ответ. Выведи только текст промпта, без пояснений."
    )


# Способ 4: SYSTEM_PROMPT + три экспертные персоны, отвечающие последовательно,
# и обязательный итоговый вывод — согласованная позиция или честное «надо в FAQ».
SYSTEM_PROMPT_CONSILIUM = SYSTEM_PROMPT + """

Разбери следующий вопрос по правилам Too Many Bones тремя голосами
последовательно:

1. Rules Lawyer — формально, опираясь только на текст правил и порядок
   разрешения эффектов.
2. Game Designer — исходя из вероятного замысла и баланса игры.
3. Skeptic — укажи, в чём первые два мнения расходятся или где ответ
   неоднозначен, и есть ли основания сомневаться в каждом из них.

В конце дай итоговый вывод: согласованный ответ или явное указание,
что вопрос требует уточнения у официального FAQ.
"""

# День 4: сравнение эффекта `temperature` на одной и той же задаче.
# Задание требует именно эти три значения (0 / 0.7 / 1.2), по 3 повтора
# на каждое — итого 9 вызовов API. Повторы нужны, чтобы увидеть не только
# содержание ответа, но и разброс внутри одной температуры (разнообразие).
DAY4_TEMPERATURES = [0.0, 0.7, 1.2]
DAY4_REPEATS_PER_TEMPERATURE = 3

# День 5: цены DeepSeek V4 в долларах за 1M токенов (off-peak).
# Источник: https://api-docs.deepseek.com/quick_start/pricing (актуально
# на момент реализации). В пиковые часы (01:00-04:00 и 06:00-10:00 UTC,
# пн-пт) ставки ×2 — для оценочной стоимости используем off-peak, это
# дефолтный режим бо́льшую часть суток и выходные.
PRICING_PER_M_TOKENS = {
    "deepseek-v4-flash": {"input": 0.22, "output": 0.66},
    "deepseek-v4-pro":   {"input": 0.66, "output": 1.98},
}

# День 5: матрица 2×2 «модель × режим рассуждения». Единственные переменные
# между 4 вызовами — это `model` и режим `thinking` (через `extra_body`,
# а не отдельным именем модели — DeepSeek рекомендует так на странице
# Thinking Mode). Порядок ячеек и номера вариантов — по таблице §2
# спецификации (строки — модель, столбцы — режим thinking):
#   1) flash + off (самый слабый/дешёвый/быстрый), 2) flash + on,
#   3) pro   + off,                                 4) pro   + on
#   (самый сильный — флагман + рассуждение).
DAY5_MATRIX = [
    ("deepseek-v4-flash", False, "Вариант 1 — deepseek-v4-flash, thinking off"),
    ("deepseek-v4-flash", True,  "Вариант 2 — deepseek-v4-flash, thinking on"),
    ("deepseek-v4-pro",   False, "Вариант 3 — deepseek-v4-pro,   thinking off"),
    ("deepseek-v4-pro",   True,  "Вариант 4 — deepseek-v4-pro,   thinking on"),
]


def respond(message, history):
    """День 1: отправляет историю диалога и новый вопрос в DeepSeek API, стримит ответ."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        yield ("Не задан DEEPSEEK_API_KEY. Скопируйте .env.example в .env "
               "и укажите ключ из https://platform.deepseek.com")
        return

    client = OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)
    messages = [{"role": "system", "content": SYSTEM_PROMPT}]
    messages += [{"role": m["role"], "content": m["content"]} for m in history]
    messages.append({"role": "user", "content": message})

    try:
        stream = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=messages,
            stream=True,
            extra_body=DEEPSEEK_EXTRA_BODY,
        )
    except Exception as exc:
        yield f"Ошибка при обращении к DeepSeek API: {exc}"
        return

    response = ""
    for chunk in stream:
        if chunk.choices and chunk.choices[0].delta.content:
            response += chunk.choices[0].delta.content
            yield response


def _get_client():
    """Создаёт OpenAI-совместимый клиент DeepSeek или бросает понятную ошибку."""
    api_key = os.getenv("DEEPSEEK_API_KEY")
    if not api_key:
        raise RuntimeError(
            "DEEPSEEK_API_KEY не задан. Скопируйте .env.example в .env "
            "и укажите ключ из https://platform.deepseek.com"
        )
    return OpenAI(api_key=api_key, base_url=DEEPSEEK_BASE_URL)


def compare_variants(question: str):
    """День 2: один и тот же вопрос уходит в DeepSeek дважды — без
    ограничений (вариант A) и с ограничениями (вариант B). Оба вызова
    логируются, JSON в варианте B парсится и валидируется."""
    logger.info("Day 2: question=%r", question)

    # --- Вариант A: без ограничений ---
    try:
        client = _get_client()
        a_resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            extra_body=DEEPSEEK_EXTRA_BODY,
        )
        a_text = a_resp.choices[0].message.content or ""
        logger.info("Variant A finish_reason=%s", a_resp.choices[0].finish_reason)
    except Exception as exc:
        a_text = f"Ошибка при обращении к DeepSeek API (вариант A): {exc}"
        logger.exception("Variant A failed")
    logger.info("Variant A response (%d chars): %s", len(a_text), a_text)

    # --- Вариант B: с ограничениями (max_tokens + stop + JSON-формат в промпте) ---
    try:
        client = _get_client()
        b_resp = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_B},
                {"role": "user", "content": question},
            ],
            max_tokens=MAX_TOKENS_B,
            stop=[STOP_SEQUENCE],
            extra_body=DEEPSEEK_EXTRA_BODY,
        )
        b_raw = b_resp.choices[0].message.content or ""
        # finish_reason == "stop" значит, что генерацию остановила стоп-
        # последовательность (или естественный конец ответа); "length" —
        # что упёрлись в max_tokens раньше, чем модель дошла до "###END###".
        b_finish_reason = b_resp.choices[0].finish_reason
    except Exception as exc:
        b_raw = f"Ошибка при обращении к DeepSeek API (вариант B): {exc}"
        logger.exception("Variant B failed")
        return a_text, b_raw, f"❌ **Ошибка API:** {exc}", None

    logger.info(
        "Variant B raw (%d chars, finish_reason=%s): %s",
        len(b_raw), b_finish_reason, b_raw,
    )

    # Параметр `stop` обычно отрезает стоп-последовательность на стороне API,
    # но разные провайдеры ведут себя по-разному — подстрахуемся и уберём
    # маркер вручную перед парсингом JSON.
    cleaned = b_raw
    if cleaned.endswith(STOP_SEQUENCE):
        cleaned = cleaned[: -len(STOP_SEQUENCE)].rstrip()

    # Проверка `json.loads()` — явный критерий приёмки дня 2.
    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.warning("Variant B JSON invalid: %s", exc)
        hint = (
            f" Причина: ответ обрезан по `max_tokens={MAX_TOKENS_B}` "
            "(finish_reason=length), модель не успела закрыть JSON."
            if b_finish_reason == "length"
            else ""
        )
        status_md = (
            f"❌ **JSON невалиден:** `{exc}`.{hint}\n\n"
            "Сырой ответ показан выше; в нём должен быть только JSON-объект "
            f"и (опционально) маркер `{STOP_SEQUENCE}`."
        )
        return a_text, b_raw, status_md, None

    logger.info("Variant B parsed OK: %s", parsed)
    status_md = (
        "✅ **JSON валиден** — `json.loads()` отработал без ошибок "
        f"(finish_reason=`{b_finish_reason}`)."
    )
    return a_text, b_raw, status_md, parsed


def run_day3(question: str):
    """День 3: один и тот же вопрос прогоняется через DeepSeek API четырьмя
    способами рассуждения. Возвращает кортеж из 5 строк для вывода рядом:
    четыре финальных ответа + сгенерированный промпт из шага 1 способа 3
    (его показываем отдельно, чтобы двухшаговая схема была прозрачной).

    Сравнение «какой способ точнее» код **не делает** — это ручное суждение
    автора по бумажной книге правил (см. §7 спецификации дня 3). Каждый вызов
    API логируется в stdout, как в дне 2; для способа 3 видны оба шага.
    """
    logger.info("Day 3: question=%r", question)

    out1 = ""
    out2 = ""
    out3 = ""
    out3_prompt = ""
    out4 = ""

    # --- Способ 1: прямой ответ (тот же SYSTEM_PROMPT, что в дне 1) ---
    try:
        client = _get_client()
        r = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT},
                {"role": "user", "content": question},
            ],
            extra_body=DEEPSEEK_EXTRA_BODY,
        )
        out1 = r.choices[0].message.content or ""
        logger.info(
            "Method 1 (direct) finish_reason=%s, %d chars: %s",
            r.choices[0].finish_reason, len(out1), out1,
        )
    except Exception as exc:
        out1 = f"❌ Ошибка API (способ 1): {exc}"
        logger.exception("Method 1 failed")

    # --- Способ 2: «решай пошагово» (рассуждение + финальный ответ абзацем) ---
    try:
        client = _get_client()
        r = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_STEP_BY_STEP},
                {"role": "user", "content": question},
            ],
            extra_body=DEEPSEEK_EXTRA_BODY,
        )
        out2 = r.choices[0].message.content or ""
        logger.info(
            "Method 2 (step by step) finish_reason=%s, %d chars: %s",
            r.choices[0].finish_reason, len(out2), out2,
        )
    except Exception as exc:
        out2 = f"❌ Ошибка API (способ 2): {exc}"
        logger.exception("Method 2 failed")

    # --- Способ 3: модель сама формулирует промпт, затем отвечает на него ---
    # Шаг 3.1: модель играет роль «мета-промптера» и генерирует запрос.
    try:
        client = _get_client()
        r = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": _build_meta_prompt(question)},
                {"role": "user", "content": question},
            ],
            extra_body=DEEPSEEK_EXTRA_BODY,
        )
        out3_prompt = (r.choices[0].message.content or "").strip()
        logger.info(
            "Method 3 step 1 (prompt generation) finish_reason=%s, %d chars: %s",
            r.choices[0].finish_reason, len(out3_prompt), out3_prompt,
        )
    except Exception as exc:
        out3_prompt = ""
        out3 = f"❌ Ошибка API (способ 3, шаг 1 — генерация промпта): {exc}"
        logger.exception("Method 3 step 1 failed")

    # Шаг 3.2: отправляем сгенерированный промпт отдельным запросом.
    # Базовая роль SYSTEM_PROMPT (как в способе 1), без «мета-промптерской»
    # обвязки — модель снова играет ассистента по правилам, но уже на
    # сгенерированном ей же запросе.
    if not out3 and out3_prompt:
        try:
            client = _get_client()
            r = client.chat.completions.create(
                model=DEEPSEEK_MODEL,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": out3_prompt},
                ],
                extra_body=DEEPSEEK_EXTRA_BODY,
            )
            out3 = r.choices[0].message.content or ""
            logger.info(
                "Method 3 step 2 (answer with generated prompt) "
                "finish_reason=%s, %d chars: %s",
                r.choices[0].finish_reason, len(out3), out3,
            )
        except Exception as exc:
            out3 = (
                "❌ Ошибка API (способ 3, шаг 2 — ответ на сгенерированный "
                f"промпт): {exc}"
            )
            logger.exception("Method 3 step 2 failed")
    elif not out3_prompt and not out3:
        out3 = (
            "❌ Шаг 1 способа 3 не вернул сгенерированный промпт — "
            "шаг 2 не выполнен."
        )

    # --- Способ 4: консилиум (Rules Lawyer / Game Designer / Skeptic) ---
    try:
        client = _get_client()
        r = client.chat.completions.create(
            model=DEEPSEEK_MODEL,
            messages=[
                {"role": "system", "content": SYSTEM_PROMPT_CONSILIUM},
                {"role": "user", "content": question},
            ],
            extra_body=DEEPSEEK_EXTRA_BODY,
        )
        out4 = r.choices[0].message.content or ""
        logger.info(
            "Method 4 (consilium) finish_reason=%s, %d chars: %s",
            r.choices[0].finish_reason, len(out4), out4,
        )
    except Exception as exc:
        out4 = f"❌ Ошибка API (способ 4): {exc}"
        logger.exception("Method 4 failed")

    return out1, out2, out3_prompt, out3, out4


def run_day4(question: str):
    """День 4: один и тот же вопрос прогоняется через DeepSeek API при трёх
    значениях `temperature` (`0.0` / `0.7` / `1.2`), по 3 повтора на каждое —
    итого 9 вызовов. Системный промпт и вопрос одинаковы во всех вызовах;
    `max_tokens`/`stop` из дня 2 не задаются, чтобы сравнивать эффект чистой
    температуры.

    Возвращает кортеж из 9 строк: индексы 0..2 — повторы при t=0.0, 3..5 —
    при t=0.7, 6..8 — при t=1.2. Каждый вызов логируется в stdout
    (temperature, номер повтора, finish_reason, длина и текст ответа) — тот
    же паттерн, что в днях 2-3. Никаких автоматических метрик или подсветки
    «лучшего» варианта — сравнение делает автор вручную (см. §7 спецификации).
    """
    logger.info("Day 4: question=%r", question)

    # Сюда сложим 9 ответов в порядке «по температурам, внутри — по повтору»:
    # 0..2 — t=0.0, 3..5 — t=0.7, 6..8 — t=1.2. Этот же порядок отдаём в UI
    # для раскладки «3 колонки по температурам × 3 строки по повторам».
    results: list[str] = [""] * (len(DAY4_TEMPERATURES) * DAY4_REPEATS_PER_TEMPERATURE)

    for t_idx, temperature in enumerate(DAY4_TEMPERATURES):
        for repeat in range(1, DAY4_REPEATS_PER_TEMPERATURE + 1):
            slot = t_idx * DAY4_REPEATS_PER_TEMPERATURE + (repeat - 1)
            label = f"temperature={temperature}, повтор {repeat}"
            try:
                client = _get_client()
                r = client.chat.completions.create(
                    model=DEEPSEEK_MODEL,
                    messages=[
                        {"role": "system", "content": SYSTEM_PROMPT},
                        {"role": "user", "content": question},
                    ],
                    temperature=temperature,
                    extra_body=DEEPSEEK_EXTRA_BODY,
                )
                text = r.choices[0].message.content or ""
                finish_reason = r.choices[0].finish_reason
                logger.info(
                    "Day 4 %s finish_reason=%s, %d chars: %s",
                    label, finish_reason, len(text), text,
                )
                # Подпись нужна и в UI, чтобы при сопоставлении 9 блоков
                # было ясно видно, какой именно вызов перед нами — код
                # подсветку «лучшего» не делает, но подписать каждый
                # ответ по его `temperature`/повтору обязан (см. §4 спецификации).
                results[slot] = (
                    f"**temperature={temperature}, повтор {repeat}** "
                    f"(finish_reason={finish_reason})\n\n"
                    f"{text}"
                )
            except Exception as exc:
                results[slot] = f"❌ **Ошибка API ({label}):** {exc}"
                logger.exception("Day 4 %s failed", label)

    return tuple(results)


def _estimate_cost_usd(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    """День 5: оценка стоимости одного вызова API в долларах по off-peak ценам
    DeepSeek V4 за 1M токенов (см. `PRICING_PER_M_TOKENS`). При `thinking=on`
    `completion_tokens` уже включает токены рассуждения — отдельный учёт не
    нужен, как и указано в §3 спецификации дня 5."""
    rates = PRICING_PER_M_TOKENS[model]
    return (
        prompt_tokens * rates["input"] / 1_000_000
        + completion_tokens * rates["output"] / 1_000_000
    )


def _green(value: str) -> str:
    """Обёртка для зелёного шрифта в Gradio Markdown (рендерит HTML)."""
    return f'<span style="color:#1a7f37;font-weight:bold">{value}</span>'


def _metric_md(label: str, cell: dict | None, mins: tuple | None = None) -> str:
    """День 5: markdown-блок метрик одной ячейки матрицы.

    `cell is None` — заявка ещё не отправлена (значение по умолчанию
    компонента до клика — единственный момент, когда пустые подписи
    видны, т.к. `run_day5` не стример и возвращает готовый результат
    сразу со всеми цифрами): показываем подписи с пустыми значениями.
    `mins` — `(min_elapsed, min_total_tokens, min_cost)` для зелёной
    подсветки минимумов по всем 4 ячейкам.
    """
    header = f"### {label}"
    if cell is None:
        return (
            f"{header}\n\n"
            f"- **⏱ Время ответа:** \n"
            f"- **🔢 Токены:** \n"
            f"- **💲 Стоимость:** \n"
            f"- **finish_reason:** "
        )
    if not cell["ok"]:
        return f"{header}\n\n❌ **Ошибка API:** {cell['error']}"

    elapsed_str = f"{cell['elapsed']:.2f} s"
    total_str = str(cell["total_tokens"])
    cost_str = f"${cell['cost']:.6f}"
    if mins is not None:
        min_elapsed, min_total_tokens, min_cost = mins
        if min_elapsed is not None and cell["elapsed"] == min_elapsed:
            elapsed_str = _green(elapsed_str)
        if min_total_tokens is not None and cell["total_tokens"] == min_total_tokens:
            total_str = _green(total_str)
        if min_cost is not None and cell["cost"] == min_cost:
            cost_str = _green(cost_str)

    return (
        f"{header}\n\n"
        f"- **⏱ Время ответа:** {elapsed_str}\n"
        f"- **🔢 Токены:** prompt={cell['prompt_tokens']} / "
        f"completion={cell['completion_tokens']} / total={total_str}\n"
        f"- **💲 Стоимость:** {cost_str}\n"
        f"- **finish_reason:** `{cell['finish_reason']}`"
    )


def run_day5(question: str):
    """День 5: один и тот же вопрос прогоняется через DeepSeek API на 4
    ячейках матрицы 2×2 (`deepseek-v4-flash`/`deepseek-v4-pro` ×
    `thinking off`/`on`). Системный промпт и вопрос одинаковы во всех
    вызовах; единственные переменные — модель и режим `thinking`,
    переключаемый через `extra_body={"thinking": {"type": "enabled"|
    "disabled"}}` (а не отдельным именем модели). `temperature`/`top_p`/
    `presence_penalty`/`frequency_penalty`/`max_tokens`/`stop` не задаются —
    это требование спецификации §3 (и в режиме `thinking=enabled` API их
    всё равно не поддерживает).

    Обычная (не генератор) функция — как `run_day3`/`run_day4`: все 4
    вызова выполняются последовательно внутри, а результат возвращается
    один раз в конце кортежем из 8 строк (`metric_1, answer_1, ...,
    metric_4, answer_4`). Это намеренно: генератор с промежуточными
    `yield` заставляет Gradio считать событие «стримящимся» и показывать
    вместо обычного индикатора ожидания только тонкую оранжевую рамку
    вокруг компонента, без спиннера и таймера. Обычная функция (как в
    дне 2) показывает стандартный оверлей Gradio на всё время вызова —
    крутящуюся иконку по центру `gr.Markdown` и таймер в углу
    `gr.Textbox`. Пока идёт вызов, во всех 4 ячейках виден `_metric_md`
    с пустыми значениями (то же самое значение стоит по умолчанию у
    `gr.Markdown` в интерфейсе) — реальные цифры и зелёная подсветка
    минимумов появляются во всех ячейках разом, когда функция вернёт
    результат.

    Наименьшее значение по каждой из трёх метрик (время, `total_tokens`,
    стоимость) подсвечено зелёным шрифтом — победители по разным
    метрикам могут быть разными ячейками. Автоматической метрики
    качества и подсветки «лучшего по качеству» код **не делает**
    (см. §7 спецификации) — это ручное суждение автора.
    """
    logger.info("Day 5: question=%r", question)

    cells: list[dict] = []
    for model, thinking_on, label in DAY5_MATRIX:
        cell: dict = {
            "model": model,
            "thinking_on": thinking_on,
            "label": label,
            "ok": False,
            "text": "",
            "elapsed": 0.0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "total_tokens": 0,
            "cost": 0.0,
            "finish_reason": None,
            "error": None,
        }
        try:
            client = _get_client()
            extra_body = (
                {"thinking": {"type": "enabled"}}
                if thinking_on
                else {"thinking": {"type": "disabled"}}
            )
            t0 = time.perf_counter()
            r = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": SYSTEM_PROMPT},
                    {"role": "user", "content": question},
                ],
                extra_body=extra_body,
            )
            elapsed = time.perf_counter() - t0

            text = r.choices[0].message.content or ""
            finish_reason = r.choices[0].finish_reason
            # `usage` теоретически может быть None (разные провайдеры
            # ведут себя по-разному) — `getattr` с дефолтом 0 страхует
            # от AttributeError, чтобы не ронять весь прогон матрицы из-за
            # одного вызова без usage.
            usage = r.usage
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or 0
            total_tokens = getattr(usage, "total_tokens", 0) or 0
            cost = _estimate_cost_usd(model, prompt_tokens, completion_tokens)

            cell.update(
                ok=True,
                text=text,
                elapsed=elapsed,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                total_tokens=total_tokens,
                cost=cost,
                finish_reason=finish_reason,
            )
            logger.info(
                "Day 5 %s model=%s thinking=%s finish_reason=%s time=%.2fs "
                "tokens(prompt/completion/total)=%d/%d/%d cost=$%.6f, "
                "%d chars: %s",
                label, model,
                "on" if thinking_on else "off",
                finish_reason, elapsed,
                prompt_tokens, completion_tokens, total_tokens, cost,
                len(text), text,
            )
        except Exception as exc:
            cell["error"] = exc
            logger.exception("Day 5 %s failed", label)

        cells.append(cell)

    # Минимумы считаем только среди успешных ячеек: если какая-то ячейка
    # упала, она просто не участвует в сравнении. Если все упали — None,
    # и подсветки не будет.
    successful = [c for c in cells if c["ok"]]
    if successful:
        mins = (
            min(c["elapsed"] for c in successful),
            min(c["total_tokens"] for c in successful),
            min(c["cost"] for c in successful),
        )
    else:
        mins = None

    # Пересобираем метрики всех 4 ячеек уже с зелёной подсветкой минимумов.
    # Сравнение по каждой метрике независимое (как требует §4 спецификации),
    # так что одна и та же ячейка может быть зелёной по времени, токенам и
    # стоимости сразу — или только по части метрик.
    result: list[str] = []
    for c in cells:
        result.append(_metric_md(c["label"], c, mins))
        result.append(c["text"] if c["ok"] else "")
    return tuple(result)


# --- Gradio-интерфейс: все дни живут в одном приложении через табы ---
with gr.Blocks(title="TooManyRules") as demo:
    gr.Markdown(
        "# TooManyRules\n"
        "Ассистент по правилам настольной игры Too Many Bones "
        "(Chip Theory Games). Без RAG, модель отвечает из общих знаний."
    )

    with gr.Tabs():
        # День 1: диалог в gr.ChatInterface — поведение не менялось.
        with gr.Tab("День 1 — чат"):
            gr.ChatInterface(
                respond,
                title="День 1: минимальный чат с DeepSeek",
                description=(
                    "Диалог с моделью через DeepSeek API. Системный промпт "
                    "задаёт роль ассистента по правилам Too Many Bones. "
                    "Без RAG, без ограничений на формат ответа."
                ),
                examples=[
                    "Какие есть дополнения у настольной игры Too Many Bones?",
                    "На сколько шагов ходят тираны?",
                    "Из каких фаз состоит ход игрока?",
                ],
            )

        # День 2: один вопрос → два ответа рядом (A без ограничений, B с ними).
        with gr.Tab("День 2 — формат ответа (A vs B)"):
            gr.Markdown(
                "Один и тот же вопрос уходит в DeepSeek API **дважды** с разной "
                "конфигурацией:\n\n"
                "- **Вариант A** — без ограничений: системный промпт из дня 1, "
                "без `max_tokens`/`stop`, без требований к формату.\n"
                "- **Вариант B** — с ограничениями: тот же вопрос, дополненный "
                "системный промпт требует строгий JSON (`rule_summary`, "
                "`details`, `confidence`) и маркер завершения "
                f"`{STOP_SEQUENCE}`; в запрос явно передаются "
                f"`max_tokens={MAX_TOKENS_B}` и `stop=['{STOP_SEQUENCE}']`.\n\n"
                "Оба вызова логируются в stdout (смотрите терминал, где "
                "запущен `app.py`)."
            )

            with gr.Row():
                question_input = gr.Textbox(
                    label="Вопрос по правилам",
                    value="Может ли гирлок ходить по диагонали?",
                    lines=2,
                    scale=4,
                )
                run_btn = gr.Button(
                    "Сравнить A и B", variant="primary", scale=1
                )

            gr.Examples(
                examples=[
                    ["Может ли гирлок ходить по диагонали?"],
                    [
                        "Еще вопрос\n"
                        "Толстая кожа на 2\n"
                        "В свой ход игнорирует первые два урона\n"
                        "Если тинк бросает один атаку а потом бот бьет на два\n"
                        "Пройдут же один урон?\n"
                        "Считается ход тинка?\n"
                        "И если есть еще один бот то плюс еще два урона?"
                    ],
                    [
                        "Из каких фаз состоит ход игрока?"
                    ],
                ],
                inputs=[question_input],
                label="Примеры вопросов (нажмите, чтобы подставить)",
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Вариант A — без ограничений")
                    output_a = gr.Textbox(
                        label="Ответ A (свободный текст)",
                        lines=18,
                        buttons=["copy"],
                    )
                with gr.Column():
                    gr.Markdown("### Вариант B — с ограничениями")
                    output_b_raw = gr.Textbox(
                        label="Ответ B — как пришёл от API",
                        lines=8,
                        buttons=["copy"],
                    )
                    output_b_status = gr.Markdown(
                        "Нажмите «Сравнить A и B», чтобы увидеть статус парсинга JSON."
                    )
                    output_b_json = gr.JSON(
                        label="Ответ B — распарсенный JSON"
                    )

            run_btn.click(
                compare_variants,
                inputs=[question_input],
                outputs=[output_a, output_b_raw, output_b_status, output_b_json],
            )
            question_input.submit(
                compare_variants,
                inputs=[question_input],
                outputs=[output_a, output_b_raw, output_b_status, output_b_json],
            )

        # День 3: один вопрос → четыре способа рассуждения (1/2/3/4).
        # Для способа 3 оба шага видны: маленький блок с промптом, который
        # модель сгенерировала сама, и большой блок с финальным ответом.
        with gr.Tab("День 3 — способы рассуждения"):
            gr.Markdown(
                "Один и тот же вопрос по правилам прогоняется через DeepSeek "
                "API **четырьмя способами рассуждения**, чтобы их можно было "
                "визуально сопоставить:\n\n"
                "- **Способ 1 — прямой ответ.** Базовый системный промпт из "
                "дня 1, без дополнительных инструкций про рассуждение.\n"
                "- **Способ 2 — «решай пошагово».** К базовому промпту "
                "добавлена инструкция: сначала расписать рассуждение, потом "
                "дать финальный ответ отдельным абзацем.\n"
                "- **Способ 3 — модель сама формулирует промпт.** Два "
                "последовательных вызова API: на первом модель генерирует "
                "идеальный промпт для разбора вопроса, на втором этот "
                "промпт отправляется отдельным запросом и модель отвечает "
                "уже на него. Оба вызова логируются в stdout.\n"
                "- **Способ 4 — консилиум экспертов.** В одном системном "
                "промпте описаны три персоны — Rules Lawyer, Game Designer "
                "и Skeptic, — которые отвечают последовательно, плюс "
                "обязательный итоговый вывод.\n\n"
                "Какой способ точнее — ручное суждение автора по бумажной "
                "книге правил; код его **не вычисляет** и не подсвечивает "
                "(см. §7 спецификации)."
            )

            with gr.Row():
                question_input_3 = gr.Textbox(
                    label="Вопрос по правилам",
                    value=(
                        "Бадди с Толстая кожа 2. Если Тинк бросает 1 атаку "
                        "и его бот бьет на два, "
                        "то сколько урона пройдет по бадди?"
                    ),
                    lines=2,
                    scale=4,
                )
                run_btn_3 = gr.Button(
                    "Сравнить 4 способа", variant="primary", scale=1
                )

            gr.Examples(
                examples=[
                    [
                        "Бадди с Толстая кожа 2. Если Тинк бросает 1 атаку "
                        "и его бот бьет на два, "
                        "то сколько урона пройдет по бадди?"
                    ],
                ],
                inputs=[question_input_3],
                label="Примеры вопросов (нажмите, чтобы подставить)",
            )

            with gr.Row():
                with gr.Column():
                    gr.Markdown("### Способ 1 — прямой ответ")
                    out_method1 = gr.Textbox(
                        label="Ответ способа 1",
                        lines=18,
                        buttons=["copy"],
                    )
                with gr.Column():
                    gr.Markdown("### Способ 2 — пошаговое рассуждение")
                    out_method2 = gr.Textbox(
                        label="Ответ способа 2",
                        lines=18,
                        buttons=["copy"],
                    )

            with gr.Row():
                with gr.Column():
                    gr.Markdown(
                        "### Способ 3 — модель сама формулирует промпт"
                    )
                    out_method3_prompt = gr.Textbox(
                        label="Шаг 1 — сгенерированный промпт",
                        lines=4,
                        buttons=["copy"],
                    )
                    out_method3 = gr.Textbox(
                        label="Шаг 2 — ответ на сгенерированный промпт",
                        lines=14,
                        buttons=["copy"],
                    )
                with gr.Column():
                    gr.Markdown("### Способ 4 — консилиум экспертов")
                    out_method4 = gr.Textbox(
                        label="Ответ способа 4",
                        lines=18,
                        buttons=["copy"],
                    )

            run_btn_3.click(
                run_day3,
                inputs=[question_input_3],
                outputs=[
                    out_method1, out_method2,
                    out_method3_prompt, out_method3,
                    out_method4,
                ],
            )
            question_input_3.submit(
                run_day3,
                inputs=[question_input_3],
                outputs=[
                    out_method1, out_method2,
                    out_method3_prompt, out_method3,
                    out_method4,
                ],
            )

        # День 4: один вопрос → 9 ответов (3 температуры × 3 повтора) для
        # визуального сопоставления эффекта `temperature`. Подписи
        # температуры и номера повтора выводятся внутри каждого блока,
        # чтобы при просмотре 9 ответов рядом сразу было видно, какой
        # именно вызов перед нами. Сравнение «какая температура лучше»
        # код не делает — это ручное суждение автора.
        with gr.Tab("День 4 — температура"):
            gr.Markdown(
                "Один и тот же вопрос по правилам прогоняется через DeepSeek "
                "API **девять раз**: 3 значения `temperature` "
                f"(`{DAY4_TEMPERATURES[0]}`, `{DAY4_TEMPERATURES[1]}`, "
                f"`{DAY4_TEMPERATURES[2]}`) × {DAY4_REPEATS_PER_TEMPERATURE} "
                "повтора. Системный промпт и вопрос одинаковы во всех 9 "
                "вызовах — единственная переменная это `temperature`. "
                "Повторы нужны, чтобы было видно разнообразие ответов при "
                "одной и той же температуре, а не только их содержание.\n\n"
                "Каждый вызов логируется в stdout (смотрите терминал, где "
                "запущен `app.py`): `temperature`, номер повтора, "
                "`finish_reason`, длина и текст ответа.\n\n"
                "Какая температура лучше подходит для каких задач — ручное "
                "суждение автора; код подсветку «лучшего» варианта **не "
                "делает** (см. §7 спецификации)."
            )

            with gr.Row():
                question_input_4 = gr.Textbox(
                    label="Вопрос по правилам",
                    value=(
                        "Объясни как работает Отравление в Too Many Bones "
                        "простыми словами, как будто объясняешь новому "
                        "игроку, и приведи короткий пример игровой ситуации, "
                        "описывающей это правило."
                    ),
                    lines=3,
                    scale=4,
                )
                run_btn_4 = gr.Button(
                    "Запустить 9 вызовов", variant="primary", scale=1
                )

            gr.Examples(
                examples=[
                    [
                        "Объясни как работает Отравление в Too Many Bones "
                        "простыми словами, как будто объясняешь новому "
                        "игроку, и приведи короткий пример игровой ситуации, "
                        "описывающей это правило."
                    ],
                ],
                inputs=[question_input_4],
                label="Примеры вопросов (нажмите, чтобы подставить)",
            )

            # 3 колонки — по одной на каждое значение `temperature`,
            # в каждой 3 текстовых блока с подписанным номером повтора.
            with gr.Row():
                with gr.Column():
                    gr.Markdown(f"### temperature = {DAY4_TEMPERATURES[0]}")
                    out_t0_r1 = gr.Textbox(
                        label="Повтор 1", lines=10, buttons=["copy"]
                    )
                    out_t0_r2 = gr.Textbox(
                        label="Повтор 2", lines=10, buttons=["copy"]
                    )
                    out_t0_r3 = gr.Textbox(
                        label="Повтор 3", lines=10, buttons=["copy"]
                    )
                with gr.Column():
                    gr.Markdown(f"### temperature = {DAY4_TEMPERATURES[1]}")
                    out_t1_r1 = gr.Textbox(
                        label="Повтор 1", lines=10, buttons=["copy"]
                    )
                    out_t1_r2 = gr.Textbox(
                        label="Повтор 2", lines=10, buttons=["copy"]
                    )
                    out_t1_r3 = gr.Textbox(
                        label="Повтор 3", lines=10, buttons=["copy"]
                    )
                with gr.Column():
                    gr.Markdown(f"### temperature = {DAY4_TEMPERATURES[2]}")
                    out_t2_r1 = gr.Textbox(
                        label="Повтор 1", lines=10, buttons=["copy"]
                    )
                    out_t2_r2 = gr.Textbox(
                        label="Повтор 2", lines=10, buttons=["copy"]
                    )
                    out_t2_r3 = gr.Textbox(
                        label="Повтор 3", lines=10, buttons=["copy"]
                    )

            # Порядок выходов в `run_day4` — 0..8, по температурам, внутри
            # по повтору (см. реализацию `run_day4`).
            day4_outputs = [
                out_t0_r1, out_t0_r2, out_t0_r3,
                out_t1_r1, out_t1_r2, out_t1_r3,
                out_t2_r1, out_t2_r2, out_t2_r3,
            ]
            run_btn_4.click(
                run_day4,
                inputs=[question_input_4],
                outputs=day4_outputs,
            )
            question_input_4.submit(
                run_day4,
                inputs=[question_input_4],
                outputs=day4_outputs,
            )

        # День 5: матрица 2×2 «модель × режим рассуждения» — 4 вызова на
        # одном и том же вопросе, чтобы увидеть раздельно эффект размера
        # модели (`flash` vs `pro`) и эффект включения рассуждения
        # (`thinking off` vs `on`). Режим `thinking` переключается через
        # `extra_body`, не через имя модели (требование §3 спецификации).
        with gr.Tab("День 5 — модель × thinking"):
            gr.Markdown(
                "Один и тот же вопрос по правилам прогоняется через DeepSeek "
                "API **четыре раза** на матрице 2×2 `модель × режим "
                "рассуждения`:\n\n"
                "- **Вариант 1:** `deepseek-v4-flash`, `thinking=off` — "
                "самый слабый/дешёвый/быстрый\n"
                "- **Вариант 2:** `deepseek-v4-flash`, `thinking=on`\n"
                "- **Вариант 3:** `deepseek-v4-pro`, `thinking=off`\n"
                "- **Вариант 4:** `deepseek-v4-pro`, `thinking=on` — "
                "самый сильный (флагман + рассуждение)\n\n"
                "Системный промпт и вопрос одинаковы во всех 4 вызовах. "
                "`temperature`/`top_p`/`presence_penalty`/`frequency_penalty`/"
                "`max_tokens`/`stop` **не задаются** — переменные это только "
                "модель и режим `thinking` "
                "(`extra_body={'thinking': {'type': 'enabled'|'disabled'}}`).\n\n"
                "Для каждого вызова показываются: время ответа, токены "
                "(`prompt`/`completion`/`total`) и оценочная стоимость в "
                "долларах по off-peak ценам DeepSeek V4 "
                "(`deepseek-v4-flash`: $0.22/$0.66, `deepseek-v4-pro`: "
                "$0.66/$1.98 за 1M input/output токенов; в пиковые часы "
                "ставки ×2 — это ориентир, а не точная выписка). "
                "**Зелёным шрифтом** выделено наименьшее значение по каждой "
                "из трёх метрик **независимо** (время, `total_tokens`, "
                "стоимость) — победители по разным метрикам могут быть "
                "разными ячейками.\n\n"
                "Каждый вызов логируется в stdout (смотрите терминал, где "
                "запущен `app.py`): модель, `thinking on/off`, "
                "`finish_reason`, время, токены, стоимость, длина и текст "
                "ответа.\n\n"
                "Сравнение **качества** ответов кодом **не делается** — это "
                "ручное суждение автора по бумажной книге правил "
                "(см. §7 спецификации)."
            )

            with gr.Row():
                question_input_5 = gr.Textbox(
                    label="Вопрос по правилам",
                    value=(
                        "Объясни как работает Отравление в Too Many Bones "
                        "простыми словами, как будто объясняешь новому "
                        "игроку, и приведи короткий пример игровой ситуации, "
                        "описывающей это правило."
                    ),
                    lines=3,
                    scale=4,
                )
                run_btn_5 = gr.Button(
                    "Запустить 4 вызова", variant="primary", scale=1
                )

            gr.Examples(
                examples=[
                    [
                        "Объясни как работает Отравление в Too Many Bones "
                        "простыми словами, как будто объясняешь новому "
                        "игроку, и приведи короткий пример игровой ситуации, "
                        "описывающей это правило."
                    ],
                ],
                inputs=[question_input_5],
                label="Примеры вопросов (нажмите, чтобы подставить)",
            )

            # Раскладка 2×2: строки — режим `thinking` (off/on), колонки —
            # модель (flash/pro). Это прямой аналог таблицы из §2 спецификации.
            #
            # В каждой ячейке два компонента: `Markdown` для метрик
            # (короткий блок, сюда попадает зелёная подсветка минимумов)
            # и `Textbox` для текста ответа. Значение по умолчанию для
            # `Markdown` — тот же `_metric_md(label, cell=None)`, что
            # используется внутри `run_day5`: подписи и пустые значения
            # метрик видны сразу, даже до первого клика.
            # `max_lines=lines` фиксирует высоту Textbox: без него он
            # растягивается под самый длинный ответ (у `thinking=on`
            # ответы обычно короче promptа, но при пустом `content` и
            # длинном `reasoning` бывают многократные расхождения в длине),
            # и 4 ячейки получались разной высоты. С `max_lines` высота
            # всех 4 ячеек одинаковая, лишний текст скроллится внутри.
            with gr.Row():
                with gr.Column():
                    out5_metric_1 = gr.Markdown(
                        value=_metric_md(DAY5_MATRIX[0][2], cell=None)
                    )
                    out5_answer_1 = gr.Textbox(
                        label="Ответ",
                        lines=12,
                        max_lines=12,
                        placeholder="⏳ Ожидание...",
                        buttons=["copy"],
                    )
                with gr.Column():
                    out5_metric_2 = gr.Markdown(
                        value=_metric_md(DAY5_MATRIX[1][2], cell=None)
                    )
                    out5_answer_2 = gr.Textbox(
                        label="Ответ",
                        lines=12,
                        max_lines=12,
                        placeholder="⏳ Ожидание...",
                        buttons=["copy"],
                    )
            with gr.Row():
                with gr.Column():
                    out5_metric_3 = gr.Markdown(
                        value=_metric_md(DAY5_MATRIX[2][2], cell=None)
                    )
                    out5_answer_3 = gr.Textbox(
                        label="Ответ",
                        lines=12,
                        max_lines=12,
                        placeholder="⏳ Ожидание...",
                        buttons=["copy"],
                    )
                with gr.Column():
                    out5_metric_4 = gr.Markdown(
                        value=_metric_md(DAY5_MATRIX[3][2], cell=None)
                    )
                    out5_answer_4 = gr.Textbox(
                        label="Ответ",
                        lines=12,
                        max_lines=12,
                        placeholder="⏳ Ожидание...",
                        buttons=["copy"],
                    )

            # Порядок: metric_1, answer_1, metric_2, answer_2, ... —
            # такой же, как `run_day5` собирает результат.
            day5_outputs = [
                out5_metric_1, out5_answer_1,
                out5_metric_2, out5_answer_2,
                out5_metric_3, out5_answer_3,
                out5_metric_4, out5_answer_4,
            ]
            # show_progress не переопределяется — остаётся дефолтный
            # "full", тот же, что и у дня 2. Важно, что `run_day5` — не
            # генератор (см. её docstring): Gradio показывает нормальный
            # оверлей ожидания (спиннер по центру `Markdown`, таймер в
            # углу `Textbox`) только для функций с одним `return`; у
            # генератора вместо этого просто тонкая рамка вокруг
            # компонента без спиннера и таймера.
            run_btn_5.click(
                run_day5,
                inputs=[question_input_5],
                outputs=day5_outputs,
            )
            question_input_5.submit(
                run_day5,
                inputs=[question_input_5],
                outputs=day5_outputs,
            )


if __name__ == "__main__":
    demo.launch()
