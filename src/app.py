# TooManyRules — день 1 (чат с DeepSeek), день 2 (формат ответа A vs B),
# день 3 (четыре способа рассуждения на одной и той же задаче).
# Запуск: python app.py (после pip install -r requirements.txt и настройки .env)

import json
import logging
import os

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
DEEPSEEK_MODEL = "deepseek-chat"

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


if __name__ == "__main__":
    demo.launch()
