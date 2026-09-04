# TooManyRules — день 1 (чат с DeepSeek) + день 2 (сравнение формата ответа A vs B).
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


# --- Gradio-интерфейс: оба дня живут в одном приложении через табы ---
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


if __name__ == "__main__":
    demo.launch()
