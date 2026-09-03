# TooManyRules — день 1: минимальный веб-чат с DeepSeek API.
# Запуск: python app.py (после pip install -r requirements.txt и настройки .env)

import os

import gradio as gr
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

DEEPSEEK_BASE_URL = "https://api.deepseek.com"
DEEPSEEK_MODEL = "deepseek-chat"

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


def respond(message, history):
    """Отправляет историю диалога и новый вопрос в DeepSeek API, стримит ответ."""
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


demo = gr.ChatInterface(
    respond,
    title="TooManyRules",
    description="Ассистент по правилам настольной игры Too Many Bones (Chip Theory Games).",
    examples=[
        "Какие есть дополнения у настольной игры Too Many Bones?",
        "На сколько шагов ходят тираны?",
        "Из каких фаз состоит ход игрока?",
    ],
)

if __name__ == "__main__":
    demo.launch()
