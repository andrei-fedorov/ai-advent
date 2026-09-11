# TooManyRules — конфиги агентов проекта (день 6, неделя 2).
#
# Здесь живёт всё «про Too Many Bones»: системный промпт проекта и набор
# пресетов. Смысл набора — показать, что то, что на неделе 1 было разными
# кусками кода (вариант B дня 2, температура дня 4, флагман с thinking
# дня 5), теперь отличается только конфигом одного и того же агента.
#
# Вызовов LLM API здесь нет и быть не должно: всё общение с моделью — в
# `agent.py`. Направление зависимостей: app.py → presets.py → agent.py.
#
# Тексты промптов скопированы из `app_week1.py`: импортировать из
# замороженного артефакта недели 1 нельзя, дублирование здесь — осознанная
# цена заморозки (см. `docs/TooManyRules — Неделя 2 архитектура.md`, §2).

from agent import DEEPSEEK_MODEL_PRO, AgentConfig

# Системный промпт дня 1 — базовая роль ассистента, общая для всех пресетов.
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

# Надстройка варианта B дня 2: та же роль плюс строгие требования к формату,
# длине и явному маркеру завершения.
SYSTEM_PROMPT_JSON = SYSTEM_PROMPT + """

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

# Ограничения варианта B дня 2, заданные не текстом промпта, а параметрами
# запроса. 350 токенов — значение, подобранное в дне 2: кириллица
# токенизируется хуже английского, и на 150 модель обрывалась посреди JSON.
STOP_SEQUENCE = "###END###"
MAX_TOKENS_JSON = 350

# День 4: температура, на которой ответы заметно расходятся между собой.
CREATIVE_TEMPERATURE = 1.2

# Пресеты: имя → конфиг. Имя пресета совпадает с `AgentConfig.name`, чтобы
# в логах и дебаг-панели было видно ровно то, что выбрано в интерфейсе.
PRESETS: dict[str, AgentConfig] = {
    "Базовый": AgentConfig(
        name="Базовый",
        system_prompt=SYSTEM_PROMPT,
        description=(
            "день 1 — базовая роль, `deepseek-flash`, thinking off, "
            "никаких ограничений формата"
        ),
    ),
    "Строгий JSON": AgentConfig(
        name="Строгий JSON",
        system_prompt=SYSTEM_PROMPT_JSON,
        max_tokens=MAX_TOKENS_JSON,
        stop=[STOP_SEQUENCE],
        description=(
            "день 2, вариант B — промпт требует JSON, "
            f"`max_tokens={MAX_TOKENS_JSON}`, `stop=['{STOP_SEQUENCE}']`; "
            "ответ уходит в чат сырым текстом, парсинг JSON — работа дня 2 "
            "и здесь не повторяется"
        ),
    ),
    "Креативный": AgentConfig(
        name="Креативный",
        system_prompt=SYSTEM_PROMPT,
        temperature=CREATIVE_TEMPERATURE,
        description=f"день 4 — тот же промпт, `temperature={CREATIVE_TEMPERATURE}`",
    ),
    "Флагман + thinking": AgentConfig(
        name="Флагман + thinking",
        system_prompt=SYSTEM_PROMPT,
        model=DEEPSEEK_MODEL_PRO,
        thinking=True,
        description=(
            f"день 5 — `{DEEPSEEK_MODEL_PRO}` с включённым режимом рассуждения; "
            "`reasoning_content` виден в дебаг-панели, дороже и медленнее"
        ),
    ),
}

DEFAULT_PRESET = "Базовый"
