# TooManyRules — приложение (день 9: чат через агента + дебаг-панель; день 10:
# четыре стратегии, checkpoint'ы и ветки диалога; день 11, неделя 3: модель
# памяти — три слоя, кандидаты в долговременную память и переключатель слоёв
# в запросе; день 12: профиль пользователя, роутер режима и переключатель
# «Профиль в запросе»; день 13: состояние задачи; день 14: инварианты).
#
# Здесь только интерфейс. LLM-логики в этом файле нет: ни клиента OpenAI,
# ни chat.completions.create — всё общение с моделью инкапсулировано в
# `agent.Agent`, конфиги агентов проекта лежат в `presets.py`, история
# диалогов, долговременная память и профиль — в `storage.py`, счёт токенов —
# в `tokens.py` (оттуда берутся только чистые функции: генератор заполнителя и
# оценка его размера, всё остальное панель получает от агента готовым),
# стратегии управления контекстом — в `context.py`, и напрямую он отсюда тоже
# не импортируется: набор стратегий агенту выдаёт `presets.make_strategies()`.
# Из `memory.py` интерфейс берёт только имена слоёв (пункты переключателя) и
# чистую `long_term_text()` для подраздела долговременной памяти: модель памяти
# агенту выдаёт `presets.make_memory()`. Из `user_profile.py` — разбор,
# проверку и текст профиля для редактора и панели: роутер агенту выдаёт
# `presets.make_router()`.
#
# День 11 (спецификация дня 11, §7): интерфейс остаётся видом на состояние
# агента — ни своей копии рабочей памяти, ни своего списка кандидатов, ни
# своего кэша долговременной памяти. Всё приходит из агента и хранилища на
# каждом событии. Панель не знает, какие ключи бывают: рабочая память и
# кандидаты рисуются из `MemoryState`, долговременная — из записей хранилища.
# Запись в долговременную память — только кнопкой «Сохранить в
# долговременную»: по умолчанию ни один кандидат не отмечен.
#
# День 12 (спецификация дня 12, §7) сохраняет это правило почти везде, с
# одним осознанным исключением: **редактор профиля — форма, а не вид.** Его
# поля не входят в `_view()`/`VIEW_OUTPUTS` — иначе отправка вопроса или
# смена стратегии затирали бы несохранённую правку. У редактора свои выходы
# (`PROFILE_EDITOR_OUTPUTS`) и свои обработчики, а профиль в запросе, режим и
# переключатель по-прежнему приходят из агента и хранилища на каждом событии.
#
# День 14 (спецификация дня 14, §8) держит то же правило и то же исключение:
# книга инвариантов, последний конфликт и ожидание подтверждения перехода
# приходят из `debug_state()["invariants"]`, постоянные инварианты — из
# хранилища, таблица переходов — из `task_state.TRANSITIONS`; своей копии у
# интерфейса нет. Исключение одно, как у профиля: **редактор инвариантов — форма,
# а не вид** (свои выходы `INVARIANT_EDITOR_OUTPUTS`, свой `demo.load`). Из
# `invariants.py` интерфейс берёт разбор, проверку и запись инвариантов для
# редактора и подписи областей: книгу агенту выдаёт `presets.make_invariants()`.
#
# Панель не знает, какие бывают стратегии и что такое сводка или факты: она
# рисует то, что вернули `debug_state()` и `ContextView` — имя, описание
# словами, текст памяти и числа. День 10 добавил в переключатель ещё две
# стратегии и не тронул код, который их рисует, — но тронул слова: «свёртка»,
# «сводка», «несвёрнутых» были зашиты в тексты панели, и для окна и фактов это
# было неправдой (спецификация дня 10, §7.1). Сегодня в текстах интерфейса нет
# ни одного слова, верного только для одной стратегии.
#
# Интерфейс — это вид на состояние агента: содержимое чата на каждом шаге
# рендерится из `agent.history`, своей копии переписки Gradio не ведёт.
# Ровно поэтому день 7 не потребовал переделки интерфейса: история приходит
# из файла, а чат рендерится из того же `agent.history`, что и раньше.
#
# Агентов в процессе много: переключатель в дебаг-панели листает реестр
# `agent.agents()`, и выбранный агент становится активным — вместе с ним
# переключаются чат, стек сообщений, конфиг и метрики. С дня 7 агенты
# переживают перезапуск: при старте процесса они поднимаются по файлам
# сессий, и открытая страница сразу показывает диалог.
#
# День 10 добавляет операцию над самой историей — checkpoint и ветку: ветка
# диалога — это **отдельная сессия и отдельный агент**, а не пункт списка
# стратегий (спецификация дня 10, §2.2). У неё свой переключатель («Ветка
# диалога»), устроенный так же, как переключатель агентов.
#
# Вкладок больше нет: дни недели 2 наращивают одну и ту же сущность, и
# «что нового сегодня» показывает дебаг-панель. Дни 1-5 живут в
# замороженном артефакте `app_week1.py` (запуск: ./run.sh week1).
#
# Запуск: ./run.sh (или python app.py из src/ с активированным .venv)

import functools
import logging
from dataclasses import asdict, replace
from datetime import datetime

import gradio as gr
import pandas as pd

from agent import (
    Agent,
    agent_by_number,
    agents,
    branch_family,
    branch_parent,
    delete_agent,
    estimate_cost_usd,
    process_stats,
)
from invariants import (
    ID_PREFIX,
    SCOPE_ALWAYS,
    SCOPE_LABELS,
    SCOPE_SESSION,
    SCOPES,
    Invariant,
    dump_invariants,
    parse_invariants,
    validate,
)
from memory import LAYER_LONG_TERM, LAYER_WORKING, REQUEST_LAYERS, long_term_text
from presets import (
    BRANCH_QUESTION,
    BRANCH_STEPS,
    COMPARISON_SCENARIO,
    CONTROL_QUESTIONS,
    DEFAULT_INVARIANTS,
    DEFAULT_PRESET,
    DEFAULT_PROFILE,
    DEFAULT_STRATEGY,
    INVARIANT_EXAMPLES,
    INVARIANT_MAX_ITEMS,
    INVARIANT_SCENARIO,
    INVARIANT_TEXT_WORDS,
    MEMORY_CONTROL_QUESTIONS,
    MEMORY_MAP,
    MEMORY_SCENARIO,
    NEXT_SESSION_QUESTIONS,
    PRESETS,
    PROFILE_QUESTION,
    PROFILE_VARIANTS,
    ROUTING_SCENARIO,
    STRATEGIES,
    TASK_SCENARIO,
    make_invariants,
    make_memory,
    make_router,
    make_strategies,
    make_task_machine,
)
from storage import (
    JsonHistoryStore,
    JsonInvariantStore,
    JsonLongTermStore,
    JsonProfileStore,
    StorageError,
    display_path,
)
from task_state import (
    EVENT_CANCEL,
    EVENT_PAUSE,
    EVENT_RESUME,
    EVENTS,
    GUARD_MODES,
    MAIN_PATH,
    NO_TASK,
    STAGE_CANCELLED,
    STAGE_DONE,
    STAGE_EXECUTION,
    STAGE_VALIDATION,
    TRANSITIONS,
)
from tokens import FILLER_MAX_TOKENS, estimate_tokens, filler_text
from user_profile import (
    CHOICE_AUTO,
    CHOICE_OFF,
    dump_profile,
    mode_by_name,
    parse_profile,
    profile_block,
    profile_changes,
    profile_text,
    validate_profile,
)

# Логирование настраивает точка входа — тот же формат, что на неделе 1.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)
logger = logging.getLogger("toomanyrules.app")

# Хранилище одно на процесс и общее для всех агентов: `session_id` разводит
# их по файлам. Создаётся на модуле, а не в обработчиках, — иначе номера
# сессий выдавали бы несколько независимых экземпляров.
STORE = JsonHistoryStore()

# Долговременная память (день 11, §7.1) — тоже одна на процесс и общая для
# всех агентов, рядом со `STORE`: её файл общий для всех сессий. Путь к
# файлу хранилище печатает в лог при создании — по нему видно, что экземпляр,
# поднятый с `TOOMANYRULES_DATA_DIR`, не пишет в память автора.
LONG_TERM = JsonLongTermStore()

# Профиль пользователя (день 12, §7.1) — тем же приёмом: один на процесс,
# общий для всех агентов, рядом со `STORE` и `LONG_TERM`. Профиль по
# умолчанию хранилище получает параметром — `storage.py` по-прежнему не
# импортирует ничего из проекта.
PROFILE = JsonProfileStore(default=dump_profile(DEFAULT_PROFILE))
# Замечания разбора профиля по умолчанию/сохранённого файла — один раз за
# процесс, а не на каждый ход (§5.1: агент их не логирует).
for _profile_note in parse_profile(PROFILE.current())[1]:
    logger.warning("профиль при старте: %s", _profile_note)

# Постоянные инварианты (день 14, §8.1) — тем же приёмом: один на процесс,
# общий для всех агентов, рядом со `STORE`, `LONG_TERM` и `PROFILE`. Список по
# умолчанию хранилище получает параметром — `storage.py` по-прежнему не
# импортирует ничего из проекта.
INVARIANTS = JsonInvariantStore(default=dump_invariants(DEFAULT_INVARIANTS))
# Путь, число инвариантов и замечания разбора — один раз за процесс, как у
# долговременной памяти и профиля (§6.10): по строке видно, что экземпляр,
# поднятый с `TOOMANYRULES_DATA_DIR`, не пишет в инварианты автора.
_start_invariants, _start_notes = parse_invariants(
    INVARIANTS.current(), EVENTS, GUARD_MODES, scope=SCOPE_ALWAYS,
    max_items=INVARIANT_MAX_ITEMS,
)
logger.info(
    "инварианты при старте: %d (%d действуют) ← %s%s",
    len(_start_invariants), sum(1 for inv in _start_invariants if inv.active),
    INVARIANTS.path, "" if INVARIANTS.saved else " — файла нет, по умолчанию",
)
for _invariant_note in _start_notes:
    logger.warning("инварианты при старте: %s", _invariant_note)

# Сколько агентов поднялось из файлов при старте процесса — заполняется
# `_restore_agents()` ниже и показывается в строке статуса при открытии
# страницы.
RESTORED_AT_START = 0


# --- Рендер дебаг-панели -------------------------------------------------

def _fmt_cost(cost_usd: float | None) -> str:
    return "н/д" if cost_usd is None else f"${cost_usd:.6f}"


def _fmt_tokens(value: int | None) -> str:
    return "н/д" if value is None else str(value)


def _fmt_int(value: int | None) -> str:
    """Число с разделителями разрядов: в бюджете контекста числа семизначные,
    и без пробелов их не прочитать."""
    return "н/д" if value is None else f"{value:,}".replace(",", " ")


def _fmt_delta(estimated: int | None, fact: int | None) -> str:
    """Расхождение оценки с фактом в процентах, со знаком. Нигде не хранится —
    считается из двух чисел, которые уже есть (агент делает так же в логе)."""
    if estimated is None or not fact:
        return "н/д"
    return f"{(estimated - fact) / fact * 100:+.1f}%"


def _fmt_saved(saved: int, full_total: int) -> str:
    """Экономия в процентах от полного контекста, со знаком. Отрицательная
    экономия — нормальный результат на коротком диалоге, и прячут её только
    фокусники."""
    if full_total <= 0:
        return "н/д"
    sign = "−" if saved >= 0 else "+"
    return f"{sign}{abs(saved) / full_total * 100:.0f}%"


def _agent_title(agent: Agent) -> str:
    """Как агент называется в подписях и статусах: номер отличает инстансы
    друг от друга, имя пресета говорит, чем они отличаются."""
    return f"#{agent.number} «{agent.config.name}»"


def _agent_choices(active: Agent) -> list[tuple[str, int]]:
    """Варианты переключателя — все агенты процесса в порядке создания.
    Значение варианта это порядковый номер, а не сам агент: в `gr.Dropdown`
    попадают только сериализуемые значения, экземпляр берётся по номеру из
    реестра. Размер стека в подписи сразу показывает, что переписка у каждого
    агента своя.

    Активного агента могли удалить из другой вкладки: объект жив, пока эта
    вкладка на него ссылается, но в реестре его уже нет. Тогда он идёт
    отдельным пунктом с пометкой — иначе значение списка осталось бы без
    варианта.
    """
    registry = agents()
    entries = [(agent, "") for agent in registry]
    if active not in registry:
        entries.insert(0, (active, " · удалён"))
    return [
        (
            f"#{agent.number} · {agent.config.name} · {agent.session_id}"
            # День 10: подпись дополняется происхождением, если это ветка.
            + (f" · ветка от {agent.branch.parent}" if agent.branch else "")
            + f" · {len(agent.history)} сообщ.{suffix}",
            agent.number,
        )
        for agent, suffix in entries
    ]


def _time_of(iso: str) -> str:
    """Время из ISO-строки (`created_at` checkpoint'а/ветки) — только часы,
    минуты и секунды: панель уже показывает сессию и её дату отдельно."""
    return iso.split("T", 1)[1] if "T" in iso else iso


def _metrics_md(last_call: dict | None, invariants_state: dict | None = None) -> str:
    """Метрики последнего вызова агента. `invariants_state` —
    `debug_state()["invariants"]` (день 14): нужен только для строки «страж не
    вызывался»."""
    header = "### Последний вызов"
    if last_call is None:
        return (
            f"{header}\n\n"
            "Вызовов ещё не было — задайте вопрос, и здесь появятся время, "
            "`finish_reason`, токены и стоимость."
        )
    if not last_call["ok"]:
        lines = [
            header,
            "",
            f"❌ **Ошибка:** {last_call['error']}",
            "",
            "Стек сообщений при ошибке не меняется — вопрос можно отправить "
            "повторно, история не задвоится.",
        ]
        # Применённое обновление памяти перед упавшим вызовом оплачено и уже
        # применено, а строки журнала у несостоявшегося хода нет: не
        # показать её здесь — значит не показать нигде, кроме накопленных
        # счётчиков.
        service = _service_lines(last_call.get("service_call"))
        # Страж инвариантов (день 14) — тем же правилом, перед разбором
        # памяти: в порядке служебных работ хода.
        service += _guard_call_lines(last_call, invariants_state)
        # Разбор памяти (день 11) — тем же правилом: применённый до упавшего
        # вызова разбор оплачен и сохранён, и показать его больше негде.
        service += _memory_call_lines(last_call.get("memory_call"))
        # Роутер профиля (день 12) — тем же правилом.
        service += _route_call_lines(last_call)
        # Трекер задачи (день 13) — тем же правилом.
        service += _task_tracker_lines(last_call)
        if service:
            lines += ["", *service]
        return "\n".join(lines)
    request = last_call.get("request_tokens") or {}
    estimated_prompt = request.get("total")
    lines = [
        header,
        "",
        f"- **⏱ Время ответа:** {last_call['elapsed']:.2f} s",
        f"- **🔢 Токены:** prompt={_fmt_tokens(last_call['prompt_tokens'])} / "
        f"completion={_fmt_tokens(last_call['completion_tokens'])} / "
        f"total={_fmt_tokens(last_call['total_tokens'])}",
        # Две строки дня 8: оценка и факт стоят рядом по обе стороны вызова —
        # до запроса точного числа не бывает, и видно, насколько мы промахнулись.
        f"- **📏 Запрос:** оценка ≈{_fmt_int(estimated_prompt)} против факта "
        f"{_fmt_int(last_call['prompt_tokens'])} "
        f"({_fmt_delta(estimated_prompt, last_call['prompt_tokens'])})",
        f"- **✍️ Ответ модели:** {_fmt_int(last_call['completion_tokens'])} "
        f"токенов, оценка по тексту "
        f"≈{_fmt_int(last_call['estimated_completion_tokens'])} "
        f"({_fmt_delta(last_call['estimated_completion_tokens'], last_call['completion_tokens'])})",
    ]
    # Кэш промпта показывается, только если API его вернул. Стоимость по нему
    # не пересчитывается — `PRICING_PER_M_TOKENS` остаётся off-peak-оценкой.
    if last_call["prompt_cache_hit_tokens"] is not None:
        lines.append(
            f"- **♻️ Кэш промпта:** из кэша "
            f"{_fmt_int(last_call['prompt_cache_hit_tokens'])} / новых "
            f"{_fmt_int(last_call['prompt_cache_miss_tokens'])} — оценка "
            f"стоимости кэш не учитывает, поэтому счёт растёт медленнее неё"
        )
    lines += [
        f"- **💲 Стоимость:** {_fmt_cost(last_call['cost_usd'])}",
        f"- **finish_reason:** `{last_call['finish_reason']}`",
        f"- **Модель:** `{last_call['model']}`",
    ]
    lines += _service_lines(last_call.get("service_call"))
    # Страж инвариантов (день 14, §8.6) — перед разбором памяти, в порядке
    # служебных работ хода.
    lines += _guard_call_lines(last_call, invariants_state)
    lines += _memory_call_lines(last_call.get("memory_call"))
    lines += _route_call_lines(last_call)
    lines += _task_tracker_lines(last_call)
    return "\n".join(lines)


def _guard_call_lines(last_call: dict, invariants_state: dict | None) -> list[str]:
    """Страж инвариантов этого хода (день 14, §8.6) — строкой следом за
    служебным вызовом стратегии и перед разбором памяти, по образцу
    `_service_lines()`. Сбой и неразобранный ответ показываются здесь же и за
    ошибку ответа не выдаются: ход состоялся, конфликт этого сообщения просто
    не проверен.

    Страж не вызывался, а инварианты есть, — строка про то, почему: причину
    знает книга (`guard_note`), панель её не выдумывает."""
    call = last_call.get("guard_call")
    if call is None:
        if not invariants_state:
            return []
        view = invariants_state["view"]
        if not view["active"] or not view["guard_note"]:
            return []
        return [
            f"- **🛡 Инварианты:** {view['active']} действуют — страж не "
            f"вызывался: {view['guard_note']}"
        ]
    if not call["ok"]:
        line = f"- **🛡 Страж инвариантов не удался — {call['label']}:** {call['error']}"
        if call["text"]:
            line += f" (начало ответа: «{call['text'][:120]}»)"
        line += (
            f" — ход состоялся, конфликт этого сообщения не проверен. "
            f"Потрачено {_fmt_int(call['total_tokens'] or 0)} токенов."
        )
        return [line]
    line = (
        f"- **🛡 Страж инвариантов — {call['label']}:** {call['memory_update']}; "
        f"ответ ≈{_fmt_int(estimate_tokens(call['text']))} токенов, вызов "
        f"{_fmt_int(call['total_tokens'])} токенов, "
        f"{_fmt_cost(call['cost_usd'])}, {call['elapsed']:.2f} s"
    )
    if call.get("finish_reason") == "length":
        line += (
            " — ⚠️ **ответ упёрся в потолок `max_tokens` и оборван на "
            "полуслове**, применён как есть"
        )
    return [line]


def _service_lines(service: dict | None) -> list[str]:
    """Служебный вызов этого хода — отдельной строкой в «Последнем вызове».

    `None` означает, что стратегия ничего не просила. Сбой (или неразобранный
    ответ) показывается здесь же и за ошибку ответа не выдаётся: ход
    состоялся, ответ игроку пришёл. Панель не знает, какая стратегия сделала
    вызов и что за память он обновляет, — слова приходят из `service['label']`
    и `service['memory_label']`/`service['memory_update']` (день 10, §5.1), в
    том числе когда стратегию после хода уже переключили.
    """
    if not service:
        return []
    if not service["ok"]:
        line = (
            f"- **🧵 Служебный вызов не удался — {service['label']}:** "
            f"{service['error']}"
        )
        if service["text"]:
            # Ответ, который не разобрался, — начало текста, чтобы было
            # видно, что именно не разобралось.
            line += f" (начало ответа: «{service['text'][:120]}»)"
        line += (
            f" — память не сдвинулась, ход состоялся, в запрос ушло всё "
            f"неучтённое. Потрачено {_fmt_int(service['total_tokens'] or 0)} "
            f"токенов."
        )
        return [line]
    line = (
        f"- **🧵 Служебный вызов — {service['label']}:** "
        f"{service['memory_label']} — {service['memory_update']}; ответ "
        f"≈{_fmt_int(estimate_tokens(service['text']))} токенов, вызов "
        f"{_fmt_int(service['total_tokens'])} токенов, "
        f"{_fmt_cost(service['cost_usd'])}, {service['elapsed']:.2f} s"
    )
    if service.get("finish_reason") == "length":
        # Обрезанный ответ применён, но выдавать его за нормальный нельзя:
        # штатно в потолок служебного вызова упираться не должно.
        line += (
            " — ⚠️ **ответ упёрся в потолок `max_tokens` и оборван на "
            "полуслове**, применён как есть"
        )
    return [line]


def _memory_call_lines(call: dict | None) -> list[str]:
    """Разбор памяти этого хода (день 11, §7.4) — строкой следом за служебным
    вызовом стратегии, по образцу `_service_lines()`. `None` — разбора не
    было. Сбой и неразобранный ответ показываются здесь же и за ошибку
    ответа не выдаются: ход состоялся."""
    if not call:
        return []
    if not call["ok"]:
        line = f"- **🧠 Разбор памяти не удался — {call['label']}:** {call['error']}"
        if call["text"]:
            line += f" (начало ответа: «{call['text'][:120]}»)"
        line += (
            f" — рабочая память и кандидаты не сдвинулись, ход состоялся; "
            f"разбор повторится на следующем ходе. Потрачено "
            f"{_fmt_int(call['total_tokens'] or 0)} токенов."
        )
        return [line]
    line = (
        f"- **🧠 Разбор памяти — {call['label']}:** {call['memory_update']}; "
        f"ответ ≈{_fmt_int(estimate_tokens(call['text']))} токенов, вызов "
        f"{_fmt_int(call['total_tokens'])} токенов, "
        f"{_fmt_cost(call['cost_usd'])}, {call['elapsed']:.2f} s"
    )
    if call.get("finish_reason") == "length":
        line += (
            " — ⚠️ **ответ упёрся в потолок `max_tokens` и оборван на "
            "полуслове**, применён как есть"
        )
    return [line]


def _route_call_lines(last_call: dict) -> list[str]:
    """Роутер профиля этого хода (день 12, §7.5) — строкой следом за
    разбором памяти, по образцу `_service_lines()`/`_memory_call_lines()`.

    Режим хода виден и без вызова роутера — выбран вручную, единственный
    режим профиля или профиль выключен, — тогда строка про сам режим, без
    вызова: `route_call` не знает панель, какая часть `ModeChoice` пришла из
    вызова, а какая нет, — это решают `profile_mode`/`profile_note`.
    """
    call = last_call.get("route_call")
    mode = last_call.get("profile_mode") or ""
    note = last_call.get("profile_note") or ""
    if call is None:
        if not note:
            return []
        label = f"«{mode}»" if mode else "нет"
        return [f"- **🧭 Режим {label}** — {note}."]
    if not call["ok"]:
        line = f"- **🧭 Роутер профиля не удался — {call['label']}:** {call['error']}"
        if call["text"]:
            line += f" (начало ответа: «{call['text'][:120]}»)"
        line += (
            f" — режим этого хода: «{mode}» ({note}), ход состоялся. "
            f"Потрачено {_fmt_int(call['total_tokens'] or 0)} токенов."
        )
        return [line]
    line = (
        f"- **🧭 Роутер профиля — {call['label']}:** режим «{mode}» — {note}; "
        f"ответ ≈{_fmt_int(estimate_tokens(call['text']))} токенов, вызов "
        f"{_fmt_int(call['total_tokens'])} токенов, "
        f"{_fmt_cost(call['cost_usd'])}, {call['elapsed']:.2f} s"
    )
    if call.get("finish_reason") == "length":
        line += (
            " — ⚠️ **ответ упёрся в потолок `max_tokens` и оборван на "
            "полуслове**, применён как есть"
        )
    return [line]


def _task_tracker_lines(last_call: dict) -> list[str]:
    """Трекер задачи этого хода (день 13, §7.5) — строкой следом за
    роутером профиля, по образцу `_route_call_lines()`.

    Трекер не вызывался, а задача есть, — отдельная строка про саму задачу:
    панель не знает причину («задачи нет», «задача завершена», …), это
    `task_note`/`task_event` из `AgentReply`.
    """
    call = last_call.get("task_call")
    note = last_call.get("task_note") or ""
    if call is None:
        return [f"- **📋 Задача:** {note}"] if note else []
    if not call["ok"]:
        line = f"- **📋 Трекер задачи не удался — {call['label']}:** {call['error']}"
        if call["text"]:
            line += f" (начало ответа: «{call['text'][:120]}»)"
        line += (
            f" — не применён, ход состоялся. Потрачено "
            f"{_fmt_int(call['total_tokens'] or 0)} токенов."
        )
        return [line]
    line = (
        f"- **📋 Трекер задачи — {call['label']}:** {note}; ответ "
        f"≈{_fmt_int(estimate_tokens(call['text']))} токенов, вызов "
        f"{_fmt_int(call['total_tokens'])} токенов, "
        f"{_fmt_cost(call['cost_usd'])}, {call['elapsed']:.2f} s"
    )
    return [line]


_LEVEL_MARKERS = {"ok": "🟢", "warn": "🟡", "danger": "🔴", "over": "🔴"}

# Тексты предупреждений не называют стратегий: панель не знает, какие они
# бывают, и новые стратегии не должны требовать правок здесь.
_LEVEL_WARNINGS = {
    "warn": (
        "⚠️ **Занято больше 70% окна.** Если активная стратегия отправляет "
        "всю историю, дальше будет только быстрее."
    ),
    "danger": (
        "🔴 **Занято больше 90% окна.** Ещё пара ходов — и запрос перестанет "
        "влезать. Стратегия сама не переключается: выбрать её можно слева, "
        "в списке «Стратегия контекста»."
    ),
    "over": (
        "🔴 **Оценка превышает окно модели.** Запрос всё равно будет отправлен "
        "и, скорее всего, отклонён целиком: ответа не будет, ход не "
        "состоится, стек сообщений и файл сессии останутся как есть. "
        "Предохранителя здесь нет намеренно: стратегия контекста — способ до "
        "переполнения не доходить, а не проверка перед вызовом."
    ),
}


def _bar(usage: dict) -> str:
    """Текстовая полоса занятости с маркером уровня. Надёжнее любого виджета
    и хорошо читается на видео."""
    marker = _LEVEL_MARKERS.get(usage["level"], "⚪")
    if usage["ratio"] is None:
        return f"{marker} `░░░░░░░░░░` н/д"
    filled = max(0, min(round(usage["ratio"] * 10), 10))
    return (
        f"{marker} `{'█' * filled}{'░' * (10 - filled)}` "
        f"{usage['ratio'] * 100:.1f}%"
    )


def _heaviest_message(usage: dict, messages: list[dict]) -> str:
    """Самое тяжёлое сообщение стека — по `per_message`, который идёт в том же
    порядке, что **отправленная** часть истории: сообщения, оставшиеся за
    пределами отправленного хвоста (свёрнутые в сводку, разобранные в факты
    или просто отброшенные окном), в этом ряду не участвуют."""
    per_message = usage["request"]["per_message"]
    if not per_message:
        return "- **Самое тяжёлое сообщение стека:** стек пуст"
    index = max(range(len(per_message)), key=lambda i: per_message[i])
    # Индекс в отправленном хвосте, а не в стеке: при сжатии хвост начинается
    # не с нулевого сообщения, и смещение нужно вернуть обратно.
    offset = max(0, len(messages) - len(per_message))
    role = "н/д"
    if offset + index < len(messages) and isinstance(messages[offset + index], dict):
        role = messages[offset + index].get("role", "н/д")
    return (
        f"- **Самое тяжёлое сообщение из отправленных:** #{offset + index} "
        f"({role}), ≈{_fmt_int(per_message[index])} токенов"
    )


def _context_md(
    usage: dict,
    messages: list[dict],
    calibration: float | None,
    calibration_calls: int,
    question: str,
) -> str:
    """Блок «Бюджет контекста»: сколько уйдёт в модель, если отправить сейчас.

    Стоит сразу под метриками последнего вызова — это про тот же запрос,
    только с другой стороны: там факт после вызова, здесь оценка до него.
    """
    request = usage["request"]
    lines = [
        "### Бюджет контекста",
        "",
        f"- **Модель:** `{usage['model']}` · окно "
        + (
            "**н/д** (модели нет в таблице контекстных окон)"
            if usage["limit"] is None
            else f"{_fmt_int(usage['limit'])} токенов"
        )
        + f", резерв под ответ {_fmt_int(usage['answer_reserve'])}",
        # Память стратегии (день 9: сводка; день 10: факты) — слагаемое
        # наравне с остальными: без неё сумма в строке не сходилась бы с
        # итогом ровно на размер этой памяти. Инварианты (день 14), профиль
        # (день 12), слои памяти дня 11 и задача (день 13) — тем же правилом,
        # в порядке блоков запроса: инварианты сразу за системой, дальше
        # профиль, долговременная и рабочая, задача, потом память стратегии.
        f"- **В запросе:** система {_fmt_int(request['system'])} + инварианты "
        f"{_fmt_int(request['invariants'])} + профиль "
        f"{_fmt_int(request['profile'])} + долговременная "
        f"{_fmt_int(request['long_term'])} + рабочая "
        f"{_fmt_int(request['working'])} + задача {_fmt_int(request['task'])} + "
        f"память стратегии {_fmt_int(request['memory'])} + история "
        f"{_fmt_int(request['history'])} ({len(request['per_message'])} сообщ.) + "
        f"вопрос {_fmt_int(request['question'])} + служебные "
        f"{_fmt_int(request['overhead'])} ≈ **{_fmt_int(request['total'])}**",
        # Отдельные корзины, а не история: что это за память/профиль и куда
        # встаёт. Инварианты (день 14) — первыми из блоков агента.
        f"- **Инварианты в запросе:** "
        + (
            f"≈{_fmt_int(request['invariants'])} токенов — блок сразу за "
            f"системным промптом, первым из блоков агента"
            if request["invariants"]
            else "ничего: инвариантов нет, они выключены или «Инварианты в "
            "запросе» выключено"
        ),
        f"- **Профиль в запросе:** "
        + (
            f"≈{_fmt_int(request['profile'])} токенов — блок сразу за "
            f"системным промптом, перед слоями памяти"
            if request["profile"]
            else "ничего: профиль выключен или его блок пуст"
        ),
        f"- **Память стратегии в запросе:** "
        + (
            f"≈{_fmt_int(request['memory'])} токенов — она уходит ведущим "
            f"`system`-сообщением сразу после системного промпта и в историю "
            f"не входит"
            if request["memory"]
            else "ничего: у активной стратегии памяти нет или она пуста"
        ),
    ]
    if question:
        lines.append(
            f"- **В поле ввода:** ≈{_fmt_int(request['question'])} токенов — "
            f"итог и полоса ниже посчитаны вместе с ними"
        )
    if usage["limit"] is None:
        lines.append(
            f"- **Занято:** ≈{_fmt_int(usage['used'])} токенов, от какой доли "
            f"окна — неизвестно: процентов и предупреждений здесь не будет"
        )
    else:
        # Отрицательный остаток — это «не влезли», и написать так честнее,
        # чем показывать свободное место со знаком минус.
        tail = (
            f"свободно ≈{_fmt_int(usage['free'])}"
            if usage["free"] >= 0
            else f"**не хватает ≈{_fmt_int(-usage['free'])}**"
        )
        lines.append(
            f"- **Занято:** ≈{_fmt_int(usage['used'])} из "
            f"{_fmt_int(usage['available'])} доступных · {tail}"
        )
    lines.append(f"- {_bar(usage)}")
    if calibration is None:
        lines.append(
            f"- **Калибровка:** оценка {_fmt_int(usage['estimated'])} показана "
            f"сырой — успешных вызовов с `usage` у этого агента ещё не было"
        )
    else:
        lines.append(
            f"- **Калибровка:** оценка {_fmt_int(usage['estimated'])} → "
            f"{_fmt_int(usage['used'])} (×{calibration:.3f} по "
            f"{calibration_calls} вызовам)"
        )
    lines.append(_heaviest_message(usage, messages))
    warning = _LEVEL_WARNINGS.get(usage["level"])
    if warning:
        lines += ["", warning]
    lines += [
        "",
        "_До запроса это **оценка**, а не факт: точное число знает только "
        "токенизатор модели, и приходит оно в `usage` вместе с ответом. "
        "Без нового вопроса показан бюджет стека — нижняя граница следующего "
        "запроса; пока вы набираете текст, панель не пересчитывается._",
    ]
    return "\n".join(lines)


def _flow_md(view: dict, totals: dict, model: str) -> str:
    """Блок «Контекст: что уходит в модель» — главный блок дня 9.

    Стоит над «Бюджетом контекста»: сначала «что отправляем», потом «сколько
    это от окна». Панель не знает, какие бывают стратегии: имя, параметры
    словами и текст памяти приходят из `StrategyState`, числа — из
    `ContextView`.
    """
    state = view["state"]
    request = view["usage"]["request"]
    estimated = view["usage"]["estimated"]
    full_total = view["full"]["total"]
    saved = view["saved_tokens"]
    memory_name = state["memory_label"].lower()

    lines = [
        "### Контекст: что уходит в модель",
        "",
        f"- **Стратегия:** «{view['strategy']}» — {state['description']}",
        # Профиль (день 12), слои памяти (день 11) и задача (день 13) —
        # слагаемыми между системой и памятью стратегии: без них сумма не
        # сошлась бы с итогом. Ни стратегия, ни профиль, ни задача друг о
        # друге не знают, и в экономию ниже они не входят — база «вся
        # история» собрана с тем же профилем, теми же слоями и тем же
        # блоком задачи.
        f"- **Состав запроса:** система {_fmt_int(request['system'])} + "
        f"инварианты {_fmt_int(request['invariants'])} + "
        f"профиль {_fmt_int(request['profile'])} + долговременная "
        f"{_fmt_int(request['long_term'])} + рабочая "
        f"{_fmt_int(request['working'])} + задача {_fmt_int(request['task'])} + "
        f"{memory_name} {_fmt_int(request['memory'])} + "
        f"{view['sent_messages']} сообщ. истории {_fmt_int(request['history'])} + "
        f"вопрос {_fmt_int(request['question'])} + служебные "
        f"{_fmt_int(request['overhead'])} ≈ **{_fmt_int(estimated)}**",
    ]

    # «За бортом» (день 10, §7.1): панель не знает, какие бывают стратегии, —
    # ушло всё / ушло не всё, и есть ли поверх хвоста память, решают числа
    # `ContextView`, а не имя стратегии.
    dropped = view["history_messages"] - view["sent_messages"]
    memory_tokens = request["memory"]
    if dropped <= 0:
        if memory_tokens:
            lines.append(
                f"- **За бортом:** ничего — все {view['history_messages']} "
                f"сообщ. стека в запросе, поверх них — {memory_name} "
                f"≈{_fmt_int(memory_tokens)} токенов"
            )
        else:
            lines.append(
                f"- **За бортом:** ничего, все {view['history_messages']} "
                f"сообщ. стека в запросе"
            )
    else:
        replacement = (
            f"вместо них — {memory_name} ≈{_fmt_int(memory_tokens)} токенов"
            if memory_tokens
            else "вместо них — ничего: отброшены без замены"
        )
        lines.append(
            f"- **За бортом:** не ушли в запрос {dropped} сообщ. из "
            f"{view['history_messages']}; {replacement}"
        )

    # Экономию есть смысл показывать только там, где что-то действительно
    # изменилось против полного контекста — либо часть истории не ушла в
    # запрос, либо поверх неё встала память.
    if dropped > 0 or memory_tokens:
        if saved >= 0:
            share = f"{estimated / full_total * 100:.0f}%" if full_total else "н/д"
            lines.append(
                f"- **Экономия:** ≈{_fmt_int(estimated)} вместо "
                f"≈{_fmt_int(full_total)} — {share} полного контекста, "
                f"сэкономлено ≈{_fmt_int(saved)} токенов "
                f"(≈{_fmt_cost(estimate_cost_usd(model, saved, 0))} по входу)"
            )
        else:
            # Так бывает: память (сводка, факты) весит больше того, что она
            # заменила. Показываем честно — это и есть цена приёма на
            # коротком диалоге.
            lines.append(
                f"- **Экономия:** её нет — запрос ≈{_fmt_int(estimated)} "
                f"токенов, то есть дороже полного контекста на "
                f"≈{_fmt_int(-saved)}"
            )

    # Итог за время жизни агента: честный ответ на вопрос «а экономит ли».
    lifetime_saved = totals["saved_tokens"]
    verdict = "сэкономлено" if lifetime_saved >= 0 else "переплачено"
    lines.append(
        f"- **За время жизни агента:** {verdict} ≈{_fmt_int(abs(lifetime_saved))} "
        f"токенов "
        f"(≈{_fmt_cost(estimate_cost_usd(model, abs(lifetime_saved), 0))} по "
        f"входу), потрачено на служебные вызовы "
        f"{_fmt_int(totals['service_tokens'])} токенов "
        f"({_fmt_cost(totals['service_cost_usd'])}) за "
        f"{totals['service_calls']} {_service_word(totals['service_calls'])}"
    )
    if view["pending_label"]:
        lines.append(f"- ⏳ **На следующем ходе:** {view['pending_label']}")
    else:
        lines.append(f"- **Дальше:** {state['note']}")
    lines += [
        "",
        "_Стратегия меняет запрос, а не память: стек сообщений и файл сессии "
        "всегда полные. Переключение стратегии диалог не трогает — один и "
        "тот же разговор можно отправить разными способами и сравнить "
        "ответы._",
    ]
    return "\n".join(lines)


def _service_word(count: int) -> str:
    """«1 служебный вызов» / «2 служебных вызова» / «5 служебных вызовов» —
    слово общее для обеих стратегий с памятью (день 10): панель не говорит
    «свёртка» там, где это могут быть факты."""
    if 11 <= count % 100 <= 14:
        return "служебных вызовов"
    match count % 10:
        case 1:
            return "служебный вызов"
        case 2 | 3 | 4:
            return "служебных вызова"
        case _:
            return "служебных вызовов"


def _memory_update(view: dict) -> dict:
    """Блок «Память стратегии»: сама память текстом — то самое «храните
    summary отдельно», которое должно быть видно.

    Подпись и содержимое приходят из `StrategyState`: панель не знает, что
    внутри — сводка, факты или ничего. Памяти нет — в поле стоит объяснение
    почему, а не пустота. Подпись без рода и числа (день 10, §7.1): у
    «Сводки» и «Фактов» они разные, а панель этого не знает.
    """
    state = view["state"]
    text = state["memory_text"]
    if not text:
        return gr.update(
            value=state["note"],
            label=f"{state['memory_label']}: пусто",
        )
    last = f" · последнее: {state['last_update']}" if state["last_update"] else ""
    return gr.update(
        value=text,
        label=(
            f"{state['memory_label']}: учтено сообщений {state['covered']} из "
            f"{view['history_messages']} · обновлений {state['updated_turns']} "
            f"· ≈{estimate_tokens(text)} токенов{last}"
        ),
    )


# --- «Рост по ходам»: журнал ходов агента в таблицу -----------------------
# Данные собираются из `agent.turns` через pandas — он уже стоит в проекте как
# зависимость Gradio, новых строк в requirements.txt день 8 не добавляет.
# Пустой журнал даёт пустой DataFrame с теми же колонками, а не None: таблица
# должна рисоваться и до первого хода.

def _turns_table(turns: list[dict]) -> pd.DataFrame:
    """Полный ряд журнала ходов агента."""
    rows = [
        {
            "ход": turn["turn"],
            "стек до хода": turn["history_before"],
            "оценка": turn["estimated_prompt_tokens"],
            "prompt": turn["prompt_tokens"],
            "completion": turn["completion_tokens"],
            "total": turn["total_tokens"],
            "оценка/факт": _fmt_delta(
                turn["estimated_prompt_tokens"], turn["prompt_tokens"]
            ),
            "стоимость": _fmt_cost(turn["cost_usd"]),
            "накопительно": _fmt_cost(turn["cumulative_cost_usd"]),
            # Колонки дня 9 — в конце: существующие не переставляются.
            # «вся история» и «служебные» рядом отвечают на главный вопрос
            # дня: экономия минус плата за служебный вызов. Название колонки
            # день 10 меняет: «без сжатия» было бы неправдой для окна — оно
            # не сжимает, а отбрасывает (спецификация дня 10, §7.1).
            "стратегия": turn["strategy"],
            "отправлено сообщ.": turn["sent_messages"],
            "вся история": turn["full_prompt_tokens"],
            "служебные": turn["service_tokens"],
            # Колонки дня 10 — снова в конце: у фактов ход идёт двумя
            # вызовами, и в журнале это должно читаться числом (спецификация
            # дня 10, §5.3). Неудачный служебный вызов тоже занял время — оно
            # в колонке есть.
            "время ответа, s": f"{turn['elapsed']:.2f}",
            "время служебного, s": f"{turn['service_elapsed']:.2f}",
            # Колонки дня 11 — в конце (§7.4): корзины слоёв в запросе хода и
            # разбор памяти этого хода. Разбор — отдельной колонкой, а не в
            # «служебных»: те — цена памяти стратегий рядом с их экономией.
            "долговременная": turn["long_term_tokens"],
            "рабочая": turn["working_tokens"],
            "разбор памяти": turn["memory_tokens"],
            "время разбора, s": f"{turn['memory_elapsed']:.2f}",
            # Колонки дня 12 — в конце (§7.5): режим, ушедший в запрос этого
            # хода (и откуда), корзина профиля и цена роутера — своя
            # колонка, а не часть «служебных» или «разбора памяти».
            "режим": (
                f"{turn['profile_mode']} ({turn['profile_note']})"
                if turn["profile_mode"]
                else (turn["profile_note"] or "—")
            ),
            "профиль": turn["profile_tokens"],
            "роутер": turn["route_tokens"],
            "время роутера, s": f"{turn['route_elapsed']:.2f}",
            # Колонки дня 13 — в конце (§5.9, §7.5): этап и шаг после хода
            # (пауза — пометкой, задачи нет — прочерком), событие перехода
            # трекера этого хода, корзина задачи и цена трекера — своя
            # колонка, а не часть «служебных»/«разбора памяти»/«роутера».
            "этап": (
                f"{turn['task_stage']}, шаг {turn['task_step']}"
                f"{' ⏸' if turn['task_paused'] else ''}"
                if turn["task_stage"]
                else "—"
            ),
            "переход": turn["task_event"] or "—",
            "задача": turn["task_tokens"],
            "трекер": turn["tracker_tokens"],
            "время трекера, s": f"{turn['tracker_elapsed']:.2f}",
            # Колонки дня 14 — в конце (§6.9, §8.6): корзина инвариантов
            # (блок уходит первым), номера конфликта этого хода (или
            # прочерк), цена стража — своя колонка, а не часть «служебных»/
            # «разбора памяти»/«роутера»/«трекера» — и его время.
            "инварианты": turn["invariant_tokens"],
            "конфликт": turn["conflict"] or "—",
            "страж": turn["guard_tokens"],
            "время стража, s": f"{turn['guard_elapsed']:.2f}",
        }
        for turn in turns
    ]
    return pd.DataFrame(
        rows,
        columns=["ход", "стек до хода", "оценка", "prompt", "completion",
                 "total", "оценка/факт", "стоимость", "накопительно",
                 "стратегия", "отправлено сообщ.", "вся история", "служебные",
                 "время ответа, s", "время служебного, s",
                 "долговременная", "рабочая", "разбор памяти",
                 "время разбора, s", "режим", "профиль", "роутер",
                 "время роутера, s", "этап", "переход", "задача", "трекер",
                 "время трекера, s", "инварианты", "конфликт", "страж",
                 "время стража, s"],
    )


def _totals_md(agent_title: str, totals: dict, model: str) -> str:
    """Накопленное агентом за время жизни. «Сбросить диалог» эти счётчики
    не обнуляет — сброшен диалог, а не агент."""
    lifetime_saved = totals["saved_tokens"]
    verdict = "сэкономлено" if lifetime_saved >= 0 else "переплачено"
    return (
        f"### Накоплено агентом {agent_title}\n\n"
        f"- **Вызовов:** {totals['calls']} (из них с ошибкой: "
        f"{totals['errors']}, служебных: {totals['service_calls']}, "
        f"разборов памяти: {totals['memory_calls']}, роутера профиля: "
        f"{totals['route_calls']}, трекера задачи: {totals['tracker_calls']}, "
        f"стража инвариантов: {totals['guard_calls']})\n"
        f"- **🔢 Токены:** prompt={totals['prompt_tokens']} / "
        f"completion={totals['completion_tokens']} / "
        f"total={totals['total_tokens']}\n"
        f"- **🧵 Служебные вызовы:** {_fmt_int(totals['service_tokens'])} "
        f"токенов, {_fmt_cost(totals['service_cost_usd'])} — это цена памяти "
        f"стратегий; на запросах при этом {verdict} "
        f"≈{_fmt_int(abs(lifetime_saved))} токенов "
        f"({_fmt_cost(estimate_cost_usd(model, abs(lifetime_saved), 0))} "
        f"по входу)\n"
        # Строка дня 11 (§7.4): разбор памяти ничего не экономит, он
        # добавляет, — и стоит отдельно от служебных вызовов стратегий.
        f"- **🧠 Разбор памяти:** {totals['memory_calls']} "
        f"{_calls_word(totals['memory_calls'])}, "
        f"{_fmt_int(totals['memory_tokens'])} токенов, "
        f"{_fmt_cost(totals['memory_cost_usd'])} — это цена слоёв памяти; в "
        f"экономию стратегии не входит\n"
        # Строка дня 12 (§7.5): роутер профиля — своя цена, не входит ни в
        # служебные вызовы стратегий, ни в разбор памяти.
        f"- **🧭 Роутер профиля:** {totals['route_calls']} "
        f"{_calls_word(totals['route_calls'])}, "
        f"{_fmt_int(totals['route_tokens'])} токенов, "
        f"{_fmt_cost(totals['route_cost_usd'])} — цена автоматического "
        f"выбора режима; в экономию стратегии не входит\n"
        # Строка дня 13 (§5.9, §7.5): трекер задачи — цена ведения задачи по
        # разговору, не входит ни в служебные вызовы, ни в разбор памяти,
        # ни в роутер.
        f"- **📋 Трекер задачи:** {totals['tracker_calls']} "
        f"{_calls_word(totals['tracker_calls'])}, "
        f"{_fmt_int(totals['tracker_tokens'])} токенов, "
        f"{_fmt_cost(totals['tracker_cost_usd'])} — цена ведения задачи по "
        f"разговору; в экономию стратегии не входит\n"
        # Строка дня 14 (§6.9, §8.6): страж инвариантов — цена сверки каждого
        # хода с ограничениями, своя строка и свои счётчики.
        f"- **🛡 Страж инвариантов:** {totals['guard_calls']} "
        f"{_calls_word(totals['guard_calls'])}, "
        f"{_fmt_int(totals['guard_tokens'])} токенов, "
        f"{_fmt_cost(totals['guard_cost_usd'])} — цена сверки каждого хода с "
        f"ограничениями; в экономию стратегии не входит\n"
        f"- **💲 Стоимость:** {_fmt_cost(totals['cost_usd'])}"
    )


def _calls_word(count: int) -> str:
    """«1 вызов» / «2 вызова» / «5 вызовов»."""
    if 11 <= count % 100 <= 14:
        return "вызовов"
    match count % 10:
        case 1:
            return "вызов"
        case 2 | 3 | 4:
            return "вызова"
        case _:
            return "вызовов"


def _process_md(process: dict) -> str:
    """Счётчики процесса: один инстанс приложения — много агентов."""
    return (
        "### Счётчики процесса\n\n"
        f"- **🤖 Агентов создано:** {process['agents_created']} "
        f"(живых в реестре: {process['agents_alive']})\n"
        f"- **Вызовов:** {process['calls']} (из них с ошибкой: {process['errors']})\n"
        f"- **🔢 Токены (total):** {process['total_tokens']}\n"
        f"- **💲 Стоимость:** {_fmt_cost(process['cost_usd'])}"
    )


def _storage_md(state: dict, session_file: dict | None) -> str:
    """Блок «Хранилище»: то же самое, что в стеке сообщений, но с точки зрения
    диска — чтобы «в памяти» и «на диске» были в кадре рядом."""
    session_id = state["session_id"]
    path = STORE.path_for(session_id)
    lines = [
        "### Хранилище",
        "",
        f"- **Класс:** `{state['store']}`",
        f"- **Каталог данных:** `{display_path(STORE.data_dir)}`",
        f"- **Сессия:** `{session_id}` → `{display_path(path)}`",
        f"- **Восстановлено при создании агента:** "
        f"{state['restored_messages']} сообщ.",
    ]
    if session_file is not None:
        messages = session_file.get("messages") or []
        lines.append(
            f"- **В файле сейчас:** {len(messages)} сообщ., "
            f"обновлён {session_file.get('updated_at') or 'н/д'}"
        )
    elif path.exists():
        lines.append("- **Файл:** есть на диске, но не читается — см. ошибку ниже.")
    else:
        lines.append(
            "- **Файла ещё нет:** он появится после первого успешного ответа "
            "и исчезнет по «Сбросить диалог» — пустых файлов сессий не бывает."
        )
    if state["store_error"]:
        lines.append(f"- ⚠️ **Хранилище:** {state['store_error']}")
    return "\n".join(lines)


# --- Ветки диалога (день 10) ----------------------------------------------
# Ветка — отдельная сессия и отдельный агент, а не пункт списка стратегий
# (спецификация дня 10, §2.2): у неё свой переключатель, устроенный так же,
# как переключатель агентов, и свой блок в панели.

def _origin_label(member: Agent) -> str:
    """Одна строка происхождения сессии — используется и в переключателе
    «Ветка диалога», и в блоке «Ветки диалога»."""
    if member.branch is None:
        return f"{member.session_id} · исходный диалог · {len(member.history)} сообщ."
    return (
        f"{member.session_id} · ветка от {member.branch.parent} · "
        f"{member.branch.checkpoint} · {len(member.history)} сообщ."
    )


def _branch_choices(agent: Agent) -> list[tuple[str, int]]:
    """Варианты переключателя «Ветка диалога» — исходный диалог и все его
    ветки (`branch_family()`), в порядке создания. Значение — номер агента,
    как у переключателя агентов (день 6)."""
    family = branch_family(agent)
    entries = [(_origin_label(member), member.number) for member in family]
    if agent not in family:
        # Активный агент, удалённый из реестра в другой вкладке, — правило
        # переключателя агентов (день 6): отдельный пункт с пометкой.
        entries.insert(0, (f"{agent.session_id} · удалён", agent.number))
    return entries


def _checkpoint_choices(agent: Agent) -> list[str]:
    """Варианты переключателя «Checkpoint» — checkpoint'ы активного агента,
    по порядку сохранения; значение по умолчанию (последний) задаёт вызывающий
    код. Список служебный — значений-кортежей с отдельной подписью не нужно,
    id читается напрямую и используется тем же текстом, что и подпись."""
    return [
        f"{cp.id} · {cp.messages} сообщ. · «{cp.strategy}» · "
        f"{_time_of(cp.created_at)}"
        for cp in agent.checkpoints
    ]


def _checkpoint_id_from_choice(choice: str | None) -> str | None:
    """Список «Checkpoint» показывает составную подпись, а не голый id —
    id из неё же и извлекается: разводить два разных значения (то, что видно,
    и то, что выбрано) незачем при таком маленьком списке."""
    if not choice:
        return None
    return choice.split(" · ", 1)[0]


def _live_branches(agent: Agent) -> list[Agent]:
    """Ветки, созданные от checkpoint'ов этого диалога, связь с которыми жива.

    Связь проверяет только `branch_parent()` — по id **и** времени
    checkpoint'а (спецификация дня 10, §5.2): одного `session_id` мало, после
    сброса диалога его старые ветки уже не «от этого диалога», а одного id
    checkpoint'а мало тем более — новый `cp1` сброшенного родителя получил бы
    ветки прежнего. Правило живёт в `agent.py`, панель его не повторяет."""
    return [
        member
        for member in branch_family(agent)
        if member.branch is not None and branch_parent(member) is agent
    ]


def _branches_md(agent: Agent) -> str:
    """Блок «Ветки диалога» (спецификация дня 10, §7.3): что это за сессия,
    какие у неё checkpoint'ы и какие ветки от них уже созданы, и всё
    семейство целиком. Ни своей копии переписки, ни своего списка веток
    интерфейс не ведёт — всё здесь читается из агента и реестра заново."""
    family = branch_family(agent)
    live = _live_branches(agent)
    lines = ["### Ветки диалога", ""]

    if agent.branch is None:
        lines.append("- **Эта сессия:** исходный диалог.")
    else:
        own = len(agent.history) - agent.branch.messages
        parent = branch_parent(agent)
        alive = (
            f"родитель {_agent_title(parent)} в процессе, checkpoint на месте"
            if parent is not None
            else "родителя в процессе нет или его checkpoint'а больше нет — "
                 "ветка живёт сама по себе"
        )
        lines.append(
            f"- **Эта сессия:** ветка от {agent.branch.parent} · "
            f"{agent.branch.checkpoint} ({_time_of(agent.branch.created_at)}): "
            f"первые {agent.branch.messages} сообщ. скопированы при создании, "
            f"своих — {own}. {alive[:1].upper()}{alive[1:]}."
        )

    if agent.checkpoints:
        lines.append("- **Checkpoint'ы этой сессии:**")
        for cp in agent.checkpoints:
            children = [
                member.session_id
                for member in live
                if member.branch.checkpoint == cp.id
            ]
            branches_note = (
                f"ветки: {', '.join(children)}" if children else "веток ещё нет"
            )
            lines.append(
                f"  - `{cp.id}` · {cp.messages} сообщ. · «{cp.strategy}» · "
                f"{_time_of(cp.created_at)} · {branches_note}"
            )
    else:
        lines.append(
            "- **Checkpoint'ы этой сессии:** нет — «Сохранить checkpoint» "
            "зафиксирует конец текущей истории."
        )

    lines.append("- **Семейство:**")
    for member in family:
        marker = " ← активна" if member.number == agent.number else ""
        lines.append(f"  - {_origin_label(member)} · «{member.strategy.name}»{marker}")

    lines += [
        "",
        "_Ветка — отдельная сессия с копией общего начала истории, со своим "
        "файлом, своей стратегией и своими счётчиками; стратегия решает, что "
        "из истории отправить, ветка — какая это история._",
    ]
    return "\n".join(lines)


# --- Слои памяти (день 11) ------------------------------------------------
# Блок «Слои памяти», ряд кандидатов и статусы про долговременную память.
# Панель не знает, какие ключи бывают: рабочая память и кандидаты приходят из
# `MemoryState` в `debug_state()["memory"]`, долговременная — из
# `debug_state()["long_term"]`, её текст — из `memory.long_term_text()`.

def _records_word(count: int) -> str:
    """«1 запись» / «2 записи» / «5 записей»."""
    if 11 <= count % 100 <= 14:
        return "записей"
    match count % 10:
        case 1:
            return "запись"
        case 2 | 3 | 4:
            return "записи"
        case _:
            return "записей"


def _long_term_records() -> str:
    """«5 записей» — сколько записей в долговременной памяти процесса."""
    count = len(LONG_TERM.entries())
    return f"{count} {_records_word(count)}"


def _long_term_status() -> str:
    """Фраза про долговременную память для статуса при открытии страницы
    (§7.1): сколько записей и откуда, или что её пока нет."""
    if LONG_TERM.error:
        return f"Долговременная память: ⚠️ {LONG_TERM.error}."
    if LONG_TERM.read_file() is None:
        return (
            f"Долговременной памяти пока нет — `{LONG_TERM.path}` появится с "
            f"первым сохранённым кандидатом."
        )
    return f"Долговременная память: {_long_term_records()} из `{LONG_TERM.path}`."


def _profile_status() -> str:
    """Фраза про профиль для статуса при открытии страницы (§7.1): сохранён
    ли он, или действует профиль по умолчанию."""
    if PROFILE.error:
        return f"Профиль: ⚠️ {PROFILE.error}."
    if not PROFILE.saved:
        return (
            f"Профиль не сохранён — действует профиль по умолчанию из "
            f"`presets.py`, `{PROFILE.path}` появится с первым сохранением."
        )
    return f"Профиль сохранён: `{PROFILE.path}`."


def _candidate_label(candidate: dict) -> str:
    """Кандидат словами — и пунктом группы «Кандидаты в долговременную
    память», и строкой блока «Слои памяти»: «коллекция: только база (было:
    база и Undertow)»."""
    value = candidate["value"] if candidate["value"] is not None else "удалить"
    previous = (
        f"было: {candidate['previous']}"
        if candidate["previous"] is not None
        else "раньше не было"
    )
    return f"{candidate['key']}: {value} ({previous})"


def _mode_block_tokens(profile, mode_name: str | None) -> int:
    """Размер блока профиля «общая часть + этот режим» в токенах — по
    `estimate_tokens()` над `user_profile.profile_block()` (§7.4)."""
    block = profile_block(profile, mode_name)
    return estimate_tokens(block["content"]) if block else 0


def _profile_md(state: dict) -> str:
    """Блок «Профиль» (день 12, §7.4) — сразу над «Слоями памяти», в порядке
    блоков запроса: профиль встаёт в запрос раньше слоёв памяти. Панель не
    знает, какие режимы бывают, — всё приходит из
    `debug_state()["profile"]` и `user_profile.profile_text()`."""
    profile_state = state["profile"]
    lines = ["### Профиль", ""]
    if not PROFILE.saved:
        lines.append(
            f"- **Где лежит:** файла нет — действует профиль по умолчанию "
            f"из `presets.py`, `{PROFILE.path}` появится с первым сохранением."
        )
    else:
        lines.append(
            f"- **Где лежит:** `{PROFILE.path}` — один на процесс, общий "
            f"для всех агентов и сессий."
        )
    if PROFILE.error:
        lines.append(f"- ⚠️ **Хранилище:** {PROFILE.error}")

    if profile_state is None:
        lines.append("- У этого агента нет профиля — он ведёт себя как на дне 11.")
        return "\n".join(lines)

    lines.append(f"- **В запросе у этого агента:** {profile_state['choice']}")
    last = profile_state["last"]
    if last is None:
        lines.append("- **Режим прошлого хода:** ходов ещё не было")
    else:
        last_bit = f"«{last['mode']}» — {last['note']}" if last["mode"] else last["note"]
        lines.append(f"- **Режим прошлого хода:** {last_bit}")

    profile_obj, _ = parse_profile(profile_state["data"])
    preview = profile_state["preview"]
    if preview["sent"] and preview["mode"]:
        next_tokens = _mode_block_tokens(profile_obj, preview["mode"])
        lines.append(
            f"- **Следующий запрос:** «{preview['mode']}» — {preview['note']}; "
            f"блок ≈{_fmt_int(next_tokens)} токенов"
        )
    else:
        lines.append(f"- **Следующий запрос:** {preview['note']} — блок в запрос не уйдёт")

    lines += ["", profile_text(profile_obj, active=preview["mode"] if preview["sent"] else None)]
    if profile_obj.modes:
        lines += ["", "Размер блока «общая часть + режим»:"]
        lines += [
            f"- «{mode.name}»: ≈{_fmt_int(_mode_block_tokens(profile_obj, mode.name))} токенов"
            for mode in profile_obj.modes
        ]
    lines += [
        "",
        "_Профиль пишет только пользователь, в редакторе слева — модель его "
        "не меняет; память — то, что агент узнал из разговоров, профиль — "
        "то, что вы задали сами._",
    ]
    return "\n".join(lines)


def _layer_mark(layer: str, layers: list[str]) -> str:
    return "✅ в запросе" if layer in layers else "⛔ в запрос не уходит"


def _layers_md(state: dict, view: dict) -> str:
    """Блок «Слои памяти» (спецификация дня 11, §7.3): что агент знает — по
    слою, в порядке таблицы §2.1, — где это лежит и уходит ли в запрос.
    Стоит над «Контекстом»: сначала что агент знает, потом что из этого
    отправляется."""
    memory_state = state["memory"]
    layers = state["request_layers"]
    long_term = state["long_term"]
    session_id = state["session_id"]
    lines = ["### Слои памяти", ""]
    if memory_state is None:
        lines.append(
            "У этого агента нет модели памяти: слоёв нет, он ведёт себя как на "
            "дне 10."
        )
        return "\n".join(lines)

    strategy_state = view["state"]
    lines += [
        "#### 💬 Краткосрочная — разговор этой сессии",
        f"- **Сессия** `{session_id}`: в стеке {view['history_messages']} "
        f"сообщ., в запрос уходит {view['sent_messages']} — стратегия "
        f"«{view['strategy']}»",
        "- **Память стратегии:** "
        + (
            f"{strategy_state['memory_label'].lower()} "
            f"≈{_fmt_int(estimate_tokens(strategy_state['memory_text']))} "
            f"токенов — подробности в блоке «Контекст» ниже"
            if strategy_state["memory_text"]
            else "нет — подробности в блоке «Контекст» ниже"
        ),
        f"- **Где лежит:** `{display_path(STORE.path_for(session_id))}` → "
        f"`messages` (память стратегий — `context.memory`); пишет агент каждым "
        f"ходом, «Сбросить диалог» очищает",
        "",
        f"#### 🗂 Рабочая — данные текущей задачи · "
        f"{_layer_mark(LAYER_WORKING, layers)}",
    ]
    if memory_state["working_text"]:
        lines += [f"- {line}" for line in memory_state["working_text"].splitlines()]
    else:
        lines.append(
            "- _пусто_ — разбор памяти заполнит её перед ответом, когда игрок "
            "скажет что-то о текущей партии"
        )
    lines += [
        f"- **Учтено разбором:** {memory_state['covered']} сообщ. из "
        f"{view['history_messages']} · разборов применилось: "
        f"{memory_state['updated_turns']}",
        f"- **Последнее изменение:** "
        f"{memory_state['last_update'] or 'в этом процессе разборов ещё не было'}",
        f"- **Где лежит:** файл сессии `{session_id}`, `context.working`; пишет "
        f"разбор памяти сразу, без подтверждения; сброс очищает, ветка получает "
        f"копию из checkpoint'а",
        "",
        f"#### 🧠 Долговременная — пользователь вообще · "
        f"{_layer_mark(LAYER_LONG_TERM, layers)}",
    ]
    if long_term is None:
        lines.append("- **не подключена:** сохранять кандидатов некуда")
    else:
        text = long_term_text(long_term["entries"], MEMORY_MAP)
        if text:
            # Пустые строки вокруг обязательны: внутри текста разделы идут
            # абзацами со своими списками.
            lines += ["", text, ""]
        else:
            lines.append(
                "- _пусто_ — сюда попадает только то, что человек сохранил "
                "кнопкой «Сохранить в долговременную»"
            )
        lines.append(
            f"- **Где лежит:** `{long_term['path']}` — одна на процесс, общая "
            f"для всех сессий и агентов; сброс, удаление агента, новая сессия "
            f"и ветка её не трогают"
        )
        if LONG_TERM.error:
            lines.append(f"- ⚠️ **Хранилище:** {LONG_TERM.error}")
    lines += ["", "#### ⏳ Ждут решения человека"]
    if memory_state["candidates"]:
        lines += [f"- {_candidate_label(c)}" for c in memory_state["candidates"]]
    else:
        lines.append("- кандидатов нет")
    if memory_state["rejected"]:
        lines.append(
            f"- **Вне карты на последнем разборе:** "
            f"{', '.join(memory_state['rejected'])} — отклонены картой памяти, "
            f"в память не попали"
        )
    lines += [
        "",
        "_Краткосрочная — разговор, рабочая — партия, долговременная — игрок. "
        "Слой в запрос выбирает переключатель «Слои памяти в запросе», запись "
        "в долговременную — человек, куда идёт ключ — карта памяти._",
    ]
    return "\n".join(lines)


def _candidates_update(state: dict) -> dict:
    """Группа «Кандидаты в долговременную память»: пункты — кандидаты
    активного агента, значение — ключ. По умолчанию не отмечено ничего:
    выбор делает человек (§2.3)."""
    candidates = (state["memory"] or {}).get("candidates") or []
    if not candidates:
        return gr.update(
            choices=[],
            value=[],
            label="Кандидаты в долговременную память",
            info=(
                "Ждущих решения записей нет: разбор памяти предложит кандидата, "
                "когда игрок скажет что-то о себе, а не о партии."
            ),
        )
    return gr.update(
        choices=[(_candidate_label(c), c["key"]) for c in candidates],
        value=[],
        label=f"Кандидаты в долговременную память — ждут решения: {len(candidates)}",
        info=(
            "Отметьте, что сохранить или отклонить. В долговременную память "
            "попадает только то, что вы сохранили кнопкой."
        ),
    )


# --- Состояние задачи (день 13) --------------------------------------------
# Блок «Состояние задачи», переключатель «Состояние задачи в запросе» и
# таблица автомата в аккордеоне. Панель не знает, какие бывают этапы и
# события, — рисует то, что вернул `TaskView` из `debug_state()["task"]`.

def _task_plan_lines(view: dict) -> list[str]:
    """Строки плана с отметками ✓/→ — тем же правилом, что блок задачи в
    запросе (`task_state._plan_block()`): на выполнении отмечены шаги, на
    проверке — все шаги и текущая проверка, на «готово»/«отменена» — все
    пункты."""
    steps, checks, results = view["steps"], view["checks"], view["results"]
    stage, step = view["stage"], view["step"]
    if not steps:
        return [
            "**План:** ещё не утверждён — его предлагает ассистент, "
            "утверждает пользователь."
        ]
    lines = ["**План:**"]
    for i, text in enumerate(steps):
        idx = i + 1
        if stage in (STAGE_DONE, STAGE_CANCELLED, STAGE_VALIDATION):
            mark = "✓"
        elif stage == STAGE_EXECUTION:
            mark = "✓" if idx < step else "→" if idx == step else " "
        else:
            mark = " "
        result = results[i] if i < len(results) else ""
        suffix = f" — итог: {result}" if result else ""
        lines.append(f"{mark} {idx}. {text}{suffix}")
    lines.append("**Проверки:**")
    for i, text in enumerate(checks):
        idx = i + 1
        if stage in (STAGE_DONE, STAGE_CANCELLED):
            mark = "✓"
        elif stage == STAGE_VALIDATION:
            mark = "✓" if idx < step else "→" if idx == step else " "
        else:
            mark = " "
        lines.append(f"{mark} {idx}. {text}")
    return lines


def _task_md(state: dict, view: dict) -> str:
    """Блок «Состояние задачи» (день 13, §7.4) — сразу под «Слоями памяти»,
    в порядке блоков запроса."""
    task = state["task"]
    lines = ["### Состояние задачи", ""]
    if task is None:
        lines.append("У этого агента нет автомата задачи: он ведёт себя как на дне 12.")
        return "\n".join(lines)

    task_view = task["view"]
    if task_view["stage"] == NO_TASK:
        lines += [
            "Задачи нет. Начать — кнопкой «Начать задачу с этим сообщением»: "
            "текст поля ввода станет целью, этап — планирование.",
            "",
            "**Допустимо сейчас:** человек — начать.",
        ]
        return "\n".join(lines)

    goal = task_view["goal"]
    lines.append(f"- **Задача:** {goal[:200]}")
    if task_view["stage"] == STAGE_CANCELLED:
        cancelled_from = (
            task_view["transitions"][-1]["from_stage"]
            if task_view["transitions"] else "?"
        )
        lines.append(f"- **Этап:** отменена (была на этапе «{cancelled_from}»)")
    else:
        stage_line = " → ".join(
            f"**{stage}**" if stage == task_view["stage"] else stage
            for stage in MAIN_PATH
        )
        if task_view["paused"]:
            when = task_view["paused_at"]
            stage_line += f" ⏸ на паузе{f' с {when}' if when else ''}"
        lines.append(f"- **Этап:** {stage_line}")
    if task_view["stage"] not in (STAGE_DONE, STAGE_CANCELLED):
        lines.append(f"- **Текущий шаг:** {task_view['step_text']}")
        lines.append(f"- **Ожидается:** {task_view['expected']}")
    lines += ["", *_task_plan_lines(task_view), ""]
    lines.append(
        f"- **Допустимо сейчас:** человек — "
        f"{', '.join(task_view['allowed_human']) or 'ничего'}; трекер — "
        f"{', '.join(task_view['allowed_tracker']) or 'ничего'}"
    )
    if task_view["transitions"]:
        last = task_view["transitions"][-1]
        note = f" — {last['note']}" if last["note"] else ""
        lines.append(
            f"- **Последний переход:** {last['event']} ({last['source']}) — "
            f"{last['from_stage']} → {last['to_stage']}, {_time_of(last['at'])}{note}"
        )
    rejection = task_view["rejection"]
    if rejection:
        lines.append(
            f"- **Последний отказ:** «{rejection['event']}» ({rejection['source']}) "
            f"— {rejection['reason']}, {_time_of(rejection['at'])}"
        )
    if not task["in_request"]:
        lines.append(
            "- **В запросе:** не уходит — «Состояние задачи в запросе» выключено"
        )
    elif not task_view["block_due"]:
        lines.append("- **В запросе:** не уходит — задача завершена")
    else:
        task_tokens = view["usage"]["request"]["task"]
        lines.append(
            f"- **В запросе:** блок ≈{_fmt_int(task_tokens)} токенов уйдёт в "
            f"следующий запрос"
        )
    lines.append(
        f"- **Где лежит:** `{display_path(STORE.path_for(state['session_id']))}` "
        f"→ `context.task`"
    )
    lines.append(
        "- **Трекер на следующем ходе:** будет вызван"
        if task_view["tracker_due"]
        else f"- **Трекер на следующем ходе:** {task_view['tracker_note']}"
    )
    if task_view["transitions"]:
        lines += [
            "",
            "**Журнал переходов:**",
            "",
            "| время | событие | источник | из → в | шаг | заметка |",
            "| --- | --- | --- | --- | --- | --- |",
        ]
        lines += [
            f"| {_time_of(r['at'])} | {r['event']} | {r['source']} | "
            f"{r['from_stage']} → {r['to_stage']} | {r['step']} | "
            f"{r['note'] or '—'} |"
            for r in task_view["transitions"]
        ]
    lines += [
        "",
        "_Этап и шаг меняет только автомат по таблице переходов; событие "
        "называет трекер по разговору или вы кнопками; модель, которая "
        "отвечает, переходов не делает._",
    ]
    return "\n".join(lines)


def _task_in_request_update(state: dict) -> dict:
    """Переключатель «Состояние задачи в запросе»: значение и активность
    подтягиваются к агенту; нет автомата — список неактивен."""
    task = state["task"]
    return gr.update(
        value=task["in_request"] if task is not None else False,
        interactive=task is not None,
    )


def _task_transitions_table_md() -> str:
    """Таблица автомата для аккордеона (§7.4) — собирается один раз при
    построении интерфейса из `task_state.TRANSITIONS`, в `_view()` не входит:
    таблица статична."""
    lines = [
        "| Событие | Кто | Из этапа | Пауза | Куда | Условие | Что значит |",
        "| --- | --- | --- | --- | --- | --- | --- |",
    ]
    for row in TRANSITIONS:
        pause = "—" if row.paused is None else ("на паузе" if row.paused else "не на паузе")
        lines.append(
            f"| {row.event} | {', '.join(row.sources)} | "
            f"{', '.join(row.stages)} | {pause} | {row.target} | "
            f"{row.condition or '—'} | {row.meaning} |"
        )
    return "\n".join(lines)


# --- Инварианты (день 14) ---------------------------------------------------
# Блок «Инварианты», переключатель «Инварианты в запросе», редактор и файл
# инвариантов. Панель не знает, какие бывают события и режимы, — рисует то,
# что вернул `InvariantView` из `debug_state()["invariants"]`.

def _invariant_line(item: dict) -> str:
    """Один инвариант книги строкой: номер, область, пометка «выключен»,
    пометка перехода и формулировка (§8.4)."""
    bits = [f"**{item['id']}**", SCOPE_LABELS.get(item["scope"], item["scope"])]
    if not item["active"]:
        bits.append("⏸ выключен")
    if item["event"]:
        bits.append(f"переход «{item['event']}»: {item['mode']}")
    bits.append(item["text"])
    return " · ".join(bits)


def _invariants_status(agent: "Agent | None" = None) -> str:
    """Фраза про инварианты для статуса при открытии страницы (§8.1): у
    активного агента — сколько действует и сколько из них постоянных; без
    агента (страница открыта на пустом реестре) — по хранилищу."""
    if INVARIANTS.error:
        return f"Инварианты: ⚠️ {INVARIANTS.error}."
    view = agent.invariant_view if agent is not None else None
    if view is not None:
        return (
            f"Инварианты: {view.active} действуют ({view.always} постоянных, "
            f"{view.session} этой сессии) — `{INVARIANTS.path}`."
        )
    return f"Инварианты: {len(_start_invariants)} в `{INVARIANTS.path}`."


def _invariants_md(state: dict, view: dict) -> str:
    """Блок «Инварианты» (день 14, §8.4) — сразу под «Последним вызовом» и
    над блоками профиля и памяти, в порядке блоков запроса: инварианты встают
    в запрос раньше профиля."""
    inv = state["invariants"]
    lines = ["### Инварианты", ""]
    if inv is None:
        lines.append("У этого агента нет инвариантов: он ведёт себя как на дне 13.")
        return "\n".join(lines)
    iv = inv["view"]
    if not iv["items"]:
        lines.append(
            "Инвариантов нет. Добавить — в редакторе под чатом; страж не "
            "вызывается, блок в запрос не уходит."
        )
        return "\n".join(lines)

    lines += [f"- {_invariant_line(item)}" for item in iv["items"]]
    lines += [
        "",
        f"- **Действуют:** {iv['active']} из {len(iv['items'])} — "
        f"{iv['always']} постоянных, {iv['session']} сессионных; над "
        f"переходами автомата — {len(iv['guards'])}",
    ]
    conflict = iv["conflict"]
    if conflict is None:
        lines.append("- **Последний конфликт:** в этом процессе конфликтов ещё не было")
    else:
        why = f" — «{conflict['why']}»" if conflict["why"] else ""
        lines.append(
            f"- **Последний конфликт:** {', '.join(conflict['ids'])}{why}, "
            f"{_time_of(conflict['at'])}"
        )
        lines += [
            f"  - {ident}: {text}"
            for ident, text in zip(conflict["ids"], conflict["texts"])
        ]
        if conflict["unknown"]:
            lines.append(
                f"  - **Отклонены — номеров нет в списке стража:** "
                f"{', '.join(conflict['unknown'])}"
            )
    if iv["last_update"]:
        lines.append(f"- **Последний вызов стража:** {iv['last_update']}")

    if not inv["in_request"]:
        lines.append(
            "- **В запросе:** не уходит — «Инварианты в запросе» выключено; "
            "страж и ограничения переходов при этом работают"
        )
    elif not iv["block_due"]:
        lines.append("- **В запросе:** не уходит — действующих инвариантов нет")
    else:
        tokens_now = view["usage"]["request"]["invariants"]
        lines.append(
            f"- **В запросе:** блок ≈{_fmt_int(tokens_now)} токенов уйдёт в "
            f"следующий запрос — первым из блоков, сразу за системным промптом"
        )
    if not INVARIANTS.saved:
        where = (
            f"файла нет — действуют инварианты по умолчанию из `presets.py`, "
            f"`{INVARIANTS.path}` появится с первым сохранением"
        )
    else:
        where = f"`{INVARIANTS.path}` — один на процесс, общий для всех агентов"
    lines.append(
        f"- **Где лежат:** постоянные — {where}; сессионные — "
        f"`{display_path(STORE.path_for(state['session_id']))}` → "
        f"`context.invariants`"
    )
    if INVARIANTS.error:
        lines.append(f"- ⚠️ **Хранилище:** {INVARIANTS.error}")
    lines.append(
        "- **Страж на следующем ходе:** будет вызван"
        if iv["guard_due"]
        else f"- **Страж на следующем ходе:** не будет вызван — {iv['guard_note']}"
    )
    pending = state["task"]["view"]["pending_note"] if state["task"] else ""
    if pending:
        lines.append(f"- **Ждёт вашего решения:** {pending} — кнопка «✅ Подтвердить переход»")
    lines += [
        "",
        "_Инварианты пишете вы — модель их не пишет и не предлагает. Над "
        "переходами автомата их проверяет код; в остальном инвариант держит "
        "модель, и видно это по её ответам, а не по гарантии._",
    ]
    return "\n".join(lines)


def _invariants_in_request_update(state: dict) -> dict:
    """Переключатель «Инварианты в запросе»: значение и активность
    подтягиваются к агенту; нет книги — переключатель неактивен."""
    inv = state["invariants"]
    return gr.update(
        value=inv["in_request"] if inv is not None else False,
        interactive=inv is not None,
    )


# --- «Агенты процесса»: расход по журналам всех агентов реестра -----------

def _turn_strategies_label(agent: Agent) -> str:
    """Стратегии, на которых прошли ходы этого агента, в порядке появления.
    Ходов не было — активная стратегия с пометкой, а не пустая строка: агент
    без единого хода тоже стоит в таблице."""
    turns = agent.turns
    if not turns:
        return f"{agent.strategy.name} (ходов не было)"
    seen: list[str] = []
    for turn in turns:
        if not seen or seen[-1] != turn.strategy:
            seen.append(turn.strategy)
    return " → ".join(seen)


def _agents_table() -> pd.DataFrame:
    """«Агенты процесса»: по строке на агента реестра, расход по его журналу
    ходов (спецификация дня 10, §7.5) — первое место, где агенты процесса
    сравниваются в одной таблице. Считается только по `agent.turns`, без
    `debug_state()` и без сборки запросов: таблица перерисовывается на каждом
    событии интерфейса. Никакой подсветки и сортировки по «эффективности» —
    порядок строк совпадает с порядком реестра."""
    columns = [
        "агент", "сессия", "ветка от", "стратегии ходов", "ходов",
        "prompt", "completion", "служебные токены", "стоимость",
        "среднее время хода, s",
        # Колонки дня 11 — в конце (§7.4): текущее положение переключателя
        # слоёв и токены разбора памяти. Стоимость и время хода включают
        # разбор.
        "слои в запросе", "токены разбора памяти",
        # Колонки дня 12 — в конце (§7.5): положение переключателя «Профиль
        # в запросе» и токены роутера. Стоимость и среднее время хода
        # включают и роутер.
        "профиль в запросе", "токены роутера",
        # Колонки дня 13 — в конце (§5.9, §7.5): этап и шаг задачи (пауза —
        # пометкой, задачи нет — прочерком) и токены трекера. Стоимость и
        # среднее время хода включают и трекер.
        "задача", "токены трекера",
        # Колонки дня 14 — в конце (§6.9, §8.6): сколько инвариантов
        # действует у агента и токены стража. Стоимость и среднее время хода
        # включают и стража.
        "инварианты", "токены стража",
    ]
    rows = []
    for a in agents():
        turns = a.turns
        if not turns or all(t.cost_usd is None for t in turns):
            cost_str = "н/д"
        else:
            total_cost = sum(
                (t.cost_usd or 0.0) + (t.service_cost_usd or 0.0)
                + (t.memory_cost_usd or 0.0) + (t.route_cost_usd or 0.0)
                + (t.tracker_cost_usd or 0.0) + (t.guard_cost_usd or 0.0)
                for t in turns
            )
            cost_str = _fmt_cost(total_cost)
        avg_time = (
            f"{sum(t.elapsed + t.service_elapsed + t.memory_elapsed + t.route_elapsed + t.tracker_elapsed + t.guard_elapsed for t in turns) / len(turns):.2f}"
            if turns else "н/д"
        )
        task_view = a.task_view
        if task_view is None or task_view.stage == NO_TASK:
            task_str = "—"
        else:
            task_str = f"{task_view.stage}, шаг {task_view.step}"
            if task_view.paused:
                task_str += " ⏸"
        invariant_view = a.invariant_view
        rows.append({
            "агент": _agent_title(a),
            "сессия": a.session_id,
            "ветка от": (
                f"{a.branch.parent} · {a.branch.checkpoint}" if a.branch else "—"
            ),
            "стратегии ходов": _turn_strategies_label(a),
            "ходов": len(turns),
            "prompt": sum(t.prompt_tokens for t in turns),
            "completion": sum(t.completion_tokens for t in turns),
            "служебные токены": sum(t.service_tokens for t in turns),
            "стоимость": cost_str,
            "среднее время хода, s": avg_time,
            "слои в запросе": ", ".join(a.request_layers) or "нет",
            "токены разбора памяти": sum(t.memory_tokens for t in turns),
            "профиль в запросе": a.profile_choice or "—",
            "токены роутера": sum(t.route_tokens for t in turns),
            "задача": task_str,
            "токены трекера": sum(t.tracker_tokens for t in turns),
            "инварианты": invariant_view.active if invariant_view else "—",
            "токены стража": sum(t.guard_tokens for t in turns),
        })
    return pd.DataFrame(rows, columns=columns)


def _profile_choice_options(choices: list[str]) -> list[tuple[str, str]]:
    """Подписи переключателя «Профиль в запросе» (§7.2): `авто` и
    `выключен` — с пояснением прямо в подписи, названия режимов — в
    кавычках. Значение списка — то, что понимает `set_profile_choice()`."""
    labels = {
        CHOICE_AUTO: "авто — режим выбирает роутер",
        CHOICE_OFF: "выключен — профиль не уходит в модель",
    }
    return [(labels.get(choice, f"«{choice}»"), choice) for choice in choices]


def _view(agent: Agent, status: str, question: str = "") -> tuple:
    """Полный вид на состояние агента — фиксированный кортеж из 34 значений
    (18 — до дня 10, 22 — до дня 11, 26 — до дня 12, 29 — до дня 13, 31 — до
    дня 14; день 14 добавляет три), позиционно раскладывающийся в
    `VIEW_OUTPUTS`. Порядок — часть контракта
    обработчиков ниже.

    Значения всех выпадающих списков — тоже часть вида: иначе после
    переключения агента панель показывала бы одного, а списки — другого.

    `question` передаёт только обработчик кнопки «Набить контекст» — это
    единственный случай, когда интерфейс сам знает, что лежит в поле ввода.
    Тогда бюджет считается вместе с этим текстом, и панель краснеет **до**
    нажатия «Отправить». Остальные обработчики зовут `_view()` как раньше, и
    бюджет показывается для стека без нового вопроса.
    """
    state = agent.debug_state()
    last_call = state["last_call"]
    reasoning = (last_call or {}).get("reasoning") or ""
    title = _agent_title(agent)
    messages = state["messages"]
    model = state["config"]["model"]
    # Вид на контекст агент уже посчитал в `debug_state()` — второй раз его
    # считать незачем. Исключение одно: заполнитель в поле ввода, ради
    # которого весь бюджет и пересчитывается вместе с ним.
    view = asdict(agent.context_view(question)) if question else state["context_view"]
    # Файл перечитывается на каждом событии интерфейса — поллинга и
    # автообновления, как и на дне 6, здесь нет. С дня 11 так же
    # перечитывается файл долговременной памяти: «в памяти» и «на диске» в
    # кадре рядом.
    session_file = STORE.read_file(state["session_id"])
    long_term_file = LONG_TERM.read_file()
    # Checkpoint'ы посчитаны заранее: список «Checkpoint» открывает значением
    # последний (§7.2), и вычислять это внутри самого кортежа было бы нечитаемо.
    checkpoint_choices = _checkpoint_choices(agent)
    return (
        # 1. чат = стек агента; номер в подписи — чей именно диалог показан
        gr.update(
            value=messages,
            label=f"Диалог агента {title} (рендерится из стека агента)",
        ),
        status,                                  # 2. строка статуса
        state["config"],                         # 3. конфиг агента
        _metrics_md(last_call, state["invariants"]),  # 4. метрики последнего вызова
        _totals_md(title, state["totals"], model),  # 5. за время жизни
        _process_md(state["process"]),           # 6. счётчики процесса
        gr.update(visible=bool(reasoning)),      # 7. аккордеон с reasoning
        reasoning,                               # 8. текст reasoning
        # 9. стек сообщений как JSON
        gr.update(
            value=messages,
            label=f"Стек сообщений агента {title} (растёт на 2 за ход)",
        ),
        # 10. переключатель агентов: список пересобирается на каждом событии,
        #     сам по себе он не обновляется — поллинга здесь нет
        gr.update(choices=_agent_choices(agent), value=agent.number),
        # 11. список пресетов подтягивается к конфигу активного агента
        gr.update(value=agent.config.name),
        # 12. блок «Хранилище»: сессия, файл, сколько восстановлено
        _storage_md(state, session_file),
        # 13. сырое содержимое файла сессии
        gr.update(
            value=session_file,
            label=f"Файл сессии {agent.session_id} на диске",
        ),
        # Значения дня 8 — в конце кортежа, как и ключи в `debug_state()`.
        # 14. бюджет контекста: оценка того, что уйдёт в модель
        _context_md(
            view["usage"],
            messages,
            state["calibration"],
            state["totals"]["calibration_calls"],
            question,
        ),
        # 15. рост по ходам: таблица из журнала агента
        gr.update(value=_turns_table(state["turns"])),
        # Значения дня 9 — следом за ними, и тоже в конце.
        # 16. список стратегий подтягивается к активной стратегии агента
        gr.update(value=state["strategy"]),
        # 17. «Контекст: что уходит в модель»
        _flow_md(view, state["totals"], model),
        # 18. «Память стратегии»: сводка текстом
        _memory_update(view),
        # Значения дня 10 — в конце кортежа и в конце VIEW_OUTPUTS (§7.6).
        # 19. переключатель «Ветка диалога» — семейство агента, живой список
        gr.update(choices=_branch_choices(agent), value=agent.number),
        # 20. переключатель «Checkpoint» — у списка своего обработчика нет,
        #     это вход кнопки «Ветка от checkpoint'а»; значение — последний
        gr.update(
            choices=checkpoint_choices,
            value=checkpoint_choices[-1] if checkpoint_choices else None,
        ),
        # 21. блок «Ветки диалога»
        _branches_md(agent),
        # 22. таблица «Агенты процесса»
        gr.update(value=_agents_table()),
        # Значения дня 11 — в конце кортежа и в конце VIEW_OUTPUTS (§7.6).
        # 23. переключатель «Слои памяти в запросе» подтягивается к агенту
        gr.update(value=list(state["request_layers"])),
        # 24. группа кандидатов: пункты — кандидаты агента, ничего не отмечено
        _candidates_update(state),
        # 25. блок «Слои памяти»
        _layers_md(state, view),
        # 26. сырое содержимое файла долговременной памяти — рядом с файлом
        #     сессии: два файла рядом и есть «хранятся отдельно»
        gr.update(
            value=long_term_file,
            label=(
                f"Файл долговременной памяти на диске · {LONG_TERM.path}"
                if long_term_file is not None
                else f"Файл долговременной памяти · {LONG_TERM.path} — файла нет"
            ),
        ),
        # Значения дня 12 — в конце кортежа и в конце VIEW_OUTPUTS (§7.7).
        # 27. переключатель «Профиль в запросе» — пункты и значение подтянуты
        #     к агенту; нет профиля у агента — список пуст и неактивен
        gr.update(
            choices=(
                _profile_choice_options(state["profile"]["choices"])
                if state["profile"] is not None
                else []
            ),
            value=state["profile"]["choice"] if state["profile"] is not None else None,
            interactive=state["profile"] is not None,
        ),
        # 28. блок «Профиль»
        _profile_md(state),
        # 29. сырое содержимое файла профиля — рядом с файлами сессии и
        #     долговременной памяти: три файла рядом и есть «три разные вещи»
        gr.update(
            value=PROFILE.read_file(),
            label=(
                f"Файл профиля на диске · {PROFILE.path}"
                if PROFILE.read_file() is not None
                else f"Файл профиля · {PROFILE.path} — файла нет"
            ),
        ),
        # Значения дня 13 — в конце кортежа и в конце VIEW_OUTPUTS (§7.7).
        # 30. переключатель «Состояние задачи в запросе» — значение и
        #     активность подтянуты к агенту; нет автомата — неактивен
        _task_in_request_update(state),
        # 31. блок «Состояние задачи»
        _task_md(state, view),
        # Значения дня 14 — в конце кортежа и в конце VIEW_OUTPUTS (§8.8).
        # 32. переключатель «Инварианты в запросе» — значение и активность
        #     подтянуты к агенту; нет книги — неактивен
        _invariants_in_request_update(state),
        # 33. блок «Инварианты»
        _invariants_md(state, view),
        # 34. сырое содержимое файла инвариантов — рядом с файлами сессии,
        #     долговременной памяти и профиля: четыре файла рядом и есть
        #     «хранится отдельно от диалога». Перечитывается на каждом
        #     событии, как остальные (спецификация называла тридцать три
        #     значения без него; файл в `_view()`, а не в выходах редактора,
        #     чтобы правка из соседней вкладки не оставляла его устаревшим).
        gr.update(
            value=INVARIANTS.read_file(),
            label=(
                f"Файл инвариантов на диске · {INVARIANTS.path}"
                if INVARIANTS.read_file() is not None
                else f"Файл инвариантов · {INVARIANTS.path} — файла нет"
            ),
        ),
    )


# --- Обработчики событий -------------------------------------------------
# Все обработчики возвращают кортеж вида (агент, поле ввода, *_view(...)) —
# одинаковый набор выходов у всех событий, чтобы панель обновлялась целиком
# и не расходилась с состоянием агента.

def _new_agent(preset_name: str) -> Agent:
    """Новый экземпляр агента с выбранным конфигом и пустым стеком. Активный
    агент живёт в `gr.State`, то есть у каждой открытой вкладки браузера он
    свой; глобального «текущего агента» в модуле нет. Сам экземпляр при этом
    попадает в реестр процесса и остаётся доступен в переключателе после того,
    как вкладка перешла на другого.

    С дня 7 у каждого агента есть своя сессия в хранилище. Файла до первого
    успешного ответа не появляется: `create_session` только выдаёт номер.
    """
    session_id = STORE.create_session(preset_name)
    return Agent(
        PRESETS[preset_name],
        session_id=session_id,
        store=STORE,
        # Набор стратегий свой у каждого агента: у «Сводки» есть память, и
        # относится она к конкретному диалогу.
        strategies=make_strategies(),
        # День 9 показывает «было → стало», и начинать надо с «было»:
        # новый агент всегда стартует на «Всей истории».
        strategy=DEFAULT_STRATEGY,
        # Модель памяти своя у каждого агента (рабочая память — это
        # сессия), долговременная память — одна на процесс (день 11, §7.1).
        memory=make_memory(),
        long_term=LONG_TERM,
        # Профиль один на процесс, роутер свой у каждого агента — тем же
        # приёмом (день 12, §7.1).
        profile=PROFILE,
        router=make_router(),
        # Автомат задачи свой у каждого агента — тем же приёмом (день 13, §7.1).
        task=make_task_machine(),
        # Книга инвариантов своя у каждого агента (сессионные инварианты —
        # это сессия), постоянные — одно хранилище на процесс (день 14, §8.1).
        invariants=make_invariants(),
        invariant_store=INVARIANTS,
    )


def _spawn(preset_name: str, reason: str):
    """Общий хвост для всех трёх способов родить агента (открытие страницы,
    смена пресета, кнопка «Новый агент») — различается только причина в
    строке статуса."""
    agent = _new_agent(preset_name)
    return (
        agent,
        gr.update(),
        *_view(
            agent,
            f"{reason} Поднят агент {_agent_title(agent)} с пустым стеком, "
            f"сессия `{agent.session_id}` — файла до первого ответа не будет. "
            f"Агентов в реестре: {process_stats()['agents_alive']}.",
        ),
    )


def _agents_word(count: int) -> str:
    """«1 агент» / «3 агента» / «5 агентов»: строку статуса читает человек."""
    if 11 <= count % 100 <= 14:
        return "агентов"
    match count % 10:
        case 1:
            return "агент"
        case 2 | 3 | 4:
            return "агента"
        case _:
            return "агентов"


def _restore_agents() -> int:
    """Поднимает агентов по файлам сессий — один раз при старте процесса.

    Вызывается на модуле, а не в `demo.load`: `demo.load` срабатывает на
    каждую открытую вкладку браузера и плодил бы копии одних и тех же
    сессий. Порядок — по возрастанию `session_id` (он же порядок создания),
    поэтому восстановленные агенты получают номера процесса #1..#N в том же
    порядке, в каком сессии заводились.

    Пресет восстанавливается по имени из `PRESETS`: на диске лежит только
    имя, а системный промпт, модель и параметры берутся из кода — правка
    промпта должна доезжать до восстановленных агентов. Имени нет в наборе
    (пресет переименовали или убрали) — берём пресет по умолчанию; падать
    из-за этого приложение не должно.
    """
    restored = 0
    for info in STORE.sessions():
        preset_name = info.preset
        if preset_name not in PRESETS:
            logger.warning(
                "сессия %s: пресет «%s» в PRESETS не найден, поднимаем на «%s»",
                info.session_id, preset_name, DEFAULT_PRESET,
            )
            preset_name = DEFAULT_PRESET
        # Экземпляр никуда не присваивается намеренно: агент сам встаёт в
        # реестр процесса, и оттуда его берут `on_load` и переключатель.
        # Стратегия здесь не передаётся: у восстановленного агента умолчания
        # нет — он берёт ту, что записана в его файле сессии.
        # Рабочая память приезжает из того же файла сессии (`context.working`),
        # кандидатов у восстановленного агента нет — они были вопросом
        # прошлого процесса (день 11, §2.4).
        Agent(
            PRESETS[preset_name],
            session_id=info.session_id,
            store=STORE,
            strategies=make_strategies(),
            memory=make_memory(),
            long_term=LONG_TERM,
            # Профиль общий, роутер свой у восстановленного агента —
            # `авто`, режима прошлого хода нет (день 12, §5.7).
            profile=PROFILE,
            router=make_router(),
            # Автомат задачи свой у восстановленного агента — состояние и
            # журнал из файла, переключатель включён (день 13, §5.8).
            task=make_task_machine(),
            # Книга своя, сессионные инварианты — из файла сессии,
            # переключатель включён, последнего конфликта нет (день 14, §6.8).
            invariants=make_invariants(),
            invariant_store=INVARIANTS,
        )
        restored += 1
    logger.info(
        "старт процесса: восстановлено %d %s из %s",
        restored, _agents_word(restored), display_path(STORE.data_dir),
    )
    return restored


def on_load(preset_name: str):
    """Открытие страницы.

    Если в реестре уже кто-то есть (агенты восстановлены при старте процесса
    или подняты из другой вкладки), новый агент не создаётся: активным
    становится последний в реестре — это и есть сессия с наибольшим номером,
    последняя заведённая. Так после перезапуска пользователь открывает
    страницу и видит свой диалог, а не пустой чат рядом с восстановленными
    агентами в переключателе. Остальные агенты никуда не делись и берутся
    одним кликом в переключателе.
    """
    registry = agents()
    if not registry:
        return _spawn(
            preset_name,
            f"Страница открыта. {_long_term_status()} {_profile_status()} "
            f"{_invariants_status()}",
        )

    target = registry[-1]
    if RESTORED_AT_START:
        # Активным может оказаться агент, созданный уже в этом процессе, —
        # например ветка: у неё `restored_messages` всегда 0 (день 10, §5.2),
        # и «0 сообщ. из файла» рядом с непустым чатом было бы неправдой.
        from_file = (
            f", из файла при старте — {target.restored_messages}"
            if target.restored_messages
            else ""
        )
        status = (
            f"Восстановлено {RESTORED_AT_START} "
            f"{_agents_word(RESTORED_AT_START)} из {display_path(STORE.data_dir)}. "
            f"Активен {_agent_title(target)} · `{target.session_id}`: "
            f"в стеке {len(target.history)} сообщ.{from_file}."
        )
    else:
        status = (
            f"Страница открыта. Активен {_agent_title(target)} · "
            f"`{target.session_id}`: в стеке {len(target.history)} сообщ."
        )
    # День 11 (§7.1): при открытии страницы видно и долговременную память —
    # сколько записей и откуда, или что её пока нет. День 12 (§7.1) —
    # добавляет то же самое про профиль.
    return (
        target,
        gr.update(),
        *_view(
            target,
            f"{status} {_long_term_status()} {_profile_status()} "
            f"{_invariants_status(target)}",
        ),
    )


def on_preset_change(preset_name: str):
    return _spawn(preset_name, "Сменён пресет — это новый агент и новый диалог.")


def on_new_agent(preset_name: str):
    """«Новый агент» — ещё один экземпляр с тем же пресетом. Сменой пресета
    двух агентов с одинаковым конфигом не получить, а именно такая пара
    нагляднее всего показывает, что стек у каждого инстанса свой."""
    return _spawn(preset_name, "Ещё один агент с тем же пресетом.")


def on_delete_agent(agent: Agent | None, preset_name: str):
    """«Удалить агент» — снять активного агента с учёта в процессе и удалить
    его файл сессии.

    Файл удаляется здесь же: иначе «удалённый» агент возвращался бы из файла
    на следующем запуске. Счётчики процесса при этом не откатываются:
    потраченные токены и деньги остались потраченными, а номера агента и
    сессии не переиспользуются. Активным становится последний из оставшихся;
    если реестр опустел — поднимаем нового, экрана без активного агента
    не бывает.
    """
    if agent is None:  # страховка на случай сессии без сработавшего load
        return _spawn(preset_name, "Удалять было нечего.")

    title = _agent_title(agent)
    delete_agent(agent.number)
    try:
        STORE.delete_session(agent.session_id)
    except StorageError as exc:
        # Агента из процесса уже убрали; о том, что файл остался на диске
        # (и агент вернётся при следующем запуске), честнее сказать вслух.
        logger.warning("сессия %s: файл не удалён — %s", agent.session_id, exc)
    # Удаление уносит краткосрочную и рабочую память этой сессии — они в её
    # файле; долговременная память общая и остаётся (день 11, §5.7).
    kept = (
        f"Его разговор и рабочая память удалены вместе с файлом сессии, "
        f"долговременная память осталась ({_long_term_records()})."
    )
    remaining = agents()
    if not remaining:
        return _spawn(
            preset_name,
            f"Агент {title} удалён, в реестре не осталось никого. {kept}",
        )

    target = remaining[-1]
    return (
        target,
        gr.update(),
        *_view(
            target,
            f"Агент {title} удалён из процесса. {kept} Активен "
            f"{_agent_title(target)}: в стеке {len(target.history)} сообщ. "
            f"Агентов в реестре: {len(remaining)}.",
        ),
    )


def on_switch_agent(agent_number: int, preset_name: str):
    """Переключение на другого агента процесса. Новых агентов не создаёт:
    берём готовый экземпляр из реестра и перерисовываем панель от него —
    следующий вопрос уйдёт уже ему."""
    target = agent_by_number(agent_number)
    if target is None:
        # Агента в реестре нет: его удалили (в том числе из этой же вкладки,
        # пунктом «удалён») или приложение перезапускали, а страница осталась
        # открытой. Поднимаем нового, чтобы вкладка не осталась без состояния.
        return _spawn(
            preset_name,
            "Выбранного агента в процессе больше нет.",
        )
    return (
        target,
        gr.update(),
        *_view(
            target,
            f"Активен агент {_agent_title(target)}: показан его стек "
            f"({len(target.history)} сообщ.), следующий вопрос уйдёт ему.",
        ),
    )


def on_strategy_change(agent: Agent | None, strategy_name: str, preset_name: str):
    """Переключение стратегии управления контекстом — единственный новый
    обработчик дня 9; день 10 добавляет ему два новых пункта в списке (окно,
    факты) и переписывает статус без арифметики и без слова «свёртка»
    (спецификация дня 10, §7.1).

    Нового агента не создаёт и стек не трогает: меняется способ сборки
    запроса, а не агент и не диалог. Этим переключатель стратегии и отличается
    от соседнего списка пресетов, и сказать об этом надо вслух — в `info`
    списка и в строке статуса.
    """
    if agent is None:  # страховка на случай сессии без сработавшего load
        agent = _new_agent(preset_name)

    agent.set_strategy(strategy_name)
    view = agent.context_view()
    state = view.state
    if view.pending_label:
        # Задача назрела — типичный случай, когда на стратегию с памятью
        # переключаются после нескольких ходов без неё (а у фактов задача
        # назревает на каждом ходе — это её штатное состояние). Следующий
        # запрос уйдёт уже после служебного вызова, и «все сообщения стека»
        # были бы неправдой. Число сообщений — из `pending_sent_messages`,
        # который знает стратегия; арифметики в интерфейсе больше нет.
        label = view.pending_label[:1].upper() + view.pending_label[1:]
        status = (
            f"Стратегия «{agent.strategy.name}». ⏳ {label} перед следующим "
            f"ответом, и в запрос уйдут {state.memory_label.lower()} и "
            f"{view.pending_sent_messages} сообщ. из {view.history_messages}."
        )
    elif state.memory_text:
        status = (
            f"Стратегия «{agent.strategy.name}»: в запрос уйдут "
            f"{state.memory_label.lower()} и {view.sent_messages} сообщ. из "
            f"{view.history_messages}."
        )
    else:
        # У стратегии без памяти (окно, вся история) сообщений уходит ровно
        # `sent_messages`; если это меньше стека — остальное отброшено, и
        # молчать об этом нечестно (окно теряет детали, а не сжимает).
        status = (
            f"Стратегия «{agent.strategy.name}»: в запрос уйдут "
            f"{view.sent_messages} сообщ. из {view.history_messages}."
        )
        if view.sent_messages < view.history_messages:
            status += " Остальные отброшены из запроса без замены."
    return (
        agent,
        gr.update(),
        *_view(agent, f"{status} Стек не тронут, история на диске полная."),
    )


def _reply_status(agent: Agent, reply) -> str:
    """Статус хода по `AgentReply` — общий код для «Отправить» и «Начать
    задачу с этим сообщением» (день 13, §7.2): вынесен из `on_send()` без
    изменения существующих текстов."""
    if reply.ok:
        status = (
            f"Ответ получен за {reply.elapsed:.2f} s. "
            f"В стеке агента {_agent_title(agent)} {len(agent.history)} сообщ."
        )
        # Ход, на котором сработал служебный вызов, говорит об этом вслух: на
        # видео момент должен читаться без панели. Одной короткой фразой,
        # словами стратегии (`label`, `memory_update`) — не словом «свёртка»:
        # у фактов эта строка появляется на каждом ходе (день 10, §7.1).
        service = reply.service_call
        if service is not None and service.ok:
            status += (
                f" 🧵 Перед ответом — {service.label}: {service.memory_update}, "
                f"{_fmt_cost(service.cost_usd)}."
            )
            if service.finish_reason == "length":
                status += (
                    " ⚠️ Ответ служебного вызова упёрся в потолок и оборван "
                    "на полуслове — применён как есть."
                )
        elif service is not None:
            status += (
                f" ⚠️ Служебный вызов не удался ({service.error}) — ход "
                f"состоялся, память стратегии не сдвинулась, в модель ушло "
                f"всё неучтённое; вызов повторится на следующем ходе."
            )
        # Страж инвариантов (день 14, §8.6) — одной короткой фразой перед
        # разбором памяти: конфликт называем прямо, ход при этом не
        # отменялся, и отказать должен сам ассистент.
        guard_call = reply.guard_call
        if reply.conflict:
            why = f"«{reply.conflict_note}»" if reply.conflict_note else "см. блок «Инварианты»"
            status += (
                f" ⛔ Конфликт с инвариантами {reply.conflict}: {why} — "
                f"ассистент должен отказать и объяснить."
            )
            if not agent.invariants_in_request:
                status += (
                    " ⚠️ «Инварианты в запросе» выключено: модель об "
                    "ограничениях не знает — страж их видит, ответ им не "
                    "следует."
                )
        elif guard_call is not None and not guard_call.ok:
            status += (
                f" ⚠️ Страж инвариантов не удался ({guard_call.error}) — ход "
                f"состоялся, конфликт этого сообщения не проверен."
            )
        # Разбор памяти (день 11, §7.4) — одной короткой фразой после
        # служебного вызова стратегии, словами самого разбора.
        memory_call = reply.memory_call
        if memory_call is not None and memory_call.ok:
            status += f" 🧠 Память — {memory_call.memory_update}."
            if memory_call.finish_reason == "length":
                status += (
                    " ⚠️ Ответ разбора памяти упёрся в потолок и оборван на "
                    "полуслове — применён как есть."
                )
        elif memory_call is not None:
            status += (
                f" ⚠️ Разбор памяти не удался ({memory_call.error}) — ход "
                f"состоялся, рабочая память и кандидаты не сдвинулись; разбор "
                f"повторится на следующем ходе."
            )
        # Трекер задачи (день 13, §7.5) — тем же приёмом, после разбора
        # памяти: короткая фраза про переход этого хода. «Событий нет» не
        # показывается — это штатный результат почти каждого хода задачи.
        task_call = reply.task_call
        task_note = reply.task_note
        if task_note and task_note != "событий нет":
            if task_note.startswith("отклонено"):
                status += f" ⚠️ Переход не состоялся: {task_note}."
            elif task_note.startswith("ждёт подтверждения"):
                # Переход остановил инвариант `с подтверждением` (день 14,
                # §8.3): состояние не менялось, решает человек.
                status += (
                    f" ⏳ Задача — {task_note}. Отметить переход — кнопкой "
                    f"«✅ Подтвердить переход»."
                )
            else:
                status += f" 📋 Задача — {task_note}."
        elif task_call is not None and not task_call.ok:
            status += (
                f" ⚠️ Трекер задачи не удался ({task_call.error}) — состояние "
                f"задачи не тронуто."
            )
        # Поле ввода чистим только при успехе; при ошибке вопрос остаётся
        # в поле, чтобы его можно было отправить повторно.
        return status
    status = f"❌ {reply.error}"
    service = reply.service_call
    if service is not None and service.ok:
        # Применённое обновление до упавшего вызова уже оплачено и
        # сохранено: следующий вопрос уйдёт с ним, и это не должно стать
        # сюрпризом.
        status += (
            f" 🧵 Перед вызовом при этом сработал служебный вызов — "
            f"{service.label}: {service.memory_update} — применённая "
            f"память уйдёт со следующим вопросом."
        )
    memory_call = reply.memory_call
    if memory_call is not None and memory_call.ok:
        status += (
            f" 🧠 Разбор памяти перед вызовом при этом применился — "
            f"{memory_call.memory_update} — рабочая память уйдёт со "
            f"следующим вопросом."
        )
    if reply.task_event:
        status += (
            f" 📋 Переход трекера задачи перед вызовом при этом применился — "
            f"{reply.task_note} — состояние уйдёт со следующим вопросом."
        )
    return status


def on_send(agent: Agent | None, message: str, preset_name: str):
    """Ход диалога. Вся работа — один вызов `agent.ask()`: стек сообщений
    ведёт агент, интерфейс только перерисовывает его состояние.

    Функция намеренно не генератор: Gradio показывает нормальный оверлей
    ожидания (спиннер и таймер) только для функций с одним `return`.
    """
    if agent is None:  # страховка на случай сессии без сработавшего load
        agent = _new_agent(preset_name)

    message = (message or "").strip()
    if not message:
        return (
            agent,
            gr.update(),
            *_view(agent, "Введите вопрос — пустой запрос в модель не уходит."),
        )

    reply = agent.ask(message)
    status = _reply_status(agent, reply)
    message_update = "" if reply.ok else gr.update()

    # Если есть кандидаты — это главное в статусе (§7.4): человеку есть что
    # решить, и без его решения в долговременную память ничего не попадёт.
    waiting = agent.memory_state.candidates if agent.memory_state else []
    if waiting:
        status = (
            f"🧠 **Ждут вашего решения: "
            f"{', '.join(candidate.key for candidate in waiting)}** — отметьте "
            f"в ряду «Кандидаты в долговременную память» под чатом и сохраните "
            f"или отклоните. {status}"
        )

    return (agent, message_update, *_view(agent, status))


def on_fill_context(agent: Agent | None, thousands: float | None, preset_name: str):
    """«Набить контекст» — положить в поле вопроса текст-заполнитель нужного
    размера. Ничего не отправляет: отправляет человек обычной кнопкой
    «Отправить», и на видео видно, что это такой же запрос, как любой другой,
    а не спецрежим.

    Настоящий диалог до переполнения окна не набрать (в окне модели больше
    миллиона токенов), а подставлять фальшивый лимит нечестно: тогда ломался
    бы наш собственный код, а не API.
    """
    if agent is None:  # страховка на случай сессии без сработавшего load
        agent = _new_agent(preset_name)

    requested = int(thousands or 0) * 1000
    filler = filler_text(requested)
    if not filler:
        return (
            agent,
            gr.update(),
            *_view(
                agent,
                "Размер заполнителя должен быть больше нуля — "
                "поле ввода не тронуто.",
            ),
        )

    estimated = estimate_tokens(filler)
    usage = agent.context_usage(filler)
    clamped = (
        f" (запрошено {_fmt_int(requested)}, обрезано до предела "
        f"{_fmt_int(FILLER_MAX_TOKENS)})"
        if requested > FILLER_MAX_TOKENS
        else ""
    )
    if usage.limit is None:
        budget = f"вместе со стеком это ≈{_fmt_int(usage.used)} токенов (лимит модели н/д)"
    else:
        budget = (
            f"вместе со стеком это ≈{_fmt_int(usage.used)} из "
            f"{_fmt_int(usage.available)} доступных "
            f"({usage.ratio * 100:.1f}%)"
        )
    status = (
        f"В поле положен заполнитель ≈{_fmt_int(estimated)} токенов{clamped}; "
        f"{budget}. Отправлять — обычной кнопкой «Отправить»."
    )
    logger.info("заполнитель: %s токенов, %s символов", estimated, len(filler))
    return (agent, filler, *_view(agent, status, question=filler))


def on_reset(agent: Agent | None, preset_name: str):
    """«Сбросить диалог» очищает стек сообщений агента. Тот же вызов удаляет
    и файл сессии — следствие правила «пустой список сообщений удаляет файл»:
    нет контекста, нет и сессии. Счётчики за время жизни агента при этом
    сохраняются, а checkpoint'ы и происхождение ветки уходят вместе с
    диалогом (день 10) — они ссылались на диалог, которого больше нет; ветки,
    созданные раньше, при этом не трогаются, у них свои файлы."""
    if agent is None:  # страховка на случай сессии без сработавшего load
        agent = _new_agent(preset_name)
        note = "агент поднят заново"
    else:
        # Что сказать про checkpoint'ы, происхождение и ветки, считается до
        # reset(): после него у агента нет ни checkpoint'ов, ни происхождения,
        # и живую связь с ветками уже не проверить. Ветки «от этого диалога» —
        # только живые (`_live_branches`): родитель и соседние ветки
        # сбрасываемой ветки к ним не относятся (спецификация дня 10, §7.2).
        had_checkpoints = bool(agent.checkpoints)
        was_branch = agent.branch is not None
        branches = len(_live_branches(agent))
        agent.reset()
        note = f"файл сессии `{agent.session_id}` удалён"
        if had_checkpoints and was_branch:
            note += ", checkpoint'ы и происхождение ветки очищены вместе с диалогом"
        elif had_checkpoints:
            note += ", checkpoint'ы очищены вместе с диалогом"
        elif was_branch:
            note += ", происхождение ветки очищено вместе с диалогом"
        if branches:
            note += f" (веток от этого диалога — {branches}, они не тронуты)"
    return (
        agent,
        gr.update(),
        *_view(
            agent,
            f"Диалог агента {_agent_title(agent)} сброшен: стек сообщений пуст, "
            f"{note}, счётчики агента сохранены. Рабочая память и кандидаты "
            f"ушли вместе с диалогом, **долговременная память не тронута** "
            f"({_long_term_records()}). Сессионные инварианты — тоже ушли, "
            f"постоянные не тронуты.",
        ),
    )


# --- Ветки диалога: обработчики (день 10) ---------------------------------

def on_save_checkpoint(agent: Agent | None, preset_name: str):
    """«Сохранить checkpoint» — фиксирует конец текущей истории вместе со
    снимком памяти стратегий. Диалог продолжать можно: checkpoint останется
    тем, каким был, — ветвиться от него можно и после десяти новых ходов."""
    if agent is None:  # страховка на случай сессии без сработавшего load
        agent = _new_agent(preset_name)

    before = {cp.id for cp in agent.checkpoints}
    checkpoint = agent.save_checkpoint()
    if checkpoint is None:
        status = (
            f"Checkpoint не сохранён: в стеке агента {_agent_title(agent)} "
            f"нет сообщений — сохранять нечего."
        )
    elif checkpoint.id in before:
        status = (
            f"Такой checkpoint уже есть: {checkpoint.id} — история и память "
            f"стратегий с прошлого сохранения не изменились, новый не создан."
        )
    else:
        status = (
            f"Checkpoint {checkpoint.id} сохранён: {checkpoint.messages} "
            f"сообщ., стратегия «{checkpoint.strategy}», снимок памяти "
            f"стратегий и рабочей памяти. Ветку от него создаёт «Ветка от "
            f"checkpoint'а»; диалог можно продолжать — checkpoint останется "
            f"тем, каким был."
        )
    return (agent, gr.update(), *_view(agent, status))


def on_fork(agent: Agent | None, checkpoint_choice: str | None, preset_name: str):
    """«Ветка от checkpoint'а» — новая сессия с общим началом истории.

    Номер сессии заводится только после того, как checkpoint найден: впустую
    он не пропадает. Ветка становится активной, исходный диалог остаётся в
    реестре и в списке «Ветка диалога» нетронутым.
    """
    if agent is None:  # страховка на случай сессии без сработавшего load
        agent = _new_agent(preset_name)

    checkpoint_id = _checkpoint_id_from_choice(checkpoint_choice)
    if checkpoint_id is None or checkpoint_id not in {cp.id for cp in agent.checkpoints}:
        return (
            agent,
            gr.update(),
            *_view(
                agent,
                "Ветка не создана: сначала выберите checkpoint в списке "
                "«Checkpoint» (кнопка «Сохранить checkpoint» — если его ещё нет).",
            ),
        )

    session_id = STORE.create_session(agent.config.name)
    # Ветке — свежая модель памяти: рабочую память она получит копией из
    # снимка checkpoint'а, долговременную — от родителя тем же хранилищем
    # (день 11, §7.1). Роутер — тоже свежий («авто», без режима прошлого
    # хода), хранилище профиля ветка берёт у родителя (день 12, §5.7).
    # Автомат задачи — тоже свежий, с состоянием и журналом из снимка
    # checkpoint'а, переключатель включён (день 13, §5.8).
    # Книга инвариантов — тоже свежая: сессионные инварианты ветка получает
    # копией из снимка checkpoint'а, постоянные — те же, файл один; переключатель
    # «Инварианты в запросе» включён (день 14, §6.8).
    branch = agent.fork(
        checkpoint_id, session_id, make_strategies(), make_memory(),
        make_router(), make_task_machine(), make_invariants(),
    )
    if branch is None:
        return (
            agent,
            gr.update(),
            *_view(
                agent,
                f"Ветка от checkpoint'а «{checkpoint_id}» не создана — см. лог.",
            ),
        )
    working = branch.memory_state.working_items if branch.memory_state else 0
    return (
        branch,
        gr.update(),
        *_view(
            branch,
            f"Создана ветка {branch.session_id} от {agent.session_id} · "
            f"{checkpoint_id}: общие {branch.branch.messages} сообщ. "
            f"скопированы, стратегия «{branch.strategy.name}», файл уже на "
            f"диске. Рабочая память — копия из снимка ({working} "
            f"{_records_word(working)}), долговременная — та же, кандидатов "
            f"нет; сессионные инварианты — копия из снимка "
            f"({_session_count(branch)}), "
            f"постоянные — те же. Исходный диалог не тронут — он в списке "
            f"«Ветка диалога».",
        ),
    )


def on_switch_branch(agent_number: int, preset_name: str):
    """Переключатель «Ветка диалога» — то же самое, что переключатель
    агентов (`on_switch_agent`), со своим статусом: диалог — это про то, какая
    история лежит в стеке, а не про то, что это отдельный агент."""
    target = agent_by_number(agent_number)
    if target is None:
        return _spawn(preset_name, "Выбранного диалога в процессе больше нет.")

    if target.branch is not None:
        status = (
            f"Активна ветка {target.session_id} (от {target.branch.parent} · "
            f"{target.branch.checkpoint}): в стеке {len(target.history)} "
            f"сообщ., из них общих {target.branch.messages}; следующий "
            f"вопрос уйдёт в неё."
        )
    else:
        branches = len(branch_family(target)) - 1
        status = (
            f"Активен исходный диалог {target.session_id}: "
            f"{len(target.history)} сообщ., веток от него — {branches}."
        )
    return (target, gr.update(), *_view(target, status))


# --- Слои памяти: обработчики (день 11) -----------------------------------

def on_layers_change(agent: Agent | None, layers: list[str] | None, preset_name: str):
    """«Слои памяти в запросе» — какие слои уходят в модель (§7.2).

    Нового агента не создаёт и ничего не пишет: меняется запрос, а не
    память (§2.5). Разбор памяти продолжает работать при любом положении —
    выключенный слой копится, и включённый обратно он сразу полон.
    """
    if agent is None:  # страховка на случай сессии без сработавшего load
        agent = _new_agent(preset_name)

    agent.set_request_layers(layers or [])
    current = agent.request_layers
    parts = [f"Слои в запросе: {', '.join(current) if current else 'нет'}."]
    state = agent.memory_state
    for layer in REQUEST_LAYERS:
        if layer in current:
            continue
        if layer == LAYER_LONG_TERM:
            parts.append(
                f"Долговременная память ({_long_term_records()}) в модель не "
                f"уходит, но разбор продолжает предлагать в неё кандидатов."
            )
        elif layer == LAYER_WORKING:
            items = state.working_items if state else 0
            parts.append(
                f"Рабочая память ({items} {_records_word(items)}) в модель не "
                f"уходит, но разбор продолжает её записывать."
            )
    parts.append("Стек и память не тронуты; краткосрочную память урезает стратегия.")
    return (agent, gr.update(), *_view(agent, " ".join(parts)))


def on_profile_choice(agent: Agent | None, choice: str, preset_name: str):
    """«Профиль в запросе» — что уходит в модель на этот ход (§7.2, §2.3):
    `авто` (роутер выбирает режим), название режима (всегда этот режим) или
    `выключен` (профиль в запрос не уходит).

    Нового агента не создаёт и ничего не пишет: меняется запрос, а не
    профиль. Событие — `select`, тем же правилом, что у остальных
    выпадающих списков (день 6): `_view()` синхронизирует значение списка на
    каждом событии, и `change` переключал бы профиль сам по себе.
    """
    if agent is None:  # страховка на случай сессии без сработавшего load
        agent = _new_agent(preset_name)

    previous = agent.profile_choice
    agent.set_profile_choice(choice)
    current = agent.profile_choice
    if current == CHOICE_AUTO:
        note = "режим выбирает роутер — это +1 вызов модели на ход"
    elif current == CHOICE_OFF:
        note = "профиль не уходит в модель; сам профиль не тронут"
    else:
        note = f"в запрос всегда уходит режим «{current}», роутер не вызывается"
    status = (
        f"Профиль в запросе: {previous} → {current}. {note[:1].upper()}{note[1:]}. "
        f"Стек, память и профиль не тронуты."
    )
    return (agent, gr.update(), *_view(agent, status))


def on_accept_candidates(agent: Agent | None, keys: list[str] | None, preset_name: str):
    """«Сохранить в долговременную» — решение человека по отмеченным
    кандидатам (§7.2). Запись в файл — сразу; сбой записи оставляет
    кандидатов ждать решения."""
    if agent is None:  # страховка на случай сессии без сработавшего load
        agent = _new_agent(preset_name)

    keys = list(keys or [])
    if not keys:
        return (
            agent,
            gr.update(),
            *_view(agent, "Отметьте кандидатов, которых сохранить в долговременную память."),
        )
    decision = agent.accept_candidates(keys)
    if decision.error:
        status = (
            f"⚠️ Кандидаты не сохранены: {decision.error}. Они остались ждать "
            f"решения — сохранить можно ещё раз."
        )
    elif not decision.keys:
        status = (
            "Отмеченных кандидатов среди ждущих решения уже нет — список "
            "обновлён, отметьте заново."
        )
    else:
        status = (
            f"В долговременную память сохранено: {', '.join(decision.keys)} — с "
            f"этого хода это уходит в запрос каждого агента и каждой новой "
            f"сессии, у которых слой включён. Файл: `{LONG_TERM.path}`."
        )
    return (agent, gr.update(), *_view(agent, status))


def on_reject_candidates(agent: Agent | None, keys: list[str] | None, preset_name: str):
    """«Отклонить» — отмеченные кандидаты убираются, в память ничего не
    пишется (§7.2)."""
    if agent is None:  # страховка на случай сессии без сработавшего load
        agent = _new_agent(preset_name)

    keys = list(keys or [])
    if not keys:
        return (
            agent,
            gr.update(),
            *_view(agent, "Отметьте кандидатов, которых отклонить."),
        )
    decision = agent.reject_candidates(keys)
    if decision.error:
        status = f"⚠️ Отклонить не удалось: {decision.error}."
    elif not decision.keys:
        status = (
            "Отмеченных кандидатов среди ждущих решения уже нет — список "
            "обновлён, отметьте заново."
        )
    else:
        status = (
            f"Отклонено: {', '.join(decision.keys)} — в долговременную память "
            f"не попало; сказанное осталось в истории разговора."
        )
    return (agent, gr.update(), *_view(agent, status))


# --- Состояние задачи: обработчики (день 13, §7.2) -------------------------

def on_start_task(agent: Agent | None, message: str, preset_name: str):
    """«Начать задачу с этим сообщением» — текст поля ввода становится
    целью; следом сразу `ask(message)` тем же сообщением, старт и первый
    ответ — один клик. Отказ не зовёт `ask()`; поле ввода не очищается —
    ни отказом, ни упавшим ответом."""
    if agent is None:  # страховка на случай сессии без сработавшего load
        agent = _new_agent(preset_name)

    message = (message or "").strip()
    if not message:
        return (
            agent,
            gr.update(),
            *_view(agent, "Введите цель задачи — ей станет текст поля ввода."),
        )

    outcome = agent.start_task(message)
    if not outcome.changed:
        return (
            agent,
            gr.update(),
            *_view(agent, f"❌ Задача не начата: {outcome.rejection.reason}."),
        )

    reply = agent.ask(message)
    status = (
        f"📋 Задача начата — этап «планирование»: цель — ваше сообщение, "
        f"план предложит ассистент, утверждаете его вы словами. "
        f"{_reply_status(agent, reply)}"
    )
    message_update = "" if reply.ok else gr.update()
    return (agent, message_update, *_view(agent, status))


def on_task_event(agent: Agent | None, preset_name: str, *, event: str):
    """«Пауза», «Продолжить» и «Отменить задачу» — одна функция на три
    кнопки: событие передаёт `functools.partial` при подписке (`event` —
    keyword-only, чтобы позиционные входы Gradio не столкнулись с ним)."""
    if agent is None:  # страховка на случай сессии без сработавшего load
        agent = _new_agent(preset_name)

    outcome = agent.task_event(event)
    if not outcome.changed:
        allowed = ", ".join(agent.task_view.allowed_human) if agent.task_view else ""
        status = f"Переход «{event}» не выполнен: {outcome.rejection.reason}."
        if outcome.rejection.reason.startswith("запрещено ограничением"):
            # Запрет инварианта не обходится ни кнопкой, ни подтверждением
            # (день 14, §2.4): снять его можно только в редакторе, и это видно.
            status += (
                " Кнопка человека запрет не обходит: чтобы разрешить переход, "
                "выключите этот инвариант флажком «Действует» в редакторе."
            )
        elif allowed:
            status += f" Допустимо сейчас: {allowed}."
    elif event == EVENT_PAUSE:
        view = agent.task_view
        status = (
            f"📋 Задача: пауза — {view.stage}, {view.step_text}. Ожидается: "
            f"{view.expected}. Вопросы задавайте как обычно — задача не "
            f"сдвинется."
        )
    elif event == EVENT_RESUME:
        view = agent.task_view
        status = (
            f"📋 Задача: продолжена — {view.stage}, {view.step_text}. "
            f"Следующий ответ начнётся с напоминания, где остановились."
        )
    else:  # EVENT_CANCEL
        status = (
            "📋 Задача отменена. Новую можно начать кнопкой «Начать задачу "
            "с этим сообщением»."
        )
    return (agent, gr.update(), *_view(agent, status))


def on_task_in_request(agent: Agent | None, enabled: bool, preset_name: str):
    """«Состояние задачи в запросе» — меняет запрос, а не задачу (§5.6):
    трекер вызывается и переходы применяются при любом положении."""
    if agent is None:  # страховка на случай сессии без сработавшего load
        agent = _new_agent(preset_name)

    agent.set_task_in_request(bool(enabled))
    state_word = "включено" if agent.task_in_request else "выключено"
    status = (
        f"Состояние задачи в запросе: {state_word}. Трекер продолжает вести "
        f"задачу; стек, память и задача не тронуты."
    )
    return (agent, gr.update(), *_view(agent, status))


# --- Инварианты: обработчики (день 14, §8.2-8.3) ---------------------------

def on_invariants_in_request(agent: Agent | None, enabled: bool, preset_name: str):
    """«Инварианты в запросе» — меняет запрос, а не работу (§6.6): страж
    вызывается, конфликт пишется и ограничения переходов действуют при любом
    положении. Нового агента не создаёт и ничего не пишет."""
    if agent is None:  # страховка на случай сессии без сработавшего load
        agent = _new_agent(preset_name)

    agent.set_invariants_in_request(bool(enabled))
    if agent.invariants_in_request:
        status = (
            "Инварианты в запросе: включено. Блок уходит в модель первым из "
            "блоков; страж сверяет, ограничения переходов действуют."
        )
    else:
        status = (
            "Инварианты в запросе: выключено. Страж продолжает сверять и "
            "показывать конфликты; ограничения переходов действуют."
        )
    return (agent, gr.update(), *_view(agent, status))


def on_confirm_transition(agent: Agent | None, preset_name: str):
    """«✅ Подтвердить переход» (§8.3): человек разрешает переход трекера,
    который остановил инвариант `с подтверждением`. Активна всегда: нечего
    подтверждать — отказ, и в статусе видно почему."""
    if agent is None:  # страховка на случай сессии без сработавшего load
        agent = _new_agent(preset_name)

    outcome = agent.confirm_transition()
    if outcome.changed:
        note = outcome.note.replace("подтверждено пользователем", "подтверждено вами")
        status = f"📋 Задача: {note}."
    else:
        reason = outcome.rejection.reason if outcome.rejection else outcome.note
        if reason.startswith("подтверждать нечего"):
            status = (
                "Подтверждать нечего: переходов, ждущих подтверждения, "
                "сейчас нет."
            )
        else:
            status = f"Переход не подтверждён: {reason}."
    return (agent, gr.update(), *_view(agent, status))


# --- Редактор инвариантов: обработчики (день 14, §8.5) ---------------------
# Второе исключение из «интерфейс — вид на состояние агента» после редактора
# профиля: это форма ввода, а её поля не входят в `_view()`/`VIEW_OUTPUTS` —
# иначе любое событие затирало бы несохранённую правку. Свои выходы
# (`INVARIANT_EDITOR_OUTPUTS`, определены при сборке интерфейса). Список
# «Инвариант для правки» — часть формы: его пункты пересобираются только
# загрузкой страницы, выбором пункта, сохранением, удалением и фокусом на самом
# списке (сессионные инварианты у каждого агента свои, а форма агента не знает).

INVARIANT_NEW = "+ новый"
NO_EVENT = "— (не про переходы)"
NO_MODE = "— (не выбран)"


def _session_count(agent: Agent) -> int:
    """Сколько сессионных инвариантов в книге агента, включая выключенные."""
    view = agent.invariant_view
    return sum(1 for item in view.items if item.scope == SCOPE_SESSION) if view else 0


def _invariant_lists(agent: Agent) -> tuple[list[Invariant], list[Invariant]]:
    """Постоянные и сессионные инварианты книги агента, в порядке номеров."""
    view = agent.invariant_view
    items = view.items if view else []
    return (
        [inv for inv in items if inv.scope == SCOPE_ALWAYS],
        [inv for inv in items if inv.scope == SCOPE_SESSION],
    )


def _with_ids(items: list[Invariant], scope: str) -> list[Invariant]:
    """Номера по порядку в списке одной области — для сообщений проверки."""
    return [
        replace(inv, id=f"{ID_PREFIX[scope]}{n}", scope=scope)
        for n, inv in enumerate(items, start=1)
    ]


def _invariant_choices(agent: Agent | None) -> list[tuple[str, str]]:
    """Пункты списка «Инвариант для правки»: «+ новый» и книга агента —
    подпись с номером, областью и началом формулировки, значение — номер."""
    choices = [(INVARIANT_NEW, INVARIANT_NEW)]
    view = agent.invariant_view if agent is not None else None
    for inv in view.items if view else []:
        off = " · выключен" if not inv.active else ""
        text = inv.text if len(inv.text) <= 60 else inv.text[:57] + "…"
        choices.append(
            (f"{inv.id} · {SCOPE_LABELS.get(inv.scope, inv.scope)}{off} · {text}", inv.id)
        )
    return choices


def _invariant_pick_update(agent: Agent | None, value: str | None):
    """Список правки с пересобранными пунктами; значение — прежнее, если оно
    ещё есть среди пунктов, иначе «+ новый»."""
    choices = _invariant_choices(agent)
    known = {choice[1] for choice in choices}
    return gr.update(choices=choices, value=value if value in known else INVARIANT_NEW)


def _empty_invariant_form():
    """Значения формы для нового инварианта: пустой текст, область «всегда»,
    без перехода, «Действует»."""
    return "", SCOPE_ALWAYS, NO_EVENT, NO_MODE, True


def _invariant_editor_outputs(agent: Agent | None, pick: str | None, clear_form: bool = False):
    """Кортеж для `INVARIANT_EDITOR_OUTPUTS`: список правки с пересобранными
    пунктами и поля формы — прежние (`gr.update()`) или очищенные."""
    fields = _empty_invariant_form() if clear_form else (gr.update(),) * 5
    return (_invariant_pick_update(agent, pick), *fields)


def _invariant_form(inv: Invariant):
    return (
        inv.text, inv.scope, inv.event or NO_EVENT, inv.mode or NO_MODE, inv.active,
    )


def on_invariant_editor_load(agent: Agent | None):
    """Загрузка формы редактора — отдельный `demo.load` (после `on_load`,
    чтобы агент вкладки уже был), не событие `_view()`: пункты списка и пустая
    форма нового инварианта."""
    return (_invariant_pick_update(agent, INVARIANT_NEW), *_empty_invariant_form())


def on_invariant_choices_refresh(agent: Agent | None, pick: str | None):
    """Фокус на списке «Инвариант для правки» — пункты пересобираются по
    книге активного агента (у каждого агента свои сессионные инварианты),
    поля формы не трогаются."""
    return _invariant_pick_update(agent, pick)


def on_pick_invariant(agent: Agent | None, pick: str):
    """Список «Инвариант для правки» — `select`: поля формы из **сохранённого**
    инварианта, а не из текущего состояния формы, — несохранённая правка при
    переключении теряется (об этом говорит `info` списка)."""
    if pick == INVARIANT_NEW or agent is None:
        return _empty_invariant_form()
    view = agent.invariant_view
    inv = next((i for i in view.items if i.id == pick), None) if view else None
    if inv is None:
        return (gr.update(),) * 5
    return _invariant_form(inv)


def _change_words(old: Invariant | None, new: Invariant, new_id: str) -> str:
    """Что изменилось, словами для статуса и лога (§8.5): «добавлен С1»,
    «П4 включён», «П3: формулировка, переход автомата»."""
    if old is None:
        return f"добавлен {new_id}"
    bits = []
    if old.scope != new.scope:
        bits.append(
            f"область {SCOPE_LABELS[old.scope]} → {SCOPE_LABELS[new.scope]}, "
            f"номер {old.id} → {new_id}"
        )
    if old.text != new.text:
        bits.append("формулировка")
    if (old.event, old.mode) != (new.event, new.mode):
        bits.append("переход автомата")
    toggled = old.active != new.active
    if toggled and not bits:
        return f"{new_id} {'включён' if new.active else 'выключен'}"
    if toggled:
        bits.append("включён" if new.active else "выключен")
    return f"{new_id}: {', '.join(bits)}"


def _where_saved(agent: Agent, permanent: bool, session: bool) -> str:
    """Куда записалась правка — для статуса: файл постоянных, файл сессии
    или оба."""
    places = []
    if permanent:
        places.append(INVARIANTS.path)
    if session:
        places.append(
            display_path(STORE.path_for(agent.session_id))
            if agent.history
            else "файл сессии появится с первым сообщением"
        )
    return " и ".join(places)


def on_save_invariant(
    agent: Agent | None,
    preset_name: str,
    pick: str,
    text: str,
    scope: str,
    event: str,
    mode: str,
    active: bool,
):
    """«Сохранить инвариант» (§8.5): книга собирается целиком, как профиль
    дня 12, — список меняется в памяти, проверяется `invariants.validate()`, и
    только потом пишется. Постоянные уходят в `INVARIANTS.save()`, сессионные —
    в `agent.set_session_invariants()`; смена области переносит инвариант:
    он удаляется из прежнего места и записывается в новое, номер при этом
    меняется (буква области)."""
    if agent is None:  # страховка на случай сессии без сработавшего load
        agent = _new_agent(preset_name)

    def reply(status: str, pick_value: str | None):
        return (
            agent, gr.update(), *_view(agent, status),
            *_invariant_editor_outputs(agent, pick_value),
        )

    if agent.invariant_view is None:
        return reply("У этого агента нет книги инвариантов — сохранять некуда.", pick)

    permanent, session = _invariant_lists(agent)
    editing = None
    if pick != INVARIANT_NEW:
        editing = next((i for i in permanent + session if i.id == pick), None)
        if editing is None:
            return reply(
                f"Инвариант «{pick}» больше не существует (список изменился "
                f"или это другой агент) — выберите его заново. Ничего не "
                f"записано.",
                INVARIANT_NEW,
            )

    candidate = Invariant(
        id=editing.id if editing else "",
        text=" ".join((text or "").split()),
        scope=scope,
        active=bool(active),
        event="" if event in (NO_EVENT, None) else event,
        mode="" if mode in (NO_MODE, None) else mode,
        added_at=editing.added_at if editing and editing.added_at
        else datetime.now().isoformat(timespec="seconds"),
    )

    new_permanent, new_session = list(permanent), list(session)
    target = new_permanent if scope == SCOPE_ALWAYS else new_session
    if editing is None:
        target.append(candidate)
    elif editing.scope == scope:
        target[next(i for i, inv in enumerate(target) if inv.id == editing.id)] = candidate
    else:
        source = new_permanent if editing.scope == SCOPE_ALWAYS else new_session
        source[:] = [inv for inv in source if inv.id != editing.id]
        target.append(candidate)

    errors = validate(
        _with_ids(new_permanent, SCOPE_ALWAYS) + _with_ids(new_session, SCOPE_SESSION),
        INVARIANT_TEXT_WORDS, INVARIANT_MAX_ITEMS, EVENTS, GUARD_MODES,
    )
    if errors:
        return reply(
            f"Инвариант не сохранён: {'; '.join(errors)}. Действует прежний "
            f"список.",
            pick,
        )

    old_permanent, old_session = dump_invariants(permanent), dump_invariants(session)
    new_permanent_dump = dump_invariants(new_permanent)
    new_session_dump = dump_invariants(new_session)
    permanent_changed = new_permanent_dump != old_permanent
    session_changed = new_session_dump != old_session
    if not permanent_changed and not session_changed:
        return reply("Инвариант не изменился — записывать нечего.", pick)

    position = next(
        i for i, inv in enumerate(new_permanent if scope == SCOPE_ALWAYS else new_session)
        if inv is candidate
    )
    new_id = f"{ID_PREFIX[scope]}{position + 1}"
    words = _change_words(editing, candidate, new_id)
    try:
        # Постоянные — первыми: сбой записи файла оставляет всё как было.
        if permanent_changed:
            INVARIANTS.save(new_permanent_dump, changed=words)
    except StorageError as exc:
        return reply(f"Инвариант не сохранён: {exc}. Действует прежний список.", pick)
    if session_changed:
        agent.set_session_invariants(new_session_dump)

    reach = (
        "Действует со следующего хода у всех агентов, у которых инварианты в "
        "запросе не выключены."
        if permanent_changed and scope == SCOPE_ALWAYS
        else "Действует со следующего хода этой сессии и её веток, у которых "
        "инварианты в запросе не выключены."
    )
    logger.info("инварианты (редактор): %s", words)
    return reply(
        f"Инвариант сохранён: {words} → "
        f"{_where_saved(agent, permanent_changed, session_changed)}. {reach}",
        new_id,
    )


def on_delete_invariant(agent: Agent | None, preset_name: str, pick: str):
    """«Удалить инвариант» (§8.5): убирает выбранный инвариант из книги.
    Удаление — не способ снять ограничение на время: выключить его можно
    флажком «Действует», и это видно."""
    if agent is None:  # страховка на случай сессии без сработавшего load
        agent = _new_agent(preset_name)

    def reply(status: str, pick_value: str | None, clear: bool = False):
        return (
            agent, gr.update(), *_view(agent, status),
            *_invariant_editor_outputs(agent, pick_value, clear_form=clear),
        )

    if agent.invariant_view is None:
        return reply("У этого агента нет книги инвариантов — удалять нечего.", pick)
    if pick == INVARIANT_NEW:
        return reply("Удалять нечего: в списке выбран «+ новый».", pick)

    permanent, session = _invariant_lists(agent)
    doomed = next((i for i in permanent + session if i.id == pick), None)
    if doomed is None:
        return reply(
            f"Инвариант «{pick}» больше не существует — список обновлён, "
            f"ничего не удалено.",
            INVARIANT_NEW,
        )
    if doomed.scope == SCOPE_ALWAYS:
        rest = [inv for inv in permanent if inv.id != doomed.id]
        try:
            INVARIANTS.save(dump_invariants(rest), changed=f"удалён {doomed.id}")
        except StorageError as exc:
            return reply(f"Инвариант не удалён: {exc}. Действует прежний список.", pick)
    else:
        rest = [inv for inv in session if inv.id != doomed.id]
        agent.set_session_invariants(dump_invariants(rest))
    logger.info("инварианты (редактор): удалён %s", doomed.id)
    return reply(
        f"Инвариант {doomed.id} удалён → "
        f"{_where_saved(agent, doomed.scope == SCOPE_ALWAYS, doomed.scope == SCOPE_SESSION)}. "
        f"Остальные номера могли сдвинуться.",
        INVARIANT_NEW,
        clear=True,
    )


# --- Редактор профиля: обработчики (день 12, §7.3) ------------------------
# Единственное исключение из «интерфейс — вид на состояние агента»: это
# форма ввода, а её поля не входят в `_view()`/`VIEW_OUTPUTS` — иначе любое
# событие (отправка вопроса, смена стратегии) затирало бы несохранённую
# правку. У формы свои выходы (`PROFILE_EDITOR_OUTPUTS`, определены при
# сборке интерфейса) и свои обработчики; профиль они читают из `PROFILE`, а
# не из агента — профиль общий для всех агентов процесса.

def on_profile_editor_load():
    """Загрузка формы редактора — отдельный `demo.load`, не событие
    `_view()`: общая часть и первый режим из сохранённого профиля (или
    профиля по умолчанию, если файла ещё нет)."""
    profile, _ = parse_profile(PROFILE.current())
    mode = profile.modes[0] if profile.modes else None
    mode_names = [m.name for m in profile.modes]
    return (
        profile.address,
        profile.language,
        profile.level,
        profile.constraints,
        gr.update(choices=mode_names, value=mode.name if mode else None),
        mode.when if mode else "",
        mode.style if mode else "",
        mode.format if mode else "",
        mode.constraints if mode else "",
        "\n".join(mode.steps) if mode else "",
    )


def on_pick_profile_mode(mode_name: str):
    """Список «Режим для правки» — событие `select` (§7.3): поля режима из
    **сохранённого** профиля, а не из текущего состояния формы —
    несохранённая правка предыдущего режима при переключении теряется (об
    этом говорит `info` списка)."""
    profile, _ = parse_profile(PROFILE.current())
    mode = mode_by_name(profile, mode_name)
    if mode is None:
        return "", "", "", "", ""
    return mode.when, mode.style, mode.format, mode.constraints, "\n".join(mode.steps)


def on_save_profile(
    agent: Agent | None,
    preset_name: str,
    address: str,
    language: str,
    level: str,
    constraints: str,
    mode_name: str,
    when: str,
    style: str,
    format_: str,
    mode_constraints: str,
    steps: str,
):
    """«Сохранить профиль» (§7.3): профиль собирается из `PROFILE.current()`
    — общая часть заменяется полями формы, режим `mode_name` — полями формы,
    остальные режимы берутся как есть. Дальше `parse_profile()` →
    `validate_profile()` → `profile_changes()` против текущего →
    `PROFILE.save()`. Возвращает `COMMON_OUTPUTS`, как и остальные
    обработчики: агент не создаётся и не меняется, но панель («Профиль»,
    файл профиля) должна перерисоваться."""
    if agent is None:  # страховка на случай сессии без сработавшего load
        agent = _new_agent(preset_name)

    old_profile, _ = parse_profile(PROFILE.current())
    data = dump_profile(old_profile)
    data["address"] = address or ""
    data["language"] = language or ""
    data["level"] = level or ""
    data["constraints"] = constraints or ""
    target = mode_by_name(old_profile, mode_name)
    if target is not None:
        for mode_data in data["modes"]:
            if mode_data["name"] == target.name:
                mode_data["when"] = when or ""
                mode_data["style"] = style or ""
                mode_data["format"] = format_ or ""
                mode_data["constraints"] = mode_constraints or ""
                mode_data["steps"] = steps or ""
                break

    new_profile, _ = parse_profile(data)
    errors = validate_profile(new_profile)
    if errors:
        status = f"Профиль не сохранён: {'; '.join(errors)}. Действует прежний профиль."
        return (agent, gr.update(), *_view(agent, status))

    changes = profile_changes(old_profile, new_profile)
    if not changes:
        status = "Профиль не изменился — записывать нечего."
        return (agent, gr.update(), *_view(agent, status))

    try:
        PROFILE.save(dump_profile(new_profile), changed=", ".join(changes))
    except StorageError as exc:
        status = f"Профиль не сохранён: {exc}. Действует прежний профиль."
        return (agent, gr.update(), *_view(agent, status))

    status = (
        f"Профиль сохранён: {', '.join(changes)} → {PROFILE.path}. Действует "
        f"со следующего хода у всех агентов, у которых профиль в запросе не "
        f"выключен."
    )
    return (agent, gr.update(), *_view(agent, status))


# --- Старт процесса ------------------------------------------------------
# Восстановление агентов происходит здесь, при импорте модуля, — один раз на
# процесс и до `demo.launch()`. Кнопки «восстановить» в интерфейсе нет и не
# нужно: к моменту, когда откроется первая страница, агенты уже в реестре.

RESTORED_AT_START = _restore_agents()


# --- Интерфейс -----------------------------------------------------------

# --- Сценарий сравнения стратегий: тексты для gr.Examples (день 10, §7.4) --
# Порядок прогона (§9.2): шаги 1-6, К1-К3, В2, КВ — В1 отдельным пунктом не
# нужен, это шаг 6 дословно. Сами тексты — контент проекта, живут в
# `presets.py` и отправляются на всех прогонах одинаково до буквы; здесь
# только подписи для примеров и порядок их показа.
_SCENARIO_TEXTS: list[str] = [
    *COMPARISON_SCENARIO,
    *CONTROL_QUESTIONS,
    BRANCH_STEPS[1],
    BRANCH_QUESTION,
]
_SCENARIO_LABELS: list[str] = [
    "Шаг 1 · цель и состав",
    "Шаг 2 · коробки и время",
    "Шаг 3 · формат и договорённость",
    "Шаг 4 · герои",
    "Шаг 5 · поправка коробок",
    "Шаг 6 / В1 · тиран Drellen",
    "К1 · контроль: время и состав",
    "К2 · контроль: договорённость",
    "К3 · контроль: поправка коробок",
    "В2 · тиран Nom",
    "КВ · сводный сетап",
]

# --- Сценарий проверки слоёв памяти: тексты для gr.Examples (день 11, §7.5) -
# Порядок прогона (§9.2): С1-С6, К1-К3, Н1 и Н3 — Н2 отдельным пунктом не
# нужен, это К2 дословно (`NEXT_SESSION_QUESTIONS[1] is
# MEMORY_CONTROL_QUESTIONS[1]`). Тексты живут в `presets.py`, здесь только
# подписи и порядок показа.
_MEMORY_SCENARIO_TEXTS: list[str] = [
    *MEMORY_SCENARIO,
    *MEMORY_CONTROL_QUESTIONS,
    NEXT_SESSION_QUESTIONS[0],
    NEXT_SESSION_QUESTIONS[2],
]
_MEMORY_SCENARIO_LABELS: list[str] = [
    "С1 · цель, состав, игровая группа",
    "С2 · коллекция и время",
    "С3 · домашнее правило (формат — теперь профиль, не память)",
    "С4 · герои",
    "С5 · поправка коллекции",
    "С6 · тиран и знание о правилах",
    "К1 · контроль: время и состав",
    "К2 / Н2 · контроль: домашнее правило",
    "К3 · контроль: поправка коллекции",
    "Н1 · новая партия: Drellen и Undertow",
    "Н3 · рабочая не протекла",
]

# --- Сценарий проверки персонализации: тексты для gr.Examples (день 12,
# §7.6, §9.1) — В1 (сравнение профилей и режимов), затем Р1-Р8 (сообщения
# для роутера, порядок прогона П4/П6). Тексты живут в `presets.py`, здесь
# только подписи и порядок показа под чатом.
_PROFILE_SCENARIO_TEXTS: list[str] = [
    PROFILE_QUESTION,
    *ROUTING_SCENARIO,
]
_PROFILE_SCENARIO_LABELS: list[str] = [
    "В1 · бой: кто ходит первым",
    "Р1 · явная ситуация партии",
    "Р2 · продолжение без маркеров",
    "Р3 · явное объяснение новичку",
    "Р4 · продолжение объяснения",
    "Р5 · явный разбор",
    "Р6 · продолжение разбора",
    "Р7 · явная смена задачи",
    "Р8 · граница (после смены профиля на Б)",
]

# --- Сценарий проверки состояния задачи: подписи для gr.Examples (день 13,
# §7.6, §9.1). Тексты живут в `presets.py` (`TASK_SCENARIO`), здесь только
# подписи и порядок показа под чатом.
_TASK_SCENARIO_LABELS: list[str] = [
    "З1 · цель — кнопкой «Начать задачу»",
    "З2 · пауза на планировании",
    "В1 · вопрос на паузе",
    "В2 · вопрос на паузе",
    "В3 · вопрос на паузе (цель и план выпадают из окна)",
    "З3 · продолжение планирования",
    "З4 · план утверждён",
    "Г · шаг выполнен (повторять)",
    "З5 · пауза на выполнении",
    "В4 · вопрос на паузе",
    "В5 · вопрос на паузе",
    "В6 · вопрос на паузе (объяснение шага выпадает из окна)",
    "З6 · продолжение выполнения (после перезапуска / в ветке)",
    "З7 · возврат к несуществующему шагу — отказ",
    "З8 · после кнопок «Пауза»/«Продолжить»",
    "З9 · проверка не пройдена — возврат к шагу",
    "Д · проверка пройдена (повторять)",
]

# --- Сценарий проверки инвариантов: подписи для gr.Examples (день 14, §8.6,
# §10.1). Тексты живут в `presets.py` (`INVARIANT_SCENARIO`), здесь только
# подписи и порядок показа под чатом.
_INVARIANT_SCENARIO_LABELS: list[str] = [
    "Н1 · обычный ход: Trove Tokens (П1 соблюдён без конфликта)",
    "К1 · перевод названий (П1)",
    "К2 · герои из Undertow (П3)",
    "Д1 · просьба забыть ограничения",
    "Н2 · вопрос об ограничении — не конфликт",
    "В1 · посторонний вопрос: длительность партии",
    "В2 · посторонний вопрос: отличия от ролевых настолок",
    "В3 · потерялся кубик (разговор об ограничениях вышел из окна)",
    "К3 · тактика на первый бой (сессионный С1)",
    "Д2 · просьба отключить ограничение на тактику",
    "Т1 · подготовка стола — кнопкой «Начать задачу»",
    "Т2 · план утверждён",
    "Т3 · шаг выполнен (повторять)",
    "Т4 · проверка сходится → ожидание подтверждения (повторять)",
    "К4 · повтор К1 — только в ветке без блока",
]


with gr.Blocks(title="TooManyRules") as demo:
    gr.Markdown(
        "# TooManyRules \n"
        "День 14: у агента появились **инварианты** — ограничения, которые "
        "ассистент не вправе нарушить. Их пишете вы, в редакторе под чатом: "
        "одни действуют всегда (файл рядом с памятью и профилем), другие — "
        "только в этой сессии; а часть можно повесить на переход автомата "
        "задачи — «запрещён» или «с подтверждением». Инварианты лежат "
        "отдельно от диалога и уходят в каждый запрос своим блоком **первым "
        "из блоков** — их не вытесняет окно контекста. Перед ответом идёт "
        "**страж** — служебный вызов, который называет конфликт нового "
        "сообщения с ними; ход при этом не отменяется, отказывает "
        "ассистент, а код только показывает конфликт. Над переходами "
        "автомата нарушение ловит сам код, в остальном инвариант держит "
        "модель — и видно это по её ответам, а не по гарантии. Это не "
        "профиль (он говорит, **как** отвечать), не память (**что** агент "
        "знает) и не состояние задачи (**где** мы в работе): инвариант "
        "говорит, **чего нельзя**. Слои памяти, профиль, задача, стратегии, "
        "checkpoint'ы и ветки работают как раньше. Сценарий проверки "
        "инвариантов — под чатом."
    )

    # Экземпляр агента живёт в состоянии сессии: у каждой открытой вкладки
    # браузера свой агент. Создаётся на `demo.load`, а не здесь, чтобы
    # gr.State не пытался копировать живой объект с http-клиентом внутри.
    agent_state = gr.State(None)

    with gr.Row():
        # --- Слева: чат ---
        with gr.Column(scale=1):
            preset_dropdown = gr.Dropdown(
                choices=[
                    (f"{name} ({config.description})", name)
                    for name, config in PRESETS.items()
                ],
                value=DEFAULT_PRESET,
                label="Пресет агента",
                # Поиск по четырём пунктам не нужен, а поле фильтра добавляет
                # состояние «текст очищен», в котором значение списка не
                # совпадает ни с одним пресетом.
                filterable=False,
                info=(
                    "Смена пресета создаёт нового агента и начинает новый "
                    "диалог: стек сообщений очищается. Предыдущий агент никуда "
                    "не девается — он остаётся в переключателе справа."
                ),
            )
            # Второй список стоит вплотную к первому и означает совсем другое,
            # поэтому разница подписана прямым текстом: пресет — это новый
            # агент и новый диалог, стратегия — тот же агент и тот же диалог,
            # собранный для модели иначе.
            strategy_dropdown = gr.Dropdown(
                choices=[
                    (f"{name} ({description})", name)
                    for name, description in STRATEGIES.items()
                ],
                value=DEFAULT_STRATEGY,
                label="Стратегия контекста",
                filterable=False,
                info=(
                    "Как из истории собирается запрос к модели. Переключение "
                    "НЕ создаёт агента и НЕ трогает стек: диалог остаётся как "
                    "есть, меняется только то, что из него уходит в модель "
                    "начиная со следующего вопроса."
                ),
            )
            # Третий список рядом — снова другая семантика (день 10, §2.2):
            # ветка — не пункт списка стратегий, а отдельная сессия с копией
            # общего начала истории; переключает её этот список, устроенный
            # так же, как переключатель агентов справа.
            branch_dropdown = gr.Dropdown(
                choices=[],
                label="Ветка диалога",
                filterable=False,
                info=(
                    "Какая история лежит в стеке. Пресет создаёт нового "
                    "агента; стратегия меняет запрос у того же агента; ветка "
                    "переключает на другую историю — отдельную сессию со "
                    "своим файлом, своей стратегией и своими счётчиками."
                ),
            )
            # «Слои памяти в запросе» (день 11, §7.2) — сразу под веткой
            # диалога: ещё один переключатель того, что уходит в модель, и
            # снова не того, что агент помнит. Пункты — имена слоёв из
            # `memory.py`, значение подтягивается к агенту в `_view()`.
            layers_group = gr.CheckboxGroup(
                choices=list(REQUEST_LAYERS),
                value=list(REQUEST_LAYERS),
                label="Слои памяти в запросе",
                info=(
                    "Выключенный слой не уходит в модель, но продолжает "
                    "записываться: разбор памяти идёт при любом положении, и "
                    "включённый обратно слой сразу полон. Краткосрочную "
                    "память (разговор) урезает стратегия контекста."
                ),
            )
            # «Инварианты в запросе» (день 14, §8.2) — над «Профилем в
            # запросе», в порядке блоков запроса: инварианты встают в запрос
            # раньше профиля. Ещё один переключатель того, что уходит в
            # модель, и снова не того, что хранится и что делает страж.
            invariants_in_request_checkbox = gr.Checkbox(
                value=True,
                label="Инварианты в запросе",
                info=(
                    "Выключенный блок не уходит в модель, но страж продолжает "
                    "сверять сообщения и показывать конфликты, а ограничения "
                    "переходов продолжают действовать; выключить сам "
                    "инвариант можно флажком «Действует» в редакторе ниже."
                ),
            )
            # «Профиль в запросе» (день 12, §7.2) — сразу под слоями памяти:
            # ещё один переключатель того, что уходит в модель. Пункты —
            # `debug_state()["profile"]["choices"]`, значение подтягивается к
            # агенту в `_view()`.
            profile_dropdown = gr.Dropdown(
                choices=[],
                label="Профиль в запросе",
                filterable=False,
                info=(
                    "Профиль задаёте вы — в блоке «Профиль» справа и в "
                    "редакторе ниже. «авто» — режим выбирает роутер (+1 "
                    "вызов модели на ход); режим — в запрос всегда уходит "
                    "он, роутер не вызывается; «выключен» меняет запрос, а "
                    "не профиль — он никуда не девается."
                ),
            )
            # «Состояние задачи в запросе» (день 13, §7.3) — сразу под
            # «Профилем в запросе»: ещё один переключатель того, что уходит
            # в модель, и снова не того, чем занят автомат.
            task_in_request_checkbox = gr.Checkbox(
                value=True,
                label="Состояние задачи в запросе",
                info=(
                    "Выключенный блок не уходит в модель, но трекер "
                    "продолжает вести задачу, и включённый обратно блок "
                    "сразу актуален. Начинают и завершают задачу кнопки под "
                    "чатом."
                ),
            )

            # Формат значения — список сообщений {"role", "content"}, то есть
            # ровно `agent.history`. Аргумент `type="messages"` из спецификации
            # не передаётся: в Gradio 6 (в проекте 6.26) он удалён, потому что
            # формат сообщений там стал единственным — старые кортежи больше
            # не поддерживаются.
            chatbot = gr.Chatbot(
                label="Диалог (рендерится из стека агента)",
                height=420,
            )
            question_input = gr.Textbox(
                label="Вопрос по правилам",
                placeholder="Например: из каких фаз состоит ход игрока?",
                lines=2,
            )
            with gr.Row():
                send_btn = gr.Button("Отправить", variant="primary", scale=2)
                reset_btn = gr.Button("Сбросить диалог", scale=1)
                # Второй агент с тем же пресетом: сменой пресета такую пару не
                # получить, а она нагляднее всего показывает раздельные стеки.
                new_agent_btn = gr.Button("Новый агент", scale=1)
                delete_agent_btn = gr.Button(
                    "Удалить агент", variant="stop", scale=1
                )
            # Ряд кнопок задачи (день 13, §7.2) — сразу под рядом «Отправить
            # / Сбросить диалог / Новый агент / Удалить агент»: старт задачи
            # идёт от поля ввода, поэтому рядом с ним. Кнопки активны
            # всегда — недопустимое нажатие отклоняет автомат, и в статусе
            # видно почему (это часть проверки, а не недосмотр интерфейса).
            with gr.Row():
                start_task_btn = gr.Button(
                    "Начать задачу с этим сообщением", variant="primary", scale=2
                )
                pause_task_btn = gr.Button("⏸ Пауза", scale=1)
                resume_task_btn = gr.Button("▶ Продолжить", scale=1)
                # «Подтвердить переход» (день 14, §8.3) — переход трекера,
                # который остановил инвариант «с подтверждением», отмечает
                # человек. Активна всегда: нечего подтверждать — отказ, и в
                # статусе видно почему.
                confirm_task_btn = gr.Button("✅ Подтвердить переход", scale=1)
                cancel_task_btn = gr.Button(
                    "Отменить задачу", variant="stop", scale=1
                )
            # Ряд checkpoint'ов и веток (день 10): своя строка под кнопками
            # чата — жест здесь другой, чем у кнопок выше (они не трогают
            # историю, «Ветка от checkpoint'а» заводит новую сессию).
            with gr.Row():
                save_checkpoint_btn = gr.Button("Сохранить checkpoint", scale=1)
                checkpoint_dropdown = gr.Dropdown(
                    choices=[],
                    label="Checkpoint",
                    filterable=False,
                    scale=2,
                    # У списка нет своего обработчика — это вход кнопки
                    # «Ветка от checkpoint'а», и правило «слушать select»
                    # к нему не относится.
                )
                fork_btn = gr.Button("Ветка от checkpoint'а", scale=1)
            # Ряд кандидатов (день 11, §7.2) — под рядом checkpoint'ов. Своего
            # обработчика у группы нет — это вход кнопок, как список
            # «Checkpoint». По умолчанию не отмечено ничего: в долговременную
            # память попадает только то, что человек отметил и сохранил.
            with gr.Row():
                candidates_group = gr.CheckboxGroup(
                    choices=[],
                    value=[],
                    label="Кандидаты в долговременную память",
                    scale=3,
                )
                with gr.Column(scale=1, min_width=160):
                    accept_btn = gr.Button(
                        "Сохранить в долговременную", variant="primary"
                    )
                    reject_btn = gr.Button("Отклонить")
            status_md = gr.Markdown("")

            # Редактор инвариантов (день 14, §8.5) — форма, а не вид, как
            # редактор профиля: её поля не входят в `_view()`/`VIEW_OUTPUTS`,
            # иначе любое событие затирало бы несохранённую правку. Открыт
            # по умолчанию: это инструмент сегодняшнего дня. Стоит под
            # строкой статуса и над закрытым редактором профиля.
            with gr.Accordion(
                "Инварианты — ограничения, которые задаёте вы",
                open=True,
            ):
                gr.Markdown(
                    "Инвариант — правило, которое ассистент не вправе "
                    "нарушить: просьба в разговоре его не отменяет. Пишете "
                    "его вы; модель инварианты не создаёт и не предлагает. "
                    "«Всегда» — постоянный (общий файл, все сессии), «сессия» "
                    "— только этот диалог и его ветки. Инвариант над "
                    "переходом автомата проверяет код: «запрещён» — переход "
                    "не состоится, «с подтверждением» — состоится по кнопке "
                    "«✅ Подтвердить переход»."
                )
                invariant_pick_dropdown = gr.Dropdown(
                    choices=[INVARIANT_NEW],
                    value=INVARIANT_NEW,
                    label="Инвариант для правки",
                    filterable=False,
                    info=(
                        "«+ новый» — добавить. Несохранённая правка при "
                        "переключении теряется; пункты обновляются по книге "
                        "активного агента."
                    ),
                )
                invariant_text_input = gr.Textbox(
                    label="Формулировка",
                    info=f"Не больше {INVARIANT_TEXT_WORDS} слов: правило с исключением в конце.",
                    lines=2,
                )
                with gr.Row():
                    invariant_scope_dropdown = gr.Dropdown(
                        choices=list(SCOPES),
                        value=SCOPE_ALWAYS,
                        label="Область",
                        filterable=False,
                        info="Смена области переносит инвариант; номер при этом меняется.",
                    )
                    invariant_event_dropdown = gr.Dropdown(
                        choices=[NO_EVENT, *EVENTS],
                        value=NO_EVENT,
                        label="Переход автомата",
                        filterable=False,
                    )
                    invariant_mode_dropdown = gr.Dropdown(
                        choices=[NO_MODE, *GUARD_MODES],
                        value=NO_MODE,
                        label="Режим перехода",
                        filterable=False,
                        info="Событие и режим задаются вместе.",
                    )
                invariant_active_checkbox = gr.Checkbox(
                    value=True,
                    label="Действует",
                    info=(
                        "Выключенный инвариант остаётся в файле, но не уходит "
                        "ни в блок, ни стражу, ни в автомат — это способ снять "
                        "ограничение, и он виден."
                    ),
                )
                with gr.Row():
                    save_invariant_btn = gr.Button("Сохранить инвариант", variant="primary")
                    delete_invariant_btn = gr.Button("Удалить инвариант", variant="stop")
                gr.Examples(
                    examples=[
                        [
                            ex["text"], ex["scope"], ex["event"] or NO_EVENT,
                            ex["mode"] or NO_MODE, ex["active"],
                        ]
                        for ex in INVARIANT_EXAMPLES
                    ],
                    inputs=[
                        invariant_text_input,
                        invariant_scope_dropdown,
                        invariant_event_dropdown,
                        invariant_mode_dropdown,
                        invariant_active_checkbox,
                    ],
                    example_labels=[ex["label"] for ex in INVARIANT_EXAMPLES],
                    label="Заготовки (заполняют форму; сохраняет человек)",
                )

            # Редактор профиля (день 12, §7.3) — форма, а не вид: её поля не
            # входят в `_view()`/`VIEW_OUTPUTS`, иначе отправка вопроса или
            # смена стратегии затирали бы несохранённую правку. Закрыт по
            # умолчанию, под строкой статуса и над блоками примеров.
            with gr.Accordion(
                "Профиль пользователя — настройки, которые задаёте вы",
                open=False,
            ):
                gr.Markdown(
                    "Общая часть верна для любой задачи; режимы — под "
                    "конкретную задачу, между ними выбирает роутер или "
                    "переключатель «Профиль в запросе» слева. Модель профиль "
                    "не пишет и не предлагает правок."
                )
                profile_address_input = gr.Textbox(label="Как к вам обращаться")
                profile_language_input = gr.Textbox(label="Язык ответов и названий")
                profile_level_input = gr.Textbox(label="Ваш уровень в игре")
                profile_common_constraints_input = gr.Textbox(
                    label="Ограничения во всех ответах"
                )
                profile_mode_dropdown = gr.Dropdown(
                    choices=[],
                    label="Режим для правки",
                    filterable=False,
                    info="Несохранённая правка режима при переключении теряется.",
                )
                profile_when_input = gr.Textbox(
                    label="Когда этот режим нужен — по этому описанию выбирает роутер"
                )
                profile_style_input = gr.Textbox(label="Стиль")
                profile_format_input = gr.Textbox(label="Формат")
                profile_mode_constraints_input = gr.Textbox(label="Ограничения режима")
                profile_steps_input = gr.Textbox(
                    label="Порядок ответа — по шагу на строку", lines=4,
                )
                save_profile_btn = gr.Button("Сохранить профиль", variant="primary")
                gr.Examples(
                    examples=[
                        [v["address"], v["language"], v["level"], v["constraints"]]
                        for v in PROFILE_VARIANTS
                    ],
                    inputs=[
                        profile_address_input,
                        profile_language_input,
                        profile_level_input,
                        profile_common_constraints_input,
                    ],
                    example_labels=[v["label"] for v in PROFILE_VARIANTS],
                    label="Профили для проверки (заполняет общую часть; сохраняет человек)",
                )

            # Свёрнуто: в кадре дня 14 остаётся только его сценарий,
            # остальные — под аккордеоном (правило «на экране — текущий день»).
            with gr.Accordion("Примеры и сценарии прошлых дней (6, 10-13)", open=False):
                gr.Examples(
                    examples=[
                        ["Из каких фаз состоит ход игрока?"],
                        ["А если в этой фазе выпал Encounter — что тогда?"],
                        ["Может ли гирлок ходить по диагонали?"],
                    ],
                    inputs=[question_input],
                    label=(
                        "Примеры (второй вопрос ссылается на первый — так видно, "
                        "что контекст держит агент)"
                    ),
                )

                # Сценарий сравнения стратегий (день 10, §7.4, §9.1): шаги 1-6,
                # К1-К3, В2 и КВ — В1 не нужен отдельным пунктом, это шаг 6
                # дословно (`BRANCH_STEPS[0] is COMPARISON_SCENARIO[5]`). Клик
                # кладёт текст в поле ввода, отправляет человек — автоматического
                # прогона в проекте нет. `examples_per_page` задан явно: по
                # умолчанию их 10, а пунктов 11.
                gr.Examples(
                    examples=[[text] for text in _SCENARIO_TEXTS],
                    inputs=[question_input],
                    example_labels=_SCENARIO_LABELS,
                    examples_per_page=len(_SCENARIO_TEXTS),
                    label=(
                        "Сценарий сравнения стратегий (шаги 1-6, контрольные "
                        "вопросы К1-К3, вторая ветка В2 и общий вопрос КВ — "
                        "см. docs/TooManyRules — День 10 сравнение стратегий.md)"
                    ),
                )

                # Сценарий проверки слоёв памяти (день 11, §7.5, §9.1): С1-С6 и
                # К1-К3 — первая партия, Н1 и Н3 — новая сессия (Н2 — это К2).
                # Клик кладёт текст в поле ввода, отправляет человек.
                gr.Examples(
                    examples=[[text] for text in _MEMORY_SCENARIO_TEXTS],
                    inputs=[question_input],
                    example_labels=_MEMORY_SCENARIO_LABELS,
                    examples_per_page=len(_MEMORY_SCENARIO_TEXTS),
                    label=(
                        "Сценарий проверки слоёв памяти (первая партия С1-С6 и "
                        "К1-К3, новая сессия Н1-Н3; Н2 — это К2 — см. "
                        "docs/TooManyRules — День 11 проверка слоёв памяти.md)"
                    ),
                )

                # Сценарий проверки персонализации (день 12, §7.6, §9): В1 — один
                # вопрос для сравнения профилей и режимов, Р1-Р8 — сообщения для
                # роутера. Порядок прогона — §9.2 спецификации дня 12.
                gr.Examples(
                    examples=[[text] for text in _PROFILE_SCENARIO_TEXTS],
                    inputs=[question_input],
                    example_labels=_PROFILE_SCENARIO_LABELS,
                    examples_per_page=len(_PROFILE_SCENARIO_TEXTS),
                    label="Сценарий проверки профиля (В1, Р1-Р8 — см. §9 спецификации дня 12)",
                )

                # Сценарий проверки состояния задачи (день 13, §7.6, §9.1): З1-З9,
                # В1-В6, Г и Д (Г и Д повторяются — по строке в списке, отправка
                # несколько раз). Клик кладёт текст в поле ввода; З1 отправляется
                # кнопкой «Начать задачу с этим сообщением», остальные — «Отправить».
                gr.Examples(
                    examples=[[text] for text in TASK_SCENARIO],
                    inputs=[question_input],
                    example_labels=_TASK_SCENARIO_LABELS,
                    examples_per_page=len(TASK_SCENARIO),
                    label=(
                        "Сценарий проверки состояния задачи (З1-З9, В1-В6, Г и Д "
                        "— см. docs/TooManyRules — День 13 проверка состояния "
                        "задачи.md)"
                    ),
                )

            # Сценарий проверки инвариантов (день 14, §8.6, §10.1): Н1-Н2,
            # К1-К4, Д1-Д2, В1-В3, Т1-Т4 (Т3 и Т4 повторяются — по строке в
            # списке, отправка несколько раз). Клик кладёт текст в поле ввода;
            # Т1 отправляется кнопкой «Начать задачу с этим сообщением», К4 —
            # только в ветке с выключенным «Инварианты в запросе», остальные —
            # «Отправить». Развёрнут в кадр.
            gr.Examples(
                examples=[[text] for text in INVARIANT_SCENARIO],
                inputs=[question_input],
                example_labels=_INVARIANT_SCENARIO_LABELS,
                examples_per_page=len(INVARIANT_SCENARIO),
                label=(
                    "Сценарий проверки инвариантов (Н, К, Д, В, Т — см. "
                    "docs/TooManyRules — День 14 проверка инвариантов.md)"
                ),
            )

        # --- Справа: дебаг-панель ---
        with gr.Column(scale=2):
            gr.Markdown("## Дебаг-панель")
            # Свёрнуто: конфиг ко дню 13 не меняется.
            with gr.Accordion("Конфиг агента (AgentConfig)", open=False):
                # Фиксированная высота со скроллом: системный промпт длинный,
                # иначе он выдавливает метрики и счётчики за экран.
                config_json = gr.JSON(
                    label="Конфиг агента (AgentConfig)",
                    height=220,
                )
            metrics_md = gr.Markdown(_metrics_md(None))
            # «Инварианты» (день 14, §8.4) — сразу под «Последним вызовом» и
            # над блоками профиля и памяти, в порядке блоков запроса:
            # инварианты встают в запрос раньше профиля. Развёрнут: это блок
            # сегодняшнего дня.
            invariants_md = gr.Markdown("")
            # Свёрнуто: инструменты дней 11-12.
            with gr.Accordion("Профиль и слои памяти (дни 11-12)", open=False):
                # «Профиль» (день 12, §7.4) — сразу под «Последним вызовом» и над
                # «Слоями памяти»: в порядке блоков запроса профиль встаёт раньше
                # слоёв памяти.
                profile_md = gr.Markdown("")
                # «Слои памяти» (день 11, §7.3) — сразу под «Профилем» и над
                # «Контекстом»: сначала что агент знает, потом что из этого
                # отправляется.
                layers_md = gr.Markdown("")
            # Свёрнуто: ко дню 14 состояние задачи — инструмент прошлого дня.
            # Блок остаётся тем же выходом `_view()`, от сворачивания ничего
            # не меняется (§8.7). Таблица автомата собрана один раз при
            # построении интерфейса, в `_view()` не входит: она статична.
            with gr.Accordion("Состояние задачи (день 13)", open=False):
                # «Состояние задачи» (день 13, §7.4) — в порядке блоков
                # запроса: задача встаёт за рабочей памятью.
                task_md = gr.Markdown("")
                with gr.Accordion("Автомат задачи — таблица переходов", open=False):
                    gr.Markdown(_task_transitions_table_md())
            # Сначала «что отправляем» (день 9), потом «сколько это от окна»
            # (день 8): блок контекста стоит над бюджетом, а сводка — сразу
            # под ним, потому что объясняет числа над собой.
            flow_md = gr.Markdown("")
            # Свёрнуто: инструмент дней 9-10.
            with gr.Accordion("Память стратегии (дни 9-10)", open=False):
                memory_box = gr.Textbox(
                    label="Память стратегии",
                    lines=6,
                    max_lines=14,
                    # Сводку не редактируют: это результат работы модели, и
                    # показывать надо её, а не отредактированную версию.
                    interactive=False,
                    # Кнопка копирования: в Gradio 6 (в проекте 6.26) она задаётся
                    # списком `buttons`, а не флагом `show_copy_button` — тот
                    # удалён, как и `type="messages"` у `gr.Chatbot`.
                    buttons=["copy"],
                )
            # Бюджет контекста — под блоками дня 9: сначала «что отправляем»,
            # потом «сколько это от окна». С метриками последнего вызова выше
            # это один и тот же запрос с двух сторон: там факт после вызова,
            # здесь оценка до него.
            context_md = gr.Markdown("")
            # Свёрнуто: инструмент дня 8.
            with gr.Accordion("Заполнитель контекста (день 8)", open=False):
                with gr.Row():
                    filler_size = gr.Number(
                        value=8,
                        label="Заполнитель, тыс. токенов",
                        precision=0,
                        minimum=0,
                        scale=1,
                    )
                    fill_btn = gr.Button("Набить контекст", scale=1)
                gr.Markdown(
                    "Кнопка кладёт текст-заполнитель в поле вопроса и ничего не "
                    "отправляет. Заполнитель, который **прошёл** в модель, "
                    "остаётся в истории и дорожает с каждым следующим ходом — "
                    "после эксперимента диалог сбрасывается кнопкой «Сбросить "
                    f"диалог». Больше {FILLER_MAX_TOKENS / 1_000_000:.1f} млн "
                    "токенов не набирается: этого уже хватает, чтобы вылезти за "
                    "окно модели."
                )
            # Свёрнуто: ко дню 13 бухгалтерия в кадре не нужна — расход
            # трекера виден в «Последнем вызове».
            with gr.Accordion(
                "Счётчики процесса и таблицы ходов и агентов (дни 8, 10)",
                open=False,
            ):
                totals_md = gr.Markdown("")
                process_md = gr.Markdown(_process_md(process_stats()))
                # «Рост по ходам» — под счётчиками процесса: это про накопление,
                # а не про один вызов.
                gr.Markdown("### Рост по ходам")
                turns_table = gr.Dataframe(
                    value=_turns_table([]),
                    label="Ходы агента (журнал живёт в процессе и на диск не едет)",
                    max_height=260,
                    wrap=True,
                )
                # «Агенты процесса» (день 10, §7.5) — сразу под «Ростом по
                # ходам»: первое место, где агенты сравниваются в одной таблице,
                # день 8 от этого сознательно отказался, день 10 делает — расход
                # токенов между стратегиями сравнивать нужно. Никакой подсветки.
                gr.Markdown("### Агенты процесса")
                agents_table = gr.Dataframe(
                    value=_agents_table(),
                    # Служебный вызов хода, основной вызов которого упал, в журнал
                    # не попадает, а деньги за него потрачены — подпись говорит об
                    # этом прямо (спецификация дня 10, §7.5).
                    label=(
                        "Расход по состоявшимся ходам всех агентов реестра; полные "
                        "числа, со служебными вызовами упавших ходов, — в "
                        "«Накоплено агентом»"
                    ),
                    max_height=260,
                    wrap=True,
                )
            # Аккордеон появляется только тогда, когда модель вернула
            # reasoning_content (пресет «Флагман + thinking»).
            with gr.Accordion(
                "reasoning_content последнего вызова",
                open=False,
                visible=False,
            ) as reasoning_accordion:
                reasoning_md = gr.Markdown("")
            # «Ветки диалога» (день 10, §7.3) — над переключателем агентов и
            # стеком сообщений: ветка — это про то, какая история лежит в
            # стеке, и этот блок готовит к чтению того, что ниже.
            branches_md = gr.Markdown("")
            # Переключатель стоит вплотную к стеку сообщений: смысл именно
            # в том, что стек — не «стек приложения», а стек конкретного
            # инстанса, и при переключении он меняется целиком.
            agent_dropdown = gr.Dropdown(
                choices=[],
                label="Агент процесса",
                filterable=False,   # список служебный, фильтр в нём только мешает
                info=(
                    "Агенты, поднятые с момента старта приложения и ещё не "
                    "удалённые. Выбор делает агента активным: переключаются "
                    "чат, стек, конфиг и метрики, и следующий вопрос уходит "
                    "ему."
                ),
            )
            # Свёрнуто: ко дню 13 в кадре нужен файл сессии (`context.task`),
            # а не стек целиком.
            with gr.Accordion("Стек сообщений агента", open=False):
                # Индексы включены, чтобы рост стека на 2 сообщения за ход был
                # виден сразу; выше max_height список скроллится, а не растягивает
                # страницу. Подпись блока приходит из `_view()` — в ней номер
                # агента, чьи сообщения показаны.
                messages_json = gr.JSON(
                    label="Стек сообщений агента (растёт на 2 за ход)",
                    show_indices=True,
                    max_height=420,
                )
            # Два блока дня 7 стоят сразу под стеком сообщений, чтобы «в
            # памяти» и «на диске» были в кадре рядом: одна и та же переписка,
            # показанная с двух сторон.
            storage_md = gr.Markdown("")
            session_file_json = gr.JSON(
                label="Файл сессии на диске",
                max_height=420,
            )
            # Свёрнуто: эти файлы в кадре дня 14 открываются одним кликом.
            with gr.Accordion(
                "Файлы долговременной памяти, профиля и инвариантов (дни 11-12, 14)",
                open=False,
            ):
                # Файл долговременной памяти (день 11, §7.4) — рядом с файлом
                # сессии: рабочая память лежит в том, что выше (`context.working`),
                # долговременная — в этом. Два файла рядом и есть «хранятся
                # отдельно».
                long_term_file_json = gr.JSON(
                    label="Файл долговременной памяти на диске",
                    max_height=320,
                )
                # Файл профиля (день 12, §7.4) — рядом с файлами сессии и
                # долговременной памяти: три файла рядом и есть «профиль, пресет
                # и память — три разные вещи».
                profile_file_json = gr.JSON(
                    label="Файл профиля на диске",
                    max_height=320,
                )
                # Файл инвариантов (день 14, §8.6) — рядом с тремя другими:
                # четыре файла рядом и есть «инварианты хранятся отдельно от
                # диалога». Сессионные — в файле сессии (`context.invariants`).
                invariants_file_json = gr.JSON(
                    label="Файл инвариантов на диске",
                    max_height=320,
                )

    # Порядок выходов совпадает с порядком значений в `_view()`. Значения
    # дня 10, дня 11, дня 12, дня 13 и дня 14 — в конце списка, как и в
    # кортеже `_view()` (§7.7, §8.8).
    VIEW_OUTPUTS = [
        chatbot,
        status_md,
        config_json,
        metrics_md,
        totals_md,
        process_md,
        reasoning_accordion,
        reasoning_md,
        messages_json,
        agent_dropdown,
        preset_dropdown,
        storage_md,
        session_file_json,
        context_md,
        turns_table,
        strategy_dropdown,
        flow_md,
        memory_box,
        branch_dropdown,
        checkpoint_dropdown,
        branches_md,
        agents_table,
        layers_group,
        candidates_group,
        layers_md,
        long_term_file_json,
        profile_dropdown,
        profile_md,
        profile_file_json,
        task_in_request_checkbox,
        task_md,
        invariants_in_request_checkbox,
        invariants_md,
        invariants_file_json,
    ]
    COMMON_OUTPUTS = [agent_state, question_input] + VIEW_OUTPUTS
    # Выходы формы редактора профиля (день 12, §7.3) — отдельно от
    # `VIEW_OUTPUTS`: это не вид на состояние агента, а поля ввода, которые
    # `_view()` не должен трогать.
    PROFILE_EDITOR_OUTPUTS = [
        profile_address_input,
        profile_language_input,
        profile_level_input,
        profile_common_constraints_input,
        profile_mode_dropdown,
        profile_when_input,
        profile_style_input,
        profile_format_input,
        profile_mode_constraints_input,
        profile_steps_input,
    ]
    # Выходы формы редактора инвариантов (день 14, §8.5) — тем же правилом:
    # отдельно от `VIEW_OUTPUTS`, это поля ввода, которые `_view()` не
    # должен трогать.
    INVARIANT_EDITOR_OUTPUTS = [
        invariant_pick_dropdown,
        invariant_text_input,
        invariant_scope_dropdown,
        invariant_event_dropdown,
        invariant_mode_dropdown,
        invariant_active_checkbox,
    ]

    load_event = demo.load(
        on_load,
        inputs=[preset_dropdown],
        outputs=COMMON_OUTPUTS,
    )
    # Оба списка слушают `select` — событие выбора пункта человеком.
    # Ни `change`, ни `input` здесь не годятся: `change` срабатывает и на
    # программное обновление значения (а `_view()` синхронизирует оба списка
    # на каждом событии), `input` в Gradio 6.26 приходит на один выбор дважды.
    # И то и другое рождало лишних агентов.
    preset_dropdown.select(
        on_preset_change,
        inputs=[preset_dropdown],
        outputs=COMMON_OUTPUTS,
    )
    # Третий список — с той же оговоркой про `select`: `_view()` синхронизирует
    # и его значение на каждом событии, так что `change` срабатывал бы на
    # программное обновление и переключал стратегию сам по себе.
    strategy_dropdown.select(
        on_strategy_change,
        inputs=[agent_state, strategy_dropdown, preset_dropdown],
        outputs=COMMON_OUTPUTS,
    )
    agent_dropdown.select(
        on_switch_agent,
        inputs=[agent_dropdown, preset_dropdown],
        outputs=COMMON_OUTPUTS,
    )
    # «Ветка диалога» — тоже `select`, тем же правилом: `_view()`
    # синхронизирует его значение на каждом событии.
    branch_dropdown.select(
        on_switch_branch,
        inputs=[branch_dropdown, preset_dropdown],
        outputs=COMMON_OUTPUTS,
    )
    new_agent_btn.click(
        on_new_agent,
        inputs=[preset_dropdown],
        outputs=COMMON_OUTPUTS,
    )
    delete_agent_btn.click(
        on_delete_agent,
        inputs=[agent_state, preset_dropdown],
        outputs=COMMON_OUTPUTS,
    )
    send_btn.click(
        on_send,
        inputs=[agent_state, question_input, preset_dropdown],
        outputs=COMMON_OUTPUTS,
    )
    question_input.submit(
        on_send,
        inputs=[agent_state, question_input, preset_dropdown],
        outputs=COMMON_OUTPUTS,
    )
    reset_btn.click(
        on_reset,
        inputs=[agent_state, preset_dropdown],
        outputs=COMMON_OUTPUTS,
    )
    save_checkpoint_btn.click(
        on_save_checkpoint,
        inputs=[agent_state, preset_dropdown],
        outputs=COMMON_OUTPUTS,
    )
    # «Checkpoint» — вход этой кнопки, у списка своего обработчика нет.
    fork_btn.click(
        on_fork,
        inputs=[agent_state, checkpoint_dropdown, preset_dropdown],
        outputs=COMMON_OUTPUTS,
    )
    # «Слои памяти в запросе» — только на действие человека: правило
    # выпадающих списков дня 6 в силе, `change` срабатывал бы и на обновление
    # значения из `_view()` и переключал бы слои сам по себе. У
    # `gr.CheckboxGroup` в Gradio 6.26 это `input`, а не `select`, как у
    # списков: проверено 16.09.2026 по сети и по логу — на клик по пункту
    # приходит ровно один запрос с уже обновлённым значением и одна строка
    # «слои в запросе» в логе, а на обновление значения из `_view()` (другие
    # кнопки) `input` не приходит вовсе. Двойного срабатывания, как у
    # `Dropdown.input` на дне 6, у группы флажков нет.
    layers_group.input(
        on_layers_change,
        inputs=[agent_state, layers_group, preset_dropdown],
        outputs=COMMON_OUTPUTS,
    )
    # Группа кандидатов — вход этих двух кнопок, своего обработчика у неё нет.
    accept_btn.click(
        on_accept_candidates,
        inputs=[agent_state, candidates_group, preset_dropdown],
        outputs=COMMON_OUTPUTS,
    )
    reject_btn.click(
        on_reject_candidates,
        inputs=[agent_state, candidates_group, preset_dropdown],
        outputs=COMMON_OUTPUTS,
    )
    # Единственный обработчик, который сам знает содержимое поля ввода и
    # передаёт его в `_view()`: бюджет и полоса должны показать, что
    # произойдёт при отправке, ещё до нажатия «Отправить».
    fill_btn.click(
        on_fill_context,
        inputs=[agent_state, filler_size, preset_dropdown],
        outputs=COMMON_OUTPUTS,
    )
    # «Профиль в запросе» — тем же правилом, что остальные выпадающие списки
    # (день 6): `select`, а не `change`/`input` — `_view()` синхронизирует
    # его значение на каждом событии.
    profile_dropdown.select(
        on_profile_choice,
        inputs=[agent_state, profile_dropdown, preset_dropdown],
        outputs=COMMON_OUTPUTS,
    )

    # Кнопки задачи (день 13, §7.2) — активны всегда, недопустимое нажатие
    # отклоняет автомат. «Начать задачу» знает содержимое поля ввода — цель;
    # остальные три кнопки — один обработчик на всех, событие передаёт
    # `functools.partial`.
    start_task_btn.click(
        on_start_task,
        inputs=[agent_state, question_input, preset_dropdown],
        outputs=COMMON_OUTPUTS,
    )
    pause_task_btn.click(
        functools.partial(on_task_event, event=EVENT_PAUSE),
        inputs=[agent_state, preset_dropdown],
        outputs=COMMON_OUTPUTS,
    )
    resume_task_btn.click(
        functools.partial(on_task_event, event=EVENT_RESUME),
        inputs=[agent_state, preset_dropdown],
        outputs=COMMON_OUTPUTS,
    )
    cancel_task_btn.click(
        functools.partial(on_task_event, event=EVENT_CANCEL),
        inputs=[agent_state, preset_dropdown],
        outputs=COMMON_OUTPUTS,
    )
    # «Состояние задачи в запросе» — `input`, тем же правилом и по тому же
    # замеру, что «Слои памяти в запросе» (`gr.Checkbox` ведёт себя как
    # `gr.CheckboxGroup` в Gradio 6.26): один запрос на клик, не срабатывает
    # на программное обновление из `_view()`.
    task_in_request_checkbox.input(
        on_task_in_request,
        inputs=[agent_state, task_in_request_checkbox, preset_dropdown],
        outputs=COMMON_OUTPUTS,
    )
    # «Подтвердить переход» (день 14, §8.3) — как остальные кнопки задачи:
    # активна всегда, нечего подтверждать — отказ со словами в статусе.
    confirm_task_btn.click(
        on_confirm_transition,
        inputs=[agent_state, preset_dropdown],
        outputs=COMMON_OUTPUTS,
    )
    # «Инварианты в запросе» — `input`, тем же правилом, что «Слои памяти в
    # запросе» и «Состояние задачи в запросе» (`gr.Checkbox` в Gradio 6.26
    # ведёт себя как `gr.CheckboxGroup`): один запрос на клик, на обновление
    # значения из `_view()` не приходит. Проверено по сети и по логу — см.
    # комментарий ниже.
    invariants_in_request_checkbox.input(
        on_invariants_in_request,
        inputs=[agent_state, invariants_in_request_checkbox, preset_dropdown],
        outputs=COMMON_OUTPUTS,
    )

    # Редактор профиля (день 12, §7.3) — форма со своими выходами и своими
    # обработчиками, отдельными от `_view()`/`COMMON_OUTPUTS`.
    demo.load(
        on_profile_editor_load,
        inputs=[],
        outputs=PROFILE_EDITOR_OUTPUTS,
    )
    # «Режим для правки» — тоже `select`: список показывает поля
    # сохранённого режима, а не то, что человек ещё не сохранил.
    profile_mode_dropdown.select(
        on_pick_profile_mode,
        inputs=[profile_mode_dropdown],
        outputs=[
            profile_when_input,
            profile_style_input,
            profile_format_input,
            profile_mode_constraints_input,
            profile_steps_input,
        ],
    )
    save_profile_btn.click(
        on_save_profile,
        inputs=[
            agent_state,
            preset_dropdown,
            profile_address_input,
            profile_language_input,
            profile_level_input,
            profile_common_constraints_input,
            profile_mode_dropdown,
            profile_when_input,
            profile_style_input,
            profile_format_input,
            profile_mode_constraints_input,
            profile_steps_input,
        ],
        outputs=COMMON_OUTPUTS,
    )

    # Редактор инвариантов (день 14, §8.5) — форма со своими выходами и своими
    # обработчиками, отдельными от `_view()`/`COMMON_OUTPUTS`. Загрузка формы —
    # следом за `on_load`: агент вкладки к этому моменту уже есть, а пункты
    # списка зависят от его сессионных инвариантов.
    load_event.then(
        on_invariant_editor_load,
        inputs=[agent_state],
        outputs=INVARIANT_EDITOR_OUTPUTS,
    )
    # «Инвариант для правки» — `select`, как «Режим для правки»: список
    # показывает поля сохранённого инварианта, а не то, что человек ещё не
    # сохранил.
    invariant_pick_dropdown.select(
        on_pick_invariant,
        inputs=[agent_state, invariant_pick_dropdown],
        outputs=[
            invariant_text_input,
            invariant_scope_dropdown,
            invariant_event_dropdown,
            invariant_mode_dropdown,
            invariant_active_checkbox,
        ],
    )
    # Фокус на списке пересобирает пункты по книге активного агента: агент
    # мог смениться, а форма о нём не знает. Поля формы не трогаются.
    invariant_pick_dropdown.focus(
        on_invariant_choices_refresh,
        inputs=[agent_state, invariant_pick_dropdown],
        outputs=[invariant_pick_dropdown],
    )
    # Сохранение и удаление возвращают `COMMON_OUTPUTS + INVARIANT_EDITOR_OUTPUTS`:
    # панель должна перерисоваться, а список инвариантов — обновиться.
    save_invariant_btn.click(
        on_save_invariant,
        inputs=[
            agent_state,
            preset_dropdown,
            invariant_pick_dropdown,
            invariant_text_input,
            invariant_scope_dropdown,
            invariant_event_dropdown,
            invariant_mode_dropdown,
            invariant_active_checkbox,
        ],
        outputs=COMMON_OUTPUTS + INVARIANT_EDITOR_OUTPUTS,
    )
    delete_invariant_btn.click(
        on_delete_invariant,
        inputs=[agent_state, preset_dropdown, invariant_pick_dropdown],
        outputs=COMMON_OUTPUTS + INVARIANT_EDITOR_OUTPUTS,
    )


if __name__ == "__main__":
    demo.launch()
