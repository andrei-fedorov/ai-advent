# TooManyRules — приложение недели 2 (день 9): чат через агента + дебаг-панель.
#
# Здесь только интерфейс. LLM-логики в этом файле нет: ни клиента OpenAI,
# ни chat.completions.create — всё общение с моделью инкапсулировано в
# `agent.Agent`, конфиги агентов проекта лежат в `presets.py`, история
# диалогов — в `storage.py`, счёт токенов — в `tokens.py` (оттуда берутся
# только чистые функции: генератор заполнителя и оценка его размера, всё
# остальное панель получает от агента готовым), стратегии управления
# контекстом — в `context.py`, и напрямую он отсюда тоже не импортируется:
# набор стратегий агенту выдаёт `presets.make_strategies()`.
#
# Панель не знает, какие бывают стратегии и что такое сводка: она рисует то,
# что вернули `debug_state()` и `ContextView` — имя, описание словами, текст
# памяти и числа. День 10 добавит стратегий, не тронув здесь ни строчки.
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
# Вкладок больше нет: дни недели 2 наращивают одну и ту же сущность, и
# «что нового сегодня» показывает дебаг-панель. Дни 1-5 живут в
# замороженном артефакте `app_week1.py` (запуск: ./run.sh week1).
#
# Запуск: ./run.sh (или python app.py из src/ с активированным .venv)

import logging
from dataclasses import asdict

import gradio as gr
import pandas as pd

from agent import (
    Agent,
    agent_by_number,
    agents,
    delete_agent,
    estimate_cost_usd,
    process_stats,
)
from presets import (
    DEFAULT_PRESET,
    DEFAULT_STRATEGY,
    PRESETS,
    STRATEGIES,
    make_strategies,
)
from storage import JsonHistoryStore, StorageError, display_path
from tokens import FILLER_MAX_TOKENS, estimate_tokens, filler_text

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


def _times_word(count: int) -> str:
    """«1 раз» / «2 раза» / «5 раз»: строку в панели читает человек."""
    if 11 <= count % 100 <= 14:
        return "раз"
    return "раза" if count % 10 in (2, 3, 4) else "раз"


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
            f"#{agent.number} · {agent.config.name} · {agent.session_id} · "
            f"{len(agent.history)} сообщ.{suffix}",
            agent.number,
        )
        for agent, suffix in entries
    ]


def _metrics_md(last_call: dict | None) -> str:
    """Метрики последнего вызова агента."""
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
        # Свёртка перед упавшим вызовом оплачена и уже применена, а строки
        # журнала у несостоявшегося хода нет: не показать её здесь — значит
        # не показать нигде, кроме накопленных счётчиков.
        service = _service_lines(last_call.get("service_call"))
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
    return "\n".join(lines)


def _service_lines(service: dict | None) -> list[str]:
    """Служебный вызов этого хода — отдельной строкой в «Последнем вызове».

    `None` означает, что стратегия свёртки не просила. Сбой свёртки
    показывается здесь же и за ошибку ответа не выдаётся: ход состоялся,
    ответ игроку пришёл.
    """
    if not service:
        return []
    if not service["ok"]:
        return [
            f"- **🧵 Свёртка не удалась:** {service['error']} — память "
            f"стратегии не сдвинулась, ход состоялся, в запрос ушло всё "
            f"несвёрнутое. Потрачено "
            f"{_fmt_int(service['total_tokens'] or 0)} токенов."
        ]
    line = (
        f"- **🧵 Свёртка:** {service['covers']} сообщ. "
        f"(≈{_fmt_int(service['folded_tokens'])} токенов) → сводка "
        f"≈{_fmt_int(estimate_tokens(service['text']))} токенов; служебный "
        f"вызов {_fmt_int(service['total_tokens'])} токенов, "
        f"{_fmt_cost(service['cost_usd'])}, {service['elapsed']:.2f} s"
    )
    if service.get("finish_reason") == "length":
        # Обрезанная сводка применена, но выдавать её за нормальную нельзя:
        # штатно в потолок служебного вызова упираться не должно.
        line += (
            " — ⚠️ **сводка упёрлась в потолок `max_tokens` и оборвана на "
            "полуслове**, применена как есть"
        )
    return [line]


_LEVEL_MARKERS = {"ok": "🟢", "warn": "🟡", "danger": "🔴", "over": "🔴"}

# Тексты предупреждений не называют стратегий: панель не знает, какие они
# бывают, и новые стратегии не должны требовать правок здесь.
_LEVEL_WARNINGS = {
    "warn": (
        "⚠️ **Занято больше 70% окна.** Если активная стратегия историю не "
        "сжимает, дальше будет только быстрее."
    ),
    "danger": (
        "🔴 **Занято больше 90% окна.** Ещё пара ходов — и запрос перестанет "
        "влезать. Сжатие само не включается: стратегия контекста "
        "переключается слева."
    ),
    "over": (
        "🔴 **Оценка превышает окно модели.** Запрос всё равно будет отправлен "
        "и, скорее всего, отклонён целиком: ответа не будет, ход не "
        "состоится, стек сообщений и файл сессии останутся как есть. "
        "Предохранителя здесь нет намеренно: сжатие — способ до переполнения "
        "не доходить, а не проверка перед вызовом."
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
    порядке, что **отправленная** часть истории: сообщения, уехавшие в сводку,
    в этом ряду не участвуют."""
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
        # Память стратегии (день 9) — слагаемое наравне с остальными: без неё
        # сумма в строке не сходилась бы с итогом ровно на размер сводки.
        f"- **В запросе:** система {_fmt_int(request['system'])} + память "
        f"{_fmt_int(request['memory'])} + история "
        f"{_fmt_int(request['history'])} ({len(request['per_message'])} сообщ.) + "
        f"вопрос {_fmt_int(request['question'])} + служебные "
        f"{_fmt_int(request['overhead'])} ≈ **{_fmt_int(request['total'])}**",
        # Отдельная корзина, а не история: что это за память и куда встаёт.
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
        f"- **Состав запроса:** система {_fmt_int(request['system'])} + "
        f"{memory_name} {_fmt_int(request['memory'])} + "
        f"{view['sent_messages']} сообщ. истории {_fmt_int(request['history'])} + "
        f"вопрос {_fmt_int(request['question'])} + служебные "
        f"{_fmt_int(request['overhead'])} ≈ **{_fmt_int(estimated)}**",
    ]

    if view["sent_messages"] >= view["history_messages"] and not request["memory"]:
        # Сюда попадает и «Вся история», и «Сводка» до первой свёртки: в обоих
        # случаях в модель уходит весь стек, и врать про экономию нечего.
        lines.append(
            f"- **За бортом:** ничего — все {view['history_messages']} сообщ. "
            f"стека ушли в запрос целиком, экономии нет"
        )
    else:
        lines.append(
            f"- **За бортом:** свёрнуто {state['covered']} сообщ. из "
            f"{view['history_messages']}, отправлено {view['sent_messages']}"
        )
        if saved >= 0:
            share = f"{estimated / full_total * 100:.0f}%" if full_total else "н/д"
            lines.append(
                f"- **Экономия:** ≈{_fmt_int(estimated)} вместо "
                f"≈{_fmt_int(full_total)} — {share} полного контекста, "
                f"сэкономлено ≈{_fmt_int(saved)} токенов "
                f"(≈{_fmt_cost(estimate_cost_usd(model, saved, 0))} по входу)"
            )
        else:
            # Так бывает: сводка длиннее того, что она заменила. Показываем
            # честно — это и есть цена приёма на коротком диалоге.
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
        f"входу), потрачено на свёртки {_fmt_int(totals['service_tokens'])} "
        f"токенов ({_fmt_cost(totals['service_cost_usd'])}) за "
        f"{totals['service_calls']} {_service_word(totals['service_calls'])}"
    )
    if view["pending_label"]:
        lines.append(f"- ⏳ **На следующем ходе:** {view['pending_label']}")
    else:
        lines.append(f"- **Дальше:** {state['note']}")
    lines += [
        "",
        "_Сжимается запрос, а не память: стек сообщений и файл сессии всегда "
        "полные. Переключение стратегии диалог не трогает — один и тот же "
        "разговор можно отправить двумя способами и сравнить ответы._",
    ]
    return "\n".join(lines)


def _service_word(count: int) -> str:
    """«1 свёртку» / «2 свёртки» / «5 свёрток»."""
    if 11 <= count % 100 <= 14:
        return "свёрток"
    match count % 10:
        case 1:
            return "свёртку"
        case 2 | 3 | 4:
            return "свёртки"
        case _:
            return "свёрток"


def _memory_update(view: dict) -> dict:
    """Блок «Память стратегии»: сама сводка текстом — то самое «храните summary
    отдельно», которое должно быть видно.

    Подпись и содержимое приходят из `StrategyState`: панель не знает, что
    внутри — сводка, факты или ничего. Памяти нет — в поле стоит объяснение
    почему, а не пустота.
    """
    state = view["state"]
    text = state["memory_text"]
    if not text:
        return gr.update(
            value=state["note"],
            label=f"{state['memory_label']}: пусто",
        )
    return gr.update(
        value=text,
        label=(
            f"{state['memory_label']}: покрывает {state['covered']} сообщ. из "
            f"{view['history_messages']}, обновлялась {state['updated_turns']} "
            f"{_times_word(state['updated_turns'])}, "
            f"≈{estimate_tokens(text)} токенов"
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
            # «без сжатия» и «служебные» рядом отвечают на главный вопрос дня:
            # экономия минус плата за свёртку.
            "стратегия": turn["strategy"],
            "отправлено сообщ.": turn["sent_messages"],
            "без сжатия": turn["full_prompt_tokens"],
            "служебные": turn["service_tokens"],
        }
        for turn in turns
    ]
    return pd.DataFrame(
        rows,
        columns=["ход", "стек до хода", "оценка", "prompt", "completion",
                 "total", "оценка/факт", "стоимость", "накопительно",
                 "стратегия", "отправлено сообщ.", "без сжатия", "служебные"],
    )


def _totals_md(agent_title: str, totals: dict, model: str) -> str:
    """Накопленное агентом за время жизни. «Сбросить диалог» эти счётчики
    не обнуляет — сброшен диалог, а не агент."""
    lifetime_saved = totals["saved_tokens"]
    verdict = "сэкономлено" if lifetime_saved >= 0 else "переплачено"
    return (
        f"### Накоплено агентом {agent_title}\n\n"
        f"- **Вызовов:** {totals['calls']} (из них с ошибкой: "
        f"{totals['errors']}, служебных: {totals['service_calls']})\n"
        f"- **🔢 Токены:** prompt={totals['prompt_tokens']} / "
        f"completion={totals['completion_tokens']} / "
        f"total={totals['total_tokens']}\n"
        f"- **🧵 Свёртки:** {_fmt_int(totals['service_tokens'])} токенов, "
        f"{_fmt_cost(totals['service_cost_usd'])} — это цена сжатия; "
        f"на запросах при этом {verdict} "
        f"≈{_fmt_int(abs(lifetime_saved))} токенов "
        f"({_fmt_cost(estimate_cost_usd(model, abs(lifetime_saved), 0))} "
        f"по входу)\n"
        f"- **💲 Стоимость:** {_fmt_cost(totals['cost_usd'])}"
    )


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


def _view(agent: Agent, status: str, question: str = "") -> tuple:
    """Полный вид на состояние агента — фиксированный кортеж из 18 значений,
    позиционно раскладывающийся в `VIEW_OUTPUTS`. Порядок — часть контракта
    обработчиков ниже.

    Значения всех трёх выпадающих списков — тоже часть вида: иначе после
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
    # автообновления, как и на дне 6, здесь нет.
    session_file = STORE.read_file(state["session_id"])
    return (
        # 1. чат = стек агента; номер в подписи — чей именно диалог показан
        gr.update(
            value=messages,
            label=f"Диалог агента {title} (рендерится из стека агента)",
        ),
        status,                                  # 2. строка статуса
        state["config"],                         # 3. конфиг агента
        _metrics_md(last_call),                  # 4. метрики последнего вызова
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
        Agent(
            PRESETS[preset_name],
            session_id=info.session_id,
            store=STORE,
            strategies=make_strategies(),
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
        return _spawn(preset_name, "Страница открыта.")

    target = registry[-1]
    if RESTORED_AT_START:
        status = (
            f"Восстановлено {RESTORED_AT_START} "
            f"{_agents_word(RESTORED_AT_START)} из {display_path(STORE.data_dir)}. "
            f"Активен {_agent_title(target)} · `{target.session_id}`: "
            f"{target.restored_messages} сообщ. из файла."
        )
    else:
        status = (
            f"Страница открыта. Активен {_agent_title(target)} · "
            f"`{target.session_id}`: в стеке {len(target.history)} сообщ."
        )
    return (target, gr.update(), *_view(target, status))


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
    remaining = agents()
    if not remaining:
        return _spawn(
            preset_name, f"Агент {title} удалён, в реестре не осталось никого."
        )

    target = remaining[-1]
    return (
        target,
        gr.update(),
        *_view(
            target,
            f"Агент {title} удалён из процесса. Активен "
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
    обработчик дня 9.

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
        # Свёртка назрела — типичный случай, когда на сжатие переключаются
        # после нескольких ходов без него. Следующий запрос уйдёт уже после
        # свёртки, и «все сообщения стека» были бы неправдой.
        label = view.pending_label[:1].upper() + view.pending_label[1:]
        status = (
            f"Стратегия «{agent.strategy.name}». ⏳ {label} перед следующим "
            f"ответом, и в запрос уйдут {state.memory_label.lower()} и "
            f"{view.sent_messages - view.pending_messages} несвёрнутых "
            f"сообщений из {view.history_messages}."
        )
    elif state.memory_text:
        status = (
            f"Стратегия «{agent.strategy.name}»: в запрос уйдут "
            f"{state.memory_label.lower()} и {view.sent_messages} несвёрнутых "
            f"сообщений из {view.history_messages}."
        )
    else:
        # Сокращение «сообщ.» уже кончается точкой, поэтому фразу закрываем
        # так, чтобы в статусе не выросло две подряд.
        status = (
            f"Стратегия «{agent.strategy.name}»: в запрос уйдут все "
            f"{view.sent_messages} сообщ. стека, памяти у стратегии пока нет."
        )
    return (
        agent,
        gr.update(),
        *_view(agent, f"{status} Стек не тронут, история на диске полная."),
    )


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
    if reply.ok:
        status = (
            f"Ответ получен за {reply.elapsed:.2f} s. "
            f"В стеке агента {_agent_title(agent)} {len(agent.history)} сообщ."
        )
        # Ход, на котором произошла свёртка, говорит об этом вслух: на видео
        # момент должен читаться без панели.
        service = reply.service_call
        if service is not None and service.ok:
            status += (
                f" 🧵 Перед ответом сработала свёртка: {service.covers} сообщ. "
                f"(≈{_fmt_int(service.folded_tokens)} токенов) уехали в "
                f"сводку ≈{_fmt_int(estimate_tokens(service.text))} токенов "
                f"за {_fmt_cost(service.cost_usd)}. Сам диалог цел — "
                f"сжался запрос, а не память."
            )
            if service.finish_reason == "length":
                status += (
                    " ⚠️ Сводка упёрлась в потолок служебного вызова и оборвана "
                    "на полуслове — применена как есть."
                )
        elif service is not None:
            status += (
                f" ⚠️ Свёртка не удалась ({service.error}) — ход состоялся, "
                f"память стратегии не сдвинулась, в модель ушло всё "
                f"несвёрнутое; свёртка повторится на следующем ходе."
            )
        # Поле ввода чистим только при успехе; при ошибке вопрос остаётся
        # в поле, чтобы его можно было отправить повторно.
        message_update = ""
    else:
        status = f"❌ {reply.error}"
        service = reply.service_call
        if service is not None and service.ok:
            # Свёртка до упавшего вызова оплачена и уже применена: следующий
            # вопрос уйдёт со сводкой, и это не должно стать сюрпризом.
            status += (
                f" 🧵 Свёртка перед вызовом при этом состоялась: "
                f"{service.covers} сообщ. уехали в сводку за "
                f"{_fmt_cost(service.cost_usd)} — она сохранена и уйдёт со "
                f"следующим вопросом."
            )
        message_update = gr.update()

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
    сохраняются."""
    if agent is None:  # страховка на случай сессии без сработавшего load
        agent = _new_agent(preset_name)
        note = "агент поднят заново"
    else:
        agent.reset()
        note = f"файл сессии `{agent.session_id}` удалён"
    return (
        agent,
        gr.update(),
        *_view(
            agent,
            f"Диалог агента {_agent_title(agent)} сброшен: стек сообщений пуст, "
            f"{note}, счётчики агента сохранены.",
        ),
    )


# --- Старт процесса ------------------------------------------------------
# Восстановление агентов происходит здесь, при импорте модуля, — один раз на
# процесс и до `demo.launch()`. Кнопки «восстановить» в интерфейсе нет и не
# нужно: к моменту, когда откроется первая страница, агенты уже в реестре.

RESTORED_AT_START = _restore_agents()


# --- Интерфейс -----------------------------------------------------------

with gr.Blocks(title="TooManyRules") as demo:
    gr.Markdown(
        "# TooManyRules \n"
        "День 9: агент управляет контекстом. Последние сообщения уходят в "
        "модель как есть, всё, что старше, сворачивается в **сводку** — "
        "отдельным вызовом модели, который виден в панели и оплачен по той же "
        "таблице цен. Стратегия переключается слева на живом диалоге: стек "
        "сообщений и файл сессии всегда полные, сжимается запрос, а не "
        "память, поэтому один и тот же разговор можно отправить обоими "
        "способами и сравнить ответы. В панели рядом стоят два числа — "
        "сколько ушло в модель и сколько ушло бы без сжатия — и честный итог: "
        "сэкономлено против потрачено на свёртки."
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
            status_md = gr.Markdown("")

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

        # --- Справа: дебаг-панель ---
        with gr.Column(scale=2):
            gr.Markdown("## Дебаг-панель")
            # Фиксированная высота со скроллом: системный промпт длинный,
            # иначе он выдавливает метрики и счётчики за экран.
            config_json = gr.JSON(
                label="Конфиг агента (AgentConfig)",
                height=220,
            )
            metrics_md = gr.Markdown(_metrics_md(None))
            # Сначала «что отправляем» (день 9), потом «сколько это от окна»
            # (день 8): блок контекста стоит над бюджетом, а сводка — сразу
            # под ним, потому что объясняет числа над собой.
            flow_md = gr.Markdown("")
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
            # Аккордеон появляется только тогда, когда модель вернула
            # reasoning_content (пресет «Флагман + thinking»).
            with gr.Accordion(
                "reasoning_content последнего вызова",
                open=False,
                visible=False,
            ) as reasoning_accordion:
                reasoning_md = gr.Markdown("")
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

    # Порядок выходов совпадает с порядком значений в `_view()`.
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
    ]
    COMMON_OUTPUTS = [agent_state, question_input] + VIEW_OUTPUTS

    demo.load(
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
    # Единственный обработчик, который сам знает содержимое поля ввода и
    # передаёт его в `_view()`: бюджет и полоса должны показать, что
    # произойдёт при отправке, ещё до нажатия «Отправить».
    fill_btn.click(
        on_fill_context,
        inputs=[agent_state, filler_size, preset_dropdown],
        outputs=COMMON_OUTPUTS,
    )


if __name__ == "__main__":
    demo.launch()
