# TooManyRules — приложение недели 2 (день 9: чат через агента + дебаг-панель;
# день 10: четыре стратегии, checkpoint'ы и ветки диалога).
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

import logging
from dataclasses import asdict

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
from presets import (
    BRANCH_QUESTION,
    BRANCH_STEPS,
    COMPARISON_SCENARIO,
    CONTROL_QUESTIONS,
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
        # Применённое обновление памяти перед упавшим вызовом оплачено и уже
        # применено, а строки журнала у несостоявшегося хода нет: не
        # показать её здесь — значит не показать нигде, кроме накопленных
        # счётчиков.
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
        "влезать. Стратегия сама не переключается: выбрать её можно слева, "
        "в списке «Стратегия контекста»."
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
        # итогом ровно на размер этой памяти.
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
        }
        for turn in turns
    ]
    return pd.DataFrame(
        rows,
        columns=["ход", "стек до хода", "оценка", "prompt", "completion",
                 "total", "оценка/факт", "стоимость", "накопительно",
                 "стратегия", "отправлено сообщ.", "вся история", "служебные"],
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
        f"- **🧵 Служебные вызовы:** {_fmt_int(totals['service_tokens'])} "
        f"токенов, {_fmt_cost(totals['service_cost_usd'])} — это цена памяти "
        f"стратегий; на запросах при этом {verdict} "
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


def _branches_md(agent: Agent) -> str:
    """Блок «Ветки диалога» (спецификация дня 10, §7.3): что это за сессия,
    какие у неё checkpoint'ы и какие ветки от них уже созданы, и всё
    семейство целиком. Ни своей копии переписки, ни своего списка веток
    интерфейс не ведёт — всё здесь читается из агента и реестра заново."""
    family = branch_family(agent)
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
                for member in family
                if member.branch is not None
                and member.branch.parent == agent.session_id
                and member.branch.checkpoint == cp.id
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
    ]
    rows = []
    for a in agents():
        turns = a.turns
        if not turns or all(t.cost_usd is None for t in turns):
            cost_str = "н/д"
        else:
            total_cost = sum(t.cost_usd or 0.0 for t in turns) + sum(
                t.service_cost_usd or 0.0 for t in turns
            )
            cost_str = _fmt_cost(total_cost)
        avg_time = (
            f"{sum(t.elapsed + t.service_elapsed for t in turns) / len(turns):.2f}"
            if turns else "н/д"
        )
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
        })
    return pd.DataFrame(rows, columns=columns)


def _view(agent: Agent, status: str, question: str = "") -> tuple:
    """Полный вид на состояние агента — фиксированный кортеж из 22 значений
    (18 — до дня 10), позиционно раскладывающийся в `VIEW_OUTPUTS`. Порядок —
    часть контракта обработчиков ниже.

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
    # автообновления, как и на дне 6, здесь нет.
    session_file = STORE.read_file(state["session_id"])
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
        # Поле ввода чистим только при успехе; при ошибке вопрос остаётся
        # в поле, чтобы его можно было отправить повторно.
        message_update = ""
    else:
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
    сохраняются, а checkpoint'ы и происхождение ветки уходят вместе с
    диалогом (день 10) — они ссылались на диалог, которого больше нет; ветки,
    созданные раньше, при этом не трогаются, у них свои файлы."""
    if agent is None:  # страховка на случай сессии без сработавшего load
        agent = _new_agent(preset_name)
        note = "агент поднят заново"
    else:
        # Есть что сказать про checkpoint'ы — сравниваем до и после reset().
        had_checkpoints = bool(agent.checkpoints)
        family_before = len(branch_family(agent)) - 1
        agent.reset()
        note = f"файл сессии `{agent.session_id}` удалён"
        if had_checkpoints:
            note += ", checkpoint'ы и происхождение ветки очищены вместе с диалогом"
        if family_before:
            note += (
                f" (веток от этого диалога — {family_before}, они не тронуты)"
            )
    return (
        agent,
        gr.update(),
        *_view(
            agent,
            f"Диалог агента {_agent_title(agent)} сброшен: стек сообщений пуст, "
            f"{note}, счётчики агента сохранены.",
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
            f"стратегий. Ветку от него создаёт «Ветка от checkpoint'а»; "
            f"диалог можно продолжать — checkpoint останется тем, каким был."
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
    branch = agent.fork(checkpoint_id, session_id, make_strategies())
    if branch is None:
        return (
            agent,
            gr.update(),
            *_view(
                agent,
                f"Ветка от checkpoint'а «{checkpoint_id}» не создана — см. лог.",
            ),
        )
    return (
        branch,
        gr.update(),
        *_view(
            branch,
            f"Создана ветка {branch.session_id} от {agent.session_id} · "
            f"{checkpoint_id}: общие {branch.branch.messages} сообщ. "
            f"скопированы, стратегия «{branch.strategy.name}», файл уже на "
            f"диске. Исходный диалог не тронут — он в списке «Ветка диалога».",
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
    "Шаг 6 / В1 · тиран Goultar",
    "К1 · контроль: время и состав",
    "К2 · контроль: договорённость",
    "К3 · контроль: поправка коробок",
    "В2 · тиран Nom",
    "КВ · сводный сетап",
]


with gr.Blocks(title="TooManyRules") as demo:
    gr.Markdown(
        "# TooManyRules \n"
        "День 10: четыре способа отправить историю в модель и один способ её "
        "разветвить. Стратегия слева переключается на живом диалоге: стек "
        "сообщений и файл сессии всегда полные, меняется только то, что из "
        "них уходит в запрос, — «Вся история» и «Сводка + последние N» "
        "(день 9) стоят рядом со **«Скользящим окном»** (последние N сообщ., "
        "остальное отброшено без замены — плата деталями, а не токенами) и "
        "**«Фактами + последними N»** (блок «ключ — значение», который "
        "обновляет служебный вызов перед каждым ответом — плата вызовом на "
        "каждом ходе). Одна и та же история отправляется разными способами, "
        "и ответы можно сравнить. Ниже чата — «Сохранить checkpoint» и "
        "«Ветка от checkpoint'а»: они заводят от текущей точки диалога "
        "независимое продолжение — отдельную сессию со своим файлом, своей "
        "стратегией и своими счётчиками; переключатель «Ветка диалога» стоит "
        "рядом со стратегией контекста. Сравнение всех стратегий на одном "
        "сценарии — в `docs/TooManyRules — День 10 сравнение стратегий.md`."
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
            # «Агенты процесса» (день 10, §7.5) — сразу под «Ростом по
            # ходам»: первое место, где агенты сравниваются в одной таблице,
            # день 8 от этого сознательно отказался, день 10 делает — расход
            # токенов между стратегиями сравнивать нужно. Никакой подсветки.
            gr.Markdown("### Агенты процесса")
            agents_table = gr.Dataframe(
                value=_agents_table(),
                label="Расход по журналам ходов всех агентов реестра",
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

    # Порядок выходов совпадает с порядком значений в `_view()`. Значения
    # дня 10 — в конце списка, как и в кортеже `_view()` (§7.6).
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
