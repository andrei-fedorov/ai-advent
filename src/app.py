# TooManyRules — приложение недели 2 (день 6): чат через агента + дебаг-панель.
#
# Здесь только интерфейс. LLM-логики в этом файле нет: ни клиента OpenAI,
# ни chat.completions.create — всё общение с моделью инкапсулировано в
# `agent.Agent`, конфиги агентов проекта лежат в `presets.py`.
#
# Интерфейс — это вид на состояние агента: содержимое чата на каждом шаге
# рендерится из `agent.history`, своей копии переписки Gradio не ведёт.
# Поэтому на дне 7, когда история начнёт приходить из файла, интерфейс
# менять не придётся.
#
# Агентов в процессе много: переключатель в дебаг-панели листает реестр
# `agent.agents()`, и выбранный агент становится активным — вместе с ним
# переключаются чат, стек сообщений, конфиг и метрики.
#
# Вкладок больше нет: дни недели 2 наращивают одну и ту же сущность, и
# «что нового сегодня» показывает дебаг-панель. Дни 1-5 живут в
# замороженном артефакте `app_week1.py` (запуск: ./run.sh week1).
#
# Запуск: ./run.sh (или python app.py из src/ с активированным .venv)

import logging

import gradio as gr

from agent import Agent, agent_by_number, agents, delete_agent, process_stats
from presets import DEFAULT_PRESET, PRESET_NOTES, PRESETS

# Логирование настраивает точка входа — тот же формат, что на неделе 1.
# Сам агент только пишет в свой logger; здесь логировать нечего.
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
)


# --- Рендер дебаг-панели -------------------------------------------------

def _fmt_cost(cost_usd: float | None) -> str:
    return "н/д" if cost_usd is None else f"${cost_usd:.6f}"


def _fmt_tokens(value: int | None) -> str:
    return "н/д" if value is None else str(value)


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
            f"#{agent.number} · {agent.config.name} · "
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
        return (
            f"{header}\n\n"
            f"❌ **Ошибка:** {last_call['error']}\n\n"
            "Стек сообщений при ошибке не меняется — вопрос можно отправить "
            "повторно, история не задвоится."
        )
    return (
        f"{header}\n\n"
        f"- **⏱ Время ответа:** {last_call['elapsed']:.2f} s\n"
        f"- **🔢 Токены:** prompt={_fmt_tokens(last_call['prompt_tokens'])} / "
        f"completion={_fmt_tokens(last_call['completion_tokens'])} / "
        f"total={_fmt_tokens(last_call['total_tokens'])}\n"
        f"- **💲 Стоимость:** {_fmt_cost(last_call['cost_usd'])}\n"
        f"- **finish_reason:** `{last_call['finish_reason']}`\n"
        f"- **Модель:** `{last_call['model']}`"
    )


def _totals_md(agent_title: str, totals: dict) -> str:
    """Накопленное агентом за время жизни. «Сбросить диалог» эти счётчики
    не обнуляет — сброшен диалог, а не агент."""
    return (
        f"### Накоплено агентом {agent_title}\n\n"
        f"- **Вызовов:** {totals['calls']} (из них с ошибкой: {totals['errors']})\n"
        f"- **🔢 Токены:** prompt={totals['prompt_tokens']} / "
        f"completion={totals['completion_tokens']} / "
        f"total={totals['total_tokens']}\n"
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


def _view(agent: Agent, status: str) -> tuple:
    """Полный вид на состояние агента — фиксированный кортеж из 11 значений,
    позиционно раскладывающийся в `VIEW_OUTPUTS`. Порядок — часть контракта
    обработчиков ниже.

    Значения обоих выпадающих списков — тоже часть вида: иначе после
    переключения агента панель показывала бы одного, а списки — другого.
    """
    state = agent.debug_state()
    last_call = state["last_call"]
    reasoning = (last_call or {}).get("reasoning") or ""
    title = _agent_title(agent)
    messages = state["messages"]
    return (
        # 1. чат = стек агента; номер в подписи — чей именно диалог показан
        gr.update(
            value=messages,
            label=f"Диалог агента {title} (рендерится из стека агента)",
        ),
        status,                                  # 2. строка статуса
        state["config"],                         # 3. конфиг агента
        _metrics_md(last_call),                  # 4. метрики последнего вызова
        _totals_md(title, state["totals"]),      # 5. за время жизни
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
    как вкладка перешла на другого."""
    return Agent(PRESETS[preset_name])


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
            f"{reason} Поднят агент {_agent_title(agent)} с пустым стеком. "
            f"Агентов в реестре: {process_stats()['agents_alive']}.",
        ),
    )


def on_load(preset_name: str):
    """Открытие страницы: поднимаем агента для этой сессии и показываем его
    конфиг ещё до первого вопроса."""
    return _spawn(preset_name, "Страница открыта.")


def on_preset_change(preset_name: str):
    return _spawn(preset_name, "Сменён пресет — это новый агент и новый диалог.")


def on_new_agent(preset_name: str):
    """«Новый агент» — ещё один экземпляр с тем же пресетом. Сменой пресета
    двух агентов с одинаковым конфигом не получить, а именно такая пара
    нагляднее всего показывает, что стек у каждого инстанса свой."""
    return _spawn(preset_name, "Ещё один агент с тем же пресетом.")


def on_delete_agent(agent: Agent | None, preset_name: str):
    """«Удалить агент» — снять активного агента с учёта в процессе.

    Счётчики процесса при этом не откатываются: потраченные токены и деньги
    остались потраченными, а номер удалённого не переиспользуется. Активным
    становится последний из оставшихся; если реестр опустел — поднимаем
    нового, экрана без активного агента не бывает.
    """
    if agent is None:  # страховка на случай сессии без сработавшего load
        return _spawn(preset_name, "Удалять было нечего.")

    title = _agent_title(agent)
    delete_agent(agent.number)
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
        # Поле ввода чистим только при успехе; при ошибке вопрос остаётся
        # в поле, чтобы его можно было отправить повторно.
        message_update = ""
    else:
        status = f"❌ {reply.error}"
        message_update = gr.update()

    return (agent, message_update, *_view(agent, status))


def on_reset(agent: Agent | None, preset_name: str):
    """«Сбросить диалог» очищает стек сообщений агента. Счётчики за время
    жизни агента при этом сохраняются."""
    if agent is None:
        agent = _new_agent(preset_name)
    else:
        agent.reset()
    return (
        agent,
        gr.update(),
        *_view(
            agent,
            f"Диалог агента {_agent_title(agent)} сброшен: стек сообщений пуст, "
            "счётчики агента сохранены.",
        ),
    )


# --- Интерфейс -----------------------------------------------------------

PRESET_NOTES_MD = "\n".join(
    f"- **{name}** — {note}" for name, note in PRESET_NOTES.items()
)

with gr.Blocks(title="TooManyRules") as demo:
    gr.Markdown(
        "# TooManyRules — агент\n"
        "Ассистент по правилам настольной игры Too Many Bones "
        "(Chip Theory Games). Без RAG, модель отвечает из общих знаний.\n\n"
        "Вся работа с LLM — внутри сущности `Agent`: конфиг, стек сообщений, "
        "вызов API, токены и стоимость. Интерфейс — только вид на её состояние: "
        "агентов в процессе живёт много, и переключатель в дебаг-панели "
        "показывает стек любого из них. "
        "Дни 1-5 (вкладки недели 1) заморожены в `app_week1.py`, "
        "запуск — `./run.sh week1`."
    )

    # Экземпляр агента живёт в состоянии сессии: у каждой открытой вкладки
    # браузера свой агент. Создаётся на `demo.load`, а не здесь, чтобы
    # gr.State не пытался копировать живой объект с http-клиентом внутри.
    agent_state = gr.State(None)

    with gr.Row():
        # --- Слева: чат ---
        with gr.Column(scale=3):
            preset_dropdown = gr.Dropdown(
                choices=list(PRESETS),
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
            gr.Markdown(
                "Один и тот же агент, разные конфиги — то, что на неделе 1 "
                "было разными кусками кода:\n\n" + PRESET_NOTES_MD
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
            totals_md = gr.Markdown("")
            process_md = gr.Markdown(_process_md(process_stats()))
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


if __name__ == "__main__":
    demo.launch()
