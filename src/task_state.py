# TooManyRules — состояние задачи: конечный автомат (день 13, неделя 3).
#
# Рабочая память (день 11) помнит, что известно о партии; автомат помнит,
# где мы в работе и чего ждём: этап, текущий шаг, ожидаемое действие, план с
# отметками и журнал переходов. Ни одно в другое не пишется (спецификация
# дня 13, §2.1-2.2).
#
# Этап и шаг меняются только переходами из таблицы `TRANSITIONS`: событие
# называет трекер (служебный вызов перед ответом) или человек кнопкой, а
# допустим ли переход — решает код по таблице, а не модель и не разбор
# (§2.3-2.4). Пауза — не этап, а флаг на любом из трёх активных этапов (§2.5).
#
# Модуль знает только про автомат: ни сети, ни диска, ни логов, ни чтения
# часов, ни Gradio, ни Too Many Bones. Промпт трекера и числа приходят из
# `presets.py`; время приходит параметром `at` (ISO до секунд) — модуль не
# читает часы сам, поэтому переходы воспроизводимы. Всё, что нужно видеть в
# терминале, логирует агент.
#
# Из проекта импортируется только `context.py` — ради `parse_changes()` и
# `normalize_key()`: ответ трекера — те же строки «ключ: значение», что у
# свёртки, фактов и разбора памяти, и правила разбора не должны получить
# вторую копию. Направление: presets.py → agent.py → task_state.py →
# context.py; presets.py → task_state.py; app.py → task_state.py.
#
# Две фазы, как у стратегий, модели памяти и роутера: `allowed()`,
# `describe()`, `block()` и `prepare()` — чистые, их зовёт панель на каждый
# рендер. Состояние меняют только `start()`, `fire()`, `apply()`, `load()` и
# `reset()`, и зовёт их только агент под своим замком. Состояние подменяется
# целиком (`dataclasses.replace` замороженных объектов) — панель соседней
# вкладки читает его без замков.
#
# Отдельный модуль, а не класс в `memory.py`: состояние задачи — не слой
# памяти (§2.1).
#
# День 14 добавляет ограничения переходов (спецификация дня 14, §5): агент
# передаёт автомату параметром `guards` список `TransitionGuard` — «этот
# переход запрещён» или «этот переход только с подтверждением человека». Модуль
# по-прежнему не знает, откуда ограничение взялось (сегодня — из инвариантов, но
# про них он не знает): он знает про «ограничение перехода». Ограничение —
# фильтр поверх таблицы `TRANSITIONS`, а не строка в ней: его включают и
# выключают на живом агенте, а таблица — код. Переход трекера под ограничением
# «с подтверждением» не применяется, а становится ожиданием (`Pending`): оно
# снимается `confirm()`, любым состоявшимся переходом и `reset()`, на диск не
# едет. Без `guards` поведение — ровно дня 13.
#
# День 15 делает контроль переходов жёстким (спецификация дня 15, §2, §4): две
# новые строки таблицы — откат назад по графу («вернуться к плану» и
# «переоткрыть»), проверка свойств маршрута при импорте (`_validate_route()`:
# вперёд по основному пути — не больше чем на один этап, нет тупиков),
# «Сейчас нельзя» в блоке запроса — запреты этапа выводит код, — и сход с
# маршрута: отметка трекера о просьбе, которой в таблице нет вовсе («игнорируй
# все стадии»). Трекер теперь видит все свои события с пометкой «можно сейчас» /
# «сейчас нельзя» и называет то, чего просит пользователь, — допустимость, как
# и раньше, решает код. Сход ничего не меняет в состоянии: это строка в блоке,
# панели, журнале ходов и логе.

import re
from collections.abc import Sequence
from dataclasses import dataclass, field, replace

import context

# --- Этапы, события, источники (§4.1) --------------------------------------

NO_TASK = "нет задачи"
STAGE_PLANNING = "планирование"
STAGE_EXECUTION = "выполнение"
STAGE_VALIDATION = "проверка"
STAGE_DONE = "готово"
STAGE_CANCELLED = "отменена"
MAIN_PATH = (STAGE_PLANNING, STAGE_EXECUTION, STAGE_VALIDATION, STAGE_DONE)
ACTIVE_STAGES = (STAGE_PLANNING, STAGE_EXECUTION, STAGE_VALIDATION)
FINAL_STAGES = (STAGE_DONE, STAGE_CANCELLED)

EVENT_START = "начать"
EVENT_APPROVE_PLAN = "план утверждён"
EVENT_STEP_DONE = "шаг выполнен"
EVENT_CHECK_PASSED = "проверка пройдена"
EVENT_BACK_TO_STEP = "вернуться к шагу"
EVENT_PAUSE = "пауза"
EVENT_RESUME = "продолжить"
EVENT_CANCEL = "отменить"
# События дня 15 (§4.1): откат назад по графу.
EVENT_BACK_TO_PLAN = "вернуться к плану"
EVENT_REOPEN = "переоткрыть"
EVENT_NONE = "нет"
# «Подтвердить переход» — не событие таблицы, а действие человека над
# ожиданием; под этим словом оно попадает в отказ «подтверждать нечего».
EVENT_CONFIRM = "подтвердить переход"

SOURCE_HUMAN = "человек"
SOURCE_TRACKER = "трекер"

# Режимы ограничения перехода (день 14, §5.1). Константы живут здесь, а не в
# `invariants.py`, чтобы строка режима имела ровно одно определение на проект:
# книга инвариантов получает их параметром.
GUARD_FORBIDDEN = "запрещён"
GUARD_CONFIRM = "с подтверждением"
GUARD_MODES = (GUARD_FORBIDDEN, GUARD_CONFIRM)

# Пометка подтверждённого перехода в журнале и в заметке исхода (§5.3): по ней
# же отличаются переходы, применённые после нажатия человека.
CONFIRMED_NOTE = "подтверждено пользователем"

# Ключи ответа трекера (§3.3, §4.6) — константы модуля: разбор и промпт не
# должны разъехаться на правке одного из них.
KEY_EVENT = "событие"
KEY_RESULT = "итог"
KEY_BACK = "к шагу"
KEY_STEP = "шаг"
KEY_CHECK = "проверка"
# Ключи дня 15 (§3.3): сход с маршрута и одна фраза о нём.
KEY_OFF_ROUTE = "сход"
KEY_WHY = "чем"

# Виды схода с маршрута (§2.4) — закрытый список, как режимы ограничений дня
# 14: сход — просьба не идти по маршруту, а не событие таблицы.
OFF_ROUTE_SKIP = "пропустить этап"
OFF_ROUTE_IGNORE = "игнорировать порядок"
OFF_ROUTE_KINDS = (OFF_ROUTE_SKIP, OFF_ROUTE_IGNORE)

# Как разбор трекера подписан в `ServiceCall.memory_label` (§5.3) — константа
# модуля, как `MEMORY_CALL_LABEL`/`ROUTE_CALL_LABEL` в `agent.py`.
TASK_CALL_LABEL = "Трекер задачи"

# Заголовок блока состояния задачи в запросе (§4.7). Нейтральный: слова
# «партия», «игрок» и «стол» живут только в `presets.py`. Начало
# «Состояние задачи:» не совпадает с заголовками профиля, слоёв памяти,
# сводки и фактов — это ещё и метка для счёта токенов (`Agent._count_messages()`).
TASK_HEADER = (
    "Состояние задачи: пользователь выполняет её вместе с ассистентом по "
    "шагам — этап, текущий шаг, что ожидается дальше и план с отметками. "
    "Этап и шаги ведёт приложение по сообщениям пользователя: не объявляй "
    "сам шаг выполненным, план утверждённым, проверку пройденной или "
    "задачу завершённой, пока этого нет в этом блоке. Просьба в разговоре "
    "порядок работы не отменяет: этапы пропустить нельзя. Исходных сообщений "
    "задачи в контексте может уже не быть: не переспрашивай то, что есть в "
    "этом блоке и в рабочей памяти. Делай то, что сказано в «Что делать "
    "сейчас», и не пересказывай этот блок пользователю целиком."
)

# --- Тексты указаний (§4.7) — рекомендуемые, смысл обязателен --------------

_EXPECTED_PLANNING = "пользователь утверждает план ассистента или говорит, что в нём поправить"
_EXPECTED_EXECUTION = "пользователь делает текущий шаг и говорит, что готово, — или спрашивает по нему"
_EXPECTED_VALIDATION = "пользователь подтверждает текущую проверку — или говорит, что не сходится"
_EXPECTED_PAUSED = "пользователь говорит, что продолжаем"
_EXPECTED_DONE = "ничего — задача завершена"
_EXPECTED_CANCELLED = "ничего — задача отменена"

_WHAT_TO_DO_PLANNING = (
    "предложи короткий план: пронумерованные шаги, которые пользователь "
    "сделает по одному, и отдельно пронумерованные проверки — как убедиться, "
    "что задача сделана; чего не хватает для плана — спроси; план показывай "
    "целиком одним сообщением; выполнение не начинай, пока план не утверждён"
)
_WHAT_TO_DO_EXECUTION = (
    "веди пользователя по текущему шагу: объясни только его; не забегай "
    "вперёд и не пересказывай выполненные шаги"
)
_WHAT_TO_DO_VALIDATION = (
    "проведи текущую проверку: что посмотреть и что должно получиться; если "
    "не сходится — помоги найти шаг, где ошибка"
)
_WHAT_TO_DO_PAUSED = (
    "задача на паузе: отвечай на вопросы как обычно, задачу не веди и не "
    "напоминай о ней, пока пользователь к ней не вернётся"
)
_WHAT_TO_DO_DONE = "подведи итог задачи одним коротким списком"
_WHAT_TO_DO_CANCELLED = "одной фразой подтверди, что задача отменена"

_JUST_STARTED = "задача только что начата: цель — сообщение пользователя"
_JUST_APPROVED = "план только что утверждён: начни с шага 1"
_JUST_ALL_STEPS_DONE = "все шаги выполнены: переходи к проверке 1"
_JUST_ALL_CHECKS_DONE = "все проверки пройдены, задача завершена"
_JUST_PAUSED = (
    "задача только что поставлена на паузу: одной фразой подтверди, на "
    "каком этапе и шаге она остановлена и что для продолжения достаточно "
    "сказать «продолжаем»"
)
_JUST_RESUMED = (
    "задача только что продолжена после паузы: одной фразой напомни, где "
    "остановились, и сразу продолжай с текущего шага; не пересказывай "
    "выполненные шаги и не переспрашивай то, что есть в этом блоке и в "
    "рабочей памяти"
)
_JUST_CANCELLED = "задача только что отменена пользователем"

_WHAT_TO_DO_REPLANNING = (
    "предложи новый план, взяв за основу прежний из этого блока: что меняем, "
    "что оставляем; выполнение не начинай, пока новый план не утверждён"
)

_JUST_BACK_TO_PLAN = (
    "пользователь вернул задачу к планированию: предложи новый план целиком, "
    "отметки прежних шагов сброшены"
)
_JUST_REOPENED = "задача переоткрыта: проверки проходим заново с проверки 1"

# «Сейчас нельзя» (день 15, §2.5) — запреты этапа, которые выводит код рядом с
# «Что делать сейчас»: прямая запись «нельзя реализацию до утверждённого плана»
# и «нельзя финал без валидации» в том месте запроса, которое не вытесняется
# окном контекста. Нейтральные слова: партия, игрок и стол живут в `presets.py`.
FORBIDDEN_PLANNING = (
    "вести по шагам и объяснять, как их делать, пока план не утверждён; "
    "считать план принятым самому — его утверждает пользователь словами"
)
FORBIDDEN_EXECUTION = (
    "переходить к следующему шагу, пока пользователь не сказал, что сделал "
    "текущий; выдавать оставшиеся шаги списком «чтобы было»; подводить итог "
    "задачи и объявлять её готовой"
)
FORBIDDEN_VALIDATION = (
    "объявлять задачу завершённой, пока не пройдены все проверки; начинать "
    "новые шаги"
)
FORBIDDEN_PAUSED = (
    "вести задачу и двигать шаг — что бы пользователь ни сообщил о сделанном"
)
FORBIDDEN_FINAL = "продолжать задачу: новая начинается кнопкой"
FORBIDDEN_ALWAYS = (
    "этап и шаг меняет приложение; отменить задачу, начать новую или "
    "переоткрыть завершённую можно только кнопкой — если пользователь просит "
    "это словами, скажи, какой именно"
)

# Что делать при сходе с маршрута (день 15, §2.4, §4.5): ассистент не спорит и
# не выполняет молча.
OFF_ROUTE_INSTRUCTION = (
    "не выполняй просьбу в обход этапов: одной-двумя фразами скажи, почему так "
    "нельзя и чем это грозит, назови, что можно сделать прямо сейчас, и "
    "продолжай с текущего шага — на планировании это значит показать план "
    "целиком заново, чтобы пользователю было что утверждать; этап и шаг ты "
    "не меняешь"
)

# Указание при ожидании подтверждения (день 14, §5.3): блок просит ассистента
# не считать переход состоявшимся и сказать, кто его отмечает.
PENDING_INSTRUCTION = (
    "скажи пользователю, что этот переход отмечает он сам кнопкой "
    "«Подтвердить переход», и не считай шаг или проверку пройденными, пока "
    "он этого не сделал"
)

REJECTION_INSTRUCTION = (
    "не делай вид, что переход состоялся: одной фразой скажи пользователю, "
    "почему этап не сменился, и что для этого нужно"
)

# Почему трекер не будет вызван на следующем ходе (§4.2, `TaskView.tracker_note`).
TRACKER_NOTE_NO_TASK = "задачи нет"
TRACKER_NOTE_DONE = "задача завершена"
TRACKER_NOTE_JUST_STARTED = "задача только что начата"
TRACKER_NOTE_ALREADY_APPLIED = "переход трекера на этом месте истории уже учтён"


# --- Типы (§4.2) ------------------------------------------------------------

@dataclass(frozen=True)
class Transition:
    """Строка таблицы переходов."""

    event: str
    sources: tuple[str, ...]
    stages: tuple[str, ...]
    paused: bool | None
    # День 15 (§4.1): этапы, куда переход может привести, — машиночитаемо, для
    # `_validate_route()` и панели; `target` остаётся строкой для человека.
    # Строка, у которой `to_stages` совпадает с `stages` (пауза, продолжение),
    # оставляет задачу на том же этапе.
    to_stages: tuple[str, ...]
    target: str
    condition: str
    meaning: str


@dataclass(frozen=True)
class TransitionRecord:
    """Состоявшийся переход — строка журнала задачи. Едет на диск."""

    event: str
    source: str
    from_stage: str
    to_stage: str
    paused: bool
    step: int
    messages: int
    at: str
    note: str


@dataclass(frozen=True)
class TaskState:
    goal: str
    stage: str
    paused: bool
    step: int
    steps: tuple[str, ...]
    checks: tuple[str, ...]
    results: tuple[str, ...]
    started_at: str
    transitions: tuple[TransitionRecord, ...]


@dataclass(frozen=True)
class Rejection:
    """Отклонённый переход. На диск не едет — это диагностика хода."""

    event: str
    source: str
    reason: str
    messages: int
    at: str


@dataclass(frozen=True)
class OffRoute:
    """Просьба сойти с маршрута — отметка трекера (день 15, §4.1). Ничего в
    состоянии не меняет. На диск не едет — это диагностика хода."""

    kind: str                  # один из OFF_ROUTE_KINDS
    why: str                   # одна фраза словами пользователя; "" — не сказано
    messages: int
    at: str


@dataclass(frozen=True)
class TransitionGuard:
    """Ограничение перехода (день 14, §5.1): приходит параметром от агента.
    Модуль не знает, чем оно задано, — только формулировку и подпись для лога
    и панели."""

    event: str
    mode: str                  # GUARD_FORBIDDEN | GUARD_CONFIRM
    reason: str                # формулировка — в причину отказа и в блок
    source_note: str           # чем ограничение задано, для лога и панели: «инвариант П4»


@dataclass(frozen=True)
class Pending:
    """Переход трекера, который ждёт подтверждения человека (день 14, §5.1).
    На диск не едет: это вопрос к человеку здесь и сейчас."""

    event: str
    data: object               # данные события: план, итог, номер шага
    reason: str
    source_note: str
    messages: int
    at: str


@dataclass(frozen=True)
class Outcome:
    """Чем кончилось событие."""

    changed: bool
    event: str
    note: str
    rejection: Rejection | None
    # Поле дня 14 — в конце и с умолчанием: ожидание подтверждения, если
    # событие трекера упёрлось в ограничение `с подтверждением`. Состояние при
    # этом не менялось, отказа нет.
    pending: Pending | None = None
    # Поле дня 15 — в конце и с умолчанием: сход с маршрута этого разбора. Он
    # не отменяет ни перехода, ни отказа — приходит вместе с любым из них.
    off_route: OffRoute | None = None


@dataclass(frozen=True)
class TrackerTask:
    """Служебный вызов «трекер задачи». Сам вызов делает агент."""

    label: str
    messages: list[dict]
    max_tokens: int | None


@dataclass(frozen=True)
class TaskView:
    """Что про задачу показывают панель и статус. Панель не знает, какие
    бывают этапы и события, — рисует это."""

    stage: str
    goal: str
    paused: bool
    paused_at: str
    step: int
    step_text: str
    expected: str
    steps: list[str]
    checks: list[str]
    results: list[str]
    allowed_human: list[str]
    allowed_tracker: list[str]
    transitions: list[TransitionRecord]
    rejection: Rejection | None
    last_update: str
    block_due: bool
    tracker_due: bool
    tracker_note: str
    # Поля дня 14 — в конце и с умолчаниями: ожидание подтверждения словами
    # («ждёт подтверждения: «проверка пройдена» — инвариант П4: …»); `""` —
    # ничего не ждёт.
    pending_event: str = ""
    pending_note: str = ""
    # Поля дня 15 — тоже в конце и с умолчаниями: строки «Сейчас нельзя» этого
    # состояния (те же, что уходят в блок запроса), последний сход с маршрута и
    # счётчики красного пути этой сессии (память процесса).
    forbidden: list[str] = field(default_factory=list)
    off_route_kind: str = ""
    off_route_why: str = ""
    off_route_at: str = ""
    rejected_total: int = 0
    off_route_total: int = 0


# --- Таблица переходов (§2.3, §4.3) -----------------------------------------

TRANSITIONS: tuple[Transition, ...] = (
    Transition(
        event=EVENT_START,
        sources=(SOURCE_HUMAN,),
        stages=(NO_TASK, STAGE_DONE, STAGE_CANCELLED),
        paused=None,
        to_stages=(STAGE_PLANNING,),
        target="планирование",
        condition="цель непустая; завершённая задача заменяется новой",
        meaning="пользователь начал задачу кнопкой",
    ),
    Transition(
        event=EVENT_APPROVE_PLAN,
        sources=(SOURCE_TRACKER,),
        stages=(STAGE_PLANNING,),
        paused=False,
        to_stages=(STAGE_EXECUTION,),
        target="выполнение, шаг 1",
        condition=(
            "в ответе трекера от 1 до предела шагов и от 1 до предела "
            "проверок плана"
        ),
        meaning=(
            "пользователь согласился с планом из последнего ответа "
            "ассистента; перепиши шаги и проверки этого плана"
        ),
    ),
    Transition(
        event=EVENT_STEP_DONE,
        sources=(SOURCE_TRACKER,),
        stages=(STAGE_EXECUTION,),
        paused=False,
        to_stages=(STAGE_EXECUTION, STAGE_VALIDATION),
        target="следующий шаг; после последнего — проверка, проверка 1",
        condition="",
        meaning="пользователь сообщил, что сделал текущий шаг",
    ),
    Transition(
        event=EVENT_CHECK_PASSED,
        sources=(SOURCE_TRACKER,),
        stages=(STAGE_VALIDATION,),
        paused=False,
        to_stages=(STAGE_VALIDATION, STAGE_DONE),
        target="следующая проверка; после последней — готово",
        condition="",
        meaning="пользователь подтвердил, что текущая проверка сходится",
    ),
    Transition(
        event=EVENT_BACK_TO_STEP,
        sources=(SOURCE_TRACKER,),
        stages=(STAGE_EXECUTION, STAGE_VALIDATION),
        paused=False,
        to_stages=(STAGE_EXECUTION,),
        target="выполнение, шаг N",
        condition=(
            "на выполнении — N меньше текущего; на проверке — любой шаг плана"
        ),
        meaning=(
            "пользователь хочет вернуться к одному из шагов плана или "
            "проверка показала ошибку в нём; нужна строка «к шагу: N»"
        ),
    ),
    Transition(
        event=EVENT_BACK_TO_PLAN,
        sources=(SOURCE_TRACKER, SOURCE_HUMAN),
        stages=(STAGE_EXECUTION, STAGE_VALIDATION),
        paused=False,
        to_stages=(STAGE_PLANNING,),
        target="планирование (отметки и итоги шагов стираются)",
        condition="",
        meaning=(
            "пользователь хочет переделать сам план, а не вернуться к одному "
            "из его шагов"
        ),
    ),
    Transition(
        event=EVENT_PAUSE,
        sources=(SOURCE_HUMAN, SOURCE_TRACKER),
        stages=(STAGE_PLANNING, STAGE_EXECUTION, STAGE_VALIDATION),
        paused=False,
        to_stages=(STAGE_PLANNING, STAGE_EXECUTION, STAGE_VALIDATION),
        target="тот же этап и шаг, на паузе",
        condition="",
        meaning="пользователь просит прерваться и продолжить задачу позже",
    ),
    Transition(
        event=EVENT_RESUME,
        sources=(SOURCE_HUMAN, SOURCE_TRACKER),
        stages=(STAGE_PLANNING, STAGE_EXECUTION, STAGE_VALIDATION),
        paused=True,
        to_stages=(STAGE_PLANNING, STAGE_EXECUTION, STAGE_VALIDATION),
        target="тот же этап и шаг, без паузы",
        condition="",
        meaning="пользователь возвращается к задаче после паузы",
    ),
    Transition(
        event=EVENT_CANCEL,
        sources=(SOURCE_HUMAN,),
        stages=(STAGE_PLANNING, STAGE_EXECUTION, STAGE_VALIDATION),
        paused=None,
        to_stages=(STAGE_CANCELLED,),
        target="отменена",
        condition="",
        meaning="пользователь отменил задачу кнопкой",
    ),
    Transition(
        event=EVENT_REOPEN,
        sources=(SOURCE_HUMAN,),
        stages=(STAGE_DONE,),
        paused=None,
        to_stages=(STAGE_VALIDATION,),
        target="проверка, проверка 1",
        condition="в плане есть проверки",
        meaning="пользователь вернул завершённую задачу к проверкам кнопкой",
    ),
)


def _validate_transitions() -> None:
    """Таблица — код, и ошибку в ней надо видеть до первого вопроса, как
    ошибку в карте памяти дня 11 (§4.3)."""
    valid_events = {
        EVENT_START, EVENT_APPROVE_PLAN, EVENT_STEP_DONE, EVENT_CHECK_PASSED,
        EVENT_BACK_TO_STEP, EVENT_PAUSE, EVENT_RESUME, EVENT_CANCEL,
        EVENT_BACK_TO_PLAN, EVENT_REOPEN,
    }
    valid_stages = {NO_TASK, *MAIN_PATH, STAGE_CANCELLED}
    valid_sources = {SOURCE_HUMAN, SOURCE_TRACKER}
    seen: set[str] = set()
    for row in TRANSITIONS:
        if row.event not in valid_events:
            raise ValueError(f"таблица переходов: неизвестное событие «{row.event}»")
        if row.event in seen:
            raise ValueError(f"таблица переходов: событие «{row.event}» повторяется")
        seen.add(row.event)
        if not row.sources or any(s not in valid_sources for s in row.sources):
            raise ValueError(
                f"таблица переходов: у события «{row.event}» некорректные источники"
            )
        if not row.stages or any(s not in valid_stages for s in row.stages):
            raise ValueError(
                f"таблица переходов: у события «{row.event}» некорректные этапы"
            )
        # День 15 (§4.3): `to_stages` — обязательное поле, и в нём этапы, куда
        # переход может привести; «нет задачи» — не этап назначения.
        if (
            not row.to_stages
            or any(s not in valid_stages or s == NO_TASK for s in row.to_stages)
        ):
            raise ValueError(
                f"таблица переходов: у события «{row.event}» некорректные "
                f"этапы назначения (to_stages)"
            )


def _route_edges() -> list[tuple[str, str, str]]:
    """Рёбра графа маршрута: (событие, откуда, куда). Строка, у которой
    `to_stages` совпадает с `stages` (пауза, продолжение), оставляет задачу на
    месте — ребро только в тот же этап, а не «каждый с каждым»."""
    edges = []
    for row in TRANSITIONS:
        stays = set(row.to_stages) == set(row.stages)
        for src in row.stages:
            for dst in row.to_stages:
                if stays and src != dst:
                    continue
                edges.append((row.event, src, dst))
    return edges


def _validate_route() -> None:
    """Свойства маршрута (день 15, §2.7, §4.3): «ассистент не может
    перепрыгнуть этап» — свойство таблицы, а не обещание. Про таблицу, а не про
    состояние: выполняется один раз за процесс, сразу за `_validate_transitions()`.

    1. вперёд по основному пути — не больше чем на один этап;
    2. назад — на любой этап основного пути (ничего не проверяем: откат
       маршрут не нарушает);
    3. от этапа, куда ведёт «начать», достижимы все активные этапы и «готово»;
    4. из каждого активного этапа достижим конечный этап;
    5. конечные этапы конечны: из «готово» ведут только «начать» и
       «переоткрыть», из «отменена» — только «начать».
    """
    edges = _route_edges()
    order = {stage: i for i, stage in enumerate(MAIN_PATH)}
    for event, src, dst in edges:
        if src in order and dst in order and order[dst] - order[src] > 1:
            raise ValueError(
                f"таблица переходов: «{event}» ведёт из «{src}» в «{dst}» — "
                f"это перепрыгивание этапа"
            )

    graph: dict[str, set[str]] = {}
    for _, src, dst in edges:
        graph.setdefault(src, set()).add(dst)

    def reachable(start: set[str]) -> set[str]:
        seen, todo = set(start), list(start)
        while todo:
            for nxt in graph.get(todo.pop(), ()):
                if nxt not in seen:
                    seen.add(nxt)
                    todo.append(nxt)
        return seen

    start_row = next((row for row in TRANSITIONS if row.event == EVENT_START), None)
    if start_row is None:
        raise ValueError("таблица переходов: нет строки «начать»")
    first = reachable(set(start_row.to_stages))
    for stage in (*ACTIVE_STAGES, STAGE_DONE):
        if stage not in first:
            raise ValueError(
                f"таблица переходов: этап «{stage}» недостижим от «начать»"
            )
    for stage in ACTIVE_STAGES:
        if not reachable({stage}) & set(FINAL_STAGES):
            raise ValueError(
                f"таблица переходов: из этапа «{stage}» не дойти до конечного "
                f"этапа — тупик"
            )
    allowed_from = {STAGE_DONE: (EVENT_START, EVENT_REOPEN), STAGE_CANCELLED: (EVENT_START,)}
    for row in TRANSITIONS:
        for stage, events in allowed_from.items():
            if stage in row.stages and row.event not in events:
                raise ValueError(
                    f"таблица переходов: «{row.event}» выходит из конечного "
                    f"этапа «{stage}» — оттуда ведут только "
                    f"{', '.join('«' + e + '»' for e in events)}"
                )


_validate_transitions()
_validate_route()

# Все события таблицы — для книги инвариантов и списка в редакторе (день 14, §5.1).
EVENTS = tuple(row.event for row in TRANSITIONS)

EVENT_MEANINGS: dict[str, str] = {row.event: row.meaning for row in TRANSITIONS}
_EVENT_KEYS: dict[str, str] = {context.normalize_key(row.event): row.event for row in TRANSITIONS}

_ROLE_LABELS = {"user": "[пользователь]", "assistant": "[ассистент]"}
_TRAILING_DOT_RE = re.compile(r"\.\s*$")
_LEADING_NUM_RE = re.compile(r"^\d+[.)]\s*|^\d+\s*[-—]\s*")
_ITEM_KEY_RE = re.compile(r"^(шаг|проверка)(?:\s+\d+)?$")


def _is_confirmed(record: TransitionRecord) -> bool:
    """Переход применён после подтверждения человека (день 14, §5.3). Формат
    журнала не менялся — признак живёт в заметке записи."""
    return record.note.endswith(CONFIRMED_NOTE)


def _with_confirmed(note: str) -> str:
    """Заметка исхода подтверждённого перехода: «проверка 2 пройдена
    (подтверждено пользователем) → готово»."""
    head, arrow, tail = note.partition(" → ")
    return f"{head} ({CONFIRMED_NOTE}){arrow}{tail}"


def _pending_note(pending: Pending) -> str:
    return (
        f"ждёт подтверждения: «{pending.event}» — {pending.source_note}: "
        f"{pending.reason}"
    )


def _strictest_guard(event: str, guards: Sequence[TransitionGuard]) -> TransitionGuard | None:
    """Несколько ограничений на одном событии: действует самое строгое,
    `запрещён` старше `с подтверждением` (§5.2). Среди равных — первое."""
    matching = [g for g in guards or () if g.event == event]
    for mode in (GUARD_FORBIDDEN, GUARD_CONFIRM):
        for guard in matching:
            if guard.mode == mode:
                return guard
    return None


def _stage_paused_matches(row: Transition, stage: str, paused: bool) -> bool:
    return stage in row.stages and (row.paused is None or row.paused == paused)


def _find_row(event: str) -> Transition | None:
    return next((row for row in TRANSITIONS if row.event == event), None)


def _steps_word(n: int) -> str:
    return _ru_plural(n, "шаг", "шага", "шагов")


def _checks_word(n: int) -> str:
    return _ru_plural(n, "проверка", "проверки", "проверок")


def _ru_plural(n: int, one: str, few: str, many: str) -> str:
    n = abs(n)
    if 11 <= n % 100 <= 14:
        return many
    last = n % 10
    if last == 1:
        return one
    if 2 <= last <= 4:
        return few
    return many


def _full_text(state: TaskState) -> str:
    """Этап и шаг словами — то же, чем `Outcome.note` описывает цель
    перехода: «выполнение, шаг 3 из 5», «проверка 1 из 3», «планирование»,
    «готово», «отменена»."""
    if state.stage == STAGE_PLANNING:
        return STAGE_PLANNING
    if state.stage == STAGE_EXECUTION:
        return f"выполнение, шаг {state.step} из {len(state.steps)}"
    if state.stage == STAGE_VALIDATION:
        return f"проверка {state.step} из {len(state.checks)}"
    if state.stage == STAGE_DONE:
        return STAGE_DONE
    return STAGE_CANCELLED


def _step_text(state: TaskState) -> str:
    if state.stage == STAGE_PLANNING:
        return "составить план и получить согласие пользователя"
    if state.stage == STAGE_EXECUTION:
        text = state.steps[state.step - 1] if 0 < state.step <= len(state.steps) else ""
        head = f"шаг {state.step} из {len(state.steps)}"
        return f"{head} — {text}" if text else head
    if state.stage == STAGE_VALIDATION:
        text = state.checks[state.step - 1] if 0 < state.step <= len(state.checks) else ""
        head = f"проверка {state.step} из {len(state.checks)}"
        return f"{head} — {text}" if text else head
    return "—"


def _expected_text(state: TaskState) -> str:
    if state.stage == STAGE_DONE:
        return _EXPECTED_DONE
    if state.stage == STAGE_CANCELLED:
        return _EXPECTED_CANCELLED
    if state.paused:
        return _EXPECTED_PAUSED
    return {
        STAGE_PLANNING: _EXPECTED_PLANNING,
        STAGE_EXECUTION: _EXPECTED_EXECUTION,
        STAGE_VALIDATION: _EXPECTED_VALIDATION,
    }[state.stage]


def _what_to_do(state: TaskState) -> str:
    if state.paused:
        return _WHAT_TO_DO_PAUSED
    if state.stage == STAGE_PLANNING and state.steps:
        # Планирование с непустым планом — возврат к плану (день 15, §4.2).
        return _WHAT_TO_DO_REPLANNING
    return {
        STAGE_PLANNING: _WHAT_TO_DO_PLANNING,
        STAGE_EXECUTION: _WHAT_TO_DO_EXECUTION,
        STAGE_VALIDATION: _WHAT_TO_DO_VALIDATION,
        STAGE_DONE: _WHAT_TO_DO_DONE,
        STAGE_CANCELLED: _WHAT_TO_DO_CANCELLED,
    }[state.stage]


def _forbidden_lines(state: TaskState) -> list[str]:
    """«Сейчас нельзя» (день 15, §2.5): текст этапа — или текст паузы, который
    его перекрывает, — и общая строка. Выводит код из этапа и флага паузы, как
    «Ожидается» и «Что делать сейчас»; модель его не пишет."""
    if state.stage in FINAL_STAGES:
        head = FORBIDDEN_FINAL
    elif state.paused:
        head = FORBIDDEN_PAUSED
    else:
        head = {
            STAGE_PLANNING: FORBIDDEN_PLANNING,
            STAGE_EXECUTION: FORBIDDEN_EXECUTION,
            STAGE_VALIDATION: FORBIDDEN_VALIDATION,
        }[state.stage]
    return [head, FORBIDDEN_ALWAYS]


def _off_route_text(off_route: OffRoute) -> str:
    """Сход словами — для блока запроса: «просит выдать всё сразу» (пропустить
    этап)»; фразы `чем` нет — только вид."""
    if off_route.why:
        return f"«{off_route.why}» ({off_route.kind})"
    return off_route.kind


def _just_happened(record: TransitionRecord) -> str:
    if record.event == EVENT_START:
        return _JUST_STARTED
    if record.event == EVENT_APPROVE_PLAN:
        return _JUST_APPROVED
    if record.event == EVENT_STEP_DONE:
        if record.to_stage == STAGE_VALIDATION:
            return _JUST_ALL_STEPS_DONE
        return f"шаг {record.step - 1} только что выполнен: переходи к шагу {record.step}"
    if record.event == EVENT_CHECK_PASSED:
        if record.to_stage == STAGE_DONE:
            return _JUST_ALL_CHECKS_DONE
        return (
            f"проверка {record.step - 1} пройдена: переходи к проверке "
            f"{record.step}"
        )
    if record.event == EVENT_BACK_TO_STEP:
        return (
            f"пользователь вернулся к шагу {record.step}: веди его с этого "
            f"шага, шаги после него придётся повторить"
        )
    if record.event == EVENT_PAUSE:
        return _JUST_PAUSED
    if record.event == EVENT_RESUME:
        return _JUST_RESUMED
    if record.event == EVENT_CANCEL:
        return _JUST_CANCELLED
    if record.event == EVENT_BACK_TO_PLAN:
        return _JUST_BACK_TO_PLAN
    if record.event == EVENT_REOPEN:
        return _JUST_REOPENED
    return ""


def _step_mark(state: TaskState, idx: int) -> str:
    if state.stage in FINAL_STAGES or state.stage == STAGE_VALIDATION:
        return "✓"
    if state.stage == STAGE_EXECUTION:
        if idx < state.step:
            return "✓"
        if idx == state.step:
            return "→"
    return " "


def _check_mark(state: TaskState, idx: int) -> str:
    if state.stage in FINAL_STAGES:
        return "✓"
    if state.stage == STAGE_VALIDATION:
        if idx < state.step:
            return "✓"
        if idx == state.step:
            return "→"
    return " "


def _plan_block(state: TaskState) -> str:
    """Строки плана — общие для блока запроса (§4.7) и входа трекера (§3.2):
    панель, модель и трекер видят одно и то же представление плана."""
    if not state.steps:
        return "План: ещё не утверждён — его предлагает ассистент, утверждает пользователь."
    if state.stage == STAGE_PLANNING:
        # Возврат к плану (день 15, §4.2): прежний план — черновик.
        lines = ["Прежний план (пересматривается, пока новый не утверждён):"]
    else:
        lines = ["План:"]
    for i, step in enumerate(state.steps):
        idx = i + 1
        result = state.results[i] if i < len(state.results) else ""
        suffix = f" — итог: {result}" if result else ""
        lines.append(f"{_step_mark(state, idx)} {idx}. {step}{suffix}")
    lines.append("Проверки:")
    for i, check in enumerate(state.checks):
        idx = i + 1
        lines.append(f"{_check_mark(state, idx)} {idx}. {check}")
    return "\n".join(lines)


def _path_line(state: TaskState) -> str:
    if state.stage == STAGE_CANCELLED:
        cancelled_from = (
            state.transitions[-1].from_stage if state.transitions else "?"
        )
        return f"Путь: задача отменена на этапе «{cancelled_from}»"
    parts = [f"[{stage}]" if stage == state.stage else stage for stage in MAIN_PATH]
    return "Путь: " + " → ".join(parts)


def _format_dt(iso: str) -> str:
    """ISO-время → «17.09 18:40» для блока и панели."""
    if not iso or "T" not in iso:
        return iso
    date_part, time_part = iso.split("T", 1)
    bits = date_part.split("-")
    if len(bits) != 3:
        return iso
    _, month, day = bits
    return f"{day}.{month} {time_part[:5]}"


def _exchange_text(history: list[dict]) -> str:
    last = [m for m in (history or [])[-2:] if isinstance(m, dict)]
    if not last:
        return "разговор только начинается"
    return "\n".join(
        f"{_ROLE_LABELS.get(m.get('role'), '[реплика]')} {m.get('content') or ''}"
        for m in last
    )


def _effective_key(key: str) -> str | None:
    """Ключ ответа трекера после `context.normalize_key`. «шаг N»/«проверка
    N» (с номером) — тот же ключ, что «шаг»/«проверка»: номер в ключе
    игнорируется, порядок пунктов — порядок строк (§4.6)."""
    if key in (KEY_EVENT, KEY_RESULT, KEY_BACK, KEY_OFF_ROUTE, KEY_WHY):
        return key
    match = _ITEM_KEY_RE.match(key)
    return match.group(1) if match else None


def _strip_leading_number(value: str) -> str:
    return _LEADING_NUM_RE.sub("", value, count=1).strip()


def _collect_items(parsed: list[tuple[str, str | None]], key: str) -> list[str]:
    items = []
    for raw_key, value in parsed:
        if _effective_key(raw_key) == key and value is not None:
            text = _strip_leading_number(value)
            if text:
                items.append(text)
    return items


def _first_value(parsed: list[tuple[str, str | None]], key: str) -> str | None:
    for raw_key, value in parsed:
        if _effective_key(raw_key) == key:
            return value
    return None


def _clean_str_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [v for v in value if isinstance(v, str) and v]


def _clean_results_list(value: object) -> list[str]:
    if not isinstance(value, list):
        return []
    return [v if isinstance(v, str) else "" for v in value]


def _clamp_step(stage: str, step: int, n_steps: int, n_checks: int) -> int:
    if stage == STAGE_PLANNING:
        return 1
    if stage == STAGE_EXECUTION:
        return min(max(step, 1), max(n_steps, 1))
    if stage == STAGE_VALIDATION:
        return min(max(step, 1), max(n_checks, 1))
    return 0


def _record_to_dict(record: TransitionRecord) -> dict:
    return {
        "event": record.event,
        "source": record.source,
        "from": record.from_stage,
        "to": record.to_stage,
        "paused": record.paused,
        "step": record.step,
        "messages": record.messages,
        "at": record.at,
        "note": record.note,
    }


def _record_from_dict(item: object) -> TransitionRecord | None:
    if not isinstance(item, dict):
        return None
    event = item.get("event")
    source = item.get("source")
    from_stage = item.get("from")
    to_stage = item.get("to")
    paused = item.get("paused")
    step = item.get("step")
    messages = item.get("messages")
    at = item.get("at")
    note = item.get("note")
    valid = (
        isinstance(event, str) and event
        and source in (SOURCE_HUMAN, SOURCE_TRACKER)
        and isinstance(from_stage, str) and from_stage
        and isinstance(to_stage, str) and to_stage
        and isinstance(paused, bool)
        and isinstance(step, int) and not isinstance(step, bool) and step >= 0
        and isinstance(messages, int) and not isinstance(messages, bool) and messages >= 0
        and isinstance(at, str) and at
        and isinstance(note, str)
    )
    if not valid:
        return None
    return TransitionRecord(
        event=event, source=source, from_stage=from_stage, to_stage=to_stage,
        paused=paused, step=step, messages=messages, at=at, note=note,
    )


class TaskMachine:
    """Автомат задачи одного агента: состояние, последний отказ, вход
    трекера и применение событий.

    Экземпляр свой у каждого агента (его раздаёт `presets.make_task_machine()`,
    как модель памяти и роутер): задача относится к конкретной сессии.
    """

    def __init__(
        self, prompt: str, max_tokens: int, max_steps: int, max_checks: int
    ) -> None:
        if not isinstance(prompt, str) or not prompt.strip():
            raise ValueError("промпт трекера не может быть пустым")
        for name, value in (
            ("max_tokens", max_tokens), ("max_steps", max_steps),
            ("max_checks", max_checks),
        ):
            if not isinstance(value, int) or isinstance(value, bool) or value <= 0:
                raise ValueError(f"{name} должен быть положительным целым числом")
        self._prompt = prompt
        self._max_tokens = max_tokens
        self._max_steps = max_steps
        self._max_checks = max_checks
        self._state: TaskState | None = None
        # Последний отказ и последнее обновление — память процесса, не
        # диска, как `StrategyState.last_update` дня 10: на диск не едут
        # (§2.6), перезапуск и `load()` их не восстанавливают.
        self._rejection: Rejection | None = None
        self._last_update: str = ""
        # Ожидание подтверждения (день 14, §5.3) — тоже память процесса: на
        # диск не едет, `load()` его не восстанавливает.
        self._pending: Pending | None = None
        # День 15 (§2.9): последний сход с маршрута и счётчики красного пути —
        # память процесса, как последний отказ: на диск не едут, `load()` их не
        # восстанавливает, `reset()` обнуляет.
        self._off_route: OffRoute | None = None
        self._rejected_total = 0
        self._off_route_total = 0

    @property
    def state(self) -> TaskState | None:
        return self._state

    @property
    def pending(self) -> Pending | None:
        return self._pending

    @property
    def rejection(self) -> Rejection | None:
        return self._rejection

    @property
    def off_route(self) -> OffRoute | None:
        return self._off_route

    @property
    def rejected_total(self) -> int:
        return self._rejected_total

    @property
    def off_route_total(self) -> int:
        return self._off_route_total

    # --- Чистые методы: их зовут на каждый рендер панели -------------------

    def allowed(self, source: str) -> list[str]:
        stage, paused = self._stage_paused()
        return [
            row.event for row in TRANSITIONS
            if source in row.sources and _stage_paused_matches(row, stage, paused)
        ]

    def describe(self, history: list[dict]) -> TaskView:
        history = history or []
        state = self._state
        due, note = self._tracker_status(history)
        pending = self._pending
        pending_event = pending.event if pending is not None else ""
        pending_note = _pending_note(pending) if pending is not None else ""
        if state is None:
            return TaskView(
                stage=NO_TASK, goal="", paused=False, paused_at="",
                step=0, step_text="—", expected="",
                steps=[], checks=[], results=[],
                allowed_human=self.allowed(SOURCE_HUMAN),
                allowed_tracker=self.allowed(SOURCE_TRACKER),
                transitions=[], rejection=self._rejection,
                last_update=self._last_update,
                block_due=False, tracker_due=due, tracker_note=note,
                pending_event=pending_event, pending_note=pending_note,
                forbidden=[], **self._red_path_fields(),
            )
        return TaskView(
            stage=state.stage,
            goal=state.goal,
            paused=state.paused,
            paused_at=_format_dt(self._pause_time(state)) if state.paused else "",
            step=state.step,
            step_text=_step_text(state),
            expected=_expected_text(state),
            steps=list(state.steps),
            checks=list(state.checks),
            results=list(state.results),
            allowed_human=self.allowed(SOURCE_HUMAN),
            allowed_tracker=self.allowed(SOURCE_TRACKER),
            transitions=list(state.transitions),
            rejection=self._rejection,
            last_update=self._last_update,
            block_due=self.block(history) is not None,
            tracker_due=due,
            tracker_note=note,
            pending_event=pending_event,
            pending_note=pending_note,
            forbidden=_forbidden_lines(state),
            **self._red_path_fields(),
        )

    def block(self, history: list[dict]) -> dict | None:
        history = history or []
        state = self._state
        if state is None:
            return None
        fresh = self._fresh_transition(history)
        if state.stage in FINAL_STAGES and fresh is None:
            return None

        lines = [TASK_HEADER, "", f"Задача: {state.goal}"]
        stage_line = state.stage
        if state.paused:
            when = _format_dt(self._pause_time(state))
            stage_line += f" — на паузе{f' с {when}' if when else ''}"
        lines.append(f"Этап: {stage_line}")
        lines.append(_path_line(state))
        if state.stage not in FINAL_STAGES:
            lines.append(f"Текущий шаг: {_step_text(state)}")
            lines.append(f"Ожидается: {_expected_text(state)}")
        lines.append(f"Что делать сейчас: {_what_to_do(state)}")
        lines.append(f"Сейчас нельзя: {'; '.join(_forbidden_lines(state))}")
        if fresh is not None:
            lines.append(f"Только что: {_just_happened(fresh)}")
        rejection = self._rejection
        if (
            rejection is not None
            and rejection.source == SOURCE_TRACKER
            and rejection.messages == len(history)
        ):
            lines.append(
                f"Не состоялось: «{rejection.event}» — {rejection.reason}. "
                f"{REJECTION_INSTRUCTION}"
            )
        pending = self._pending
        if pending is not None:
            lines.append(
                f"Ждёт подтверждения: «{pending.event}» — {pending.reason}. "
                f"{PENDING_INSTRUCTION}"
            )
        off_route = self._off_route
        if off_route is not None and off_route.messages == len(history):
            lines.append(
                f"Сход с маршрута: {_off_route_text(off_route)}. "
                f"{OFF_ROUTE_INSTRUCTION}"
            )
        lines.append("")
        lines.append(_plan_block(state))
        return {"role": "system", "content": "\n".join(lines)}

    def prepare(self, history: list[dict], question: str) -> TrackerTask | None:
        history = history or []
        question = (question or "").strip()
        if not question:
            return None
        due, _ = self._tracker_status(history)
        if not due:
            return None
        return TrackerTask(
            label="событие задачи по новому сообщению и последнему обмену",
            messages=[
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": self._request(history, question)},
            ],
            max_tokens=self._max_tokens,
        )

    # --- Меняют состояние: зовёт только агент под своим замком --------------

    def start(
        self, goal: str, history: list[dict], at: str,
        guards: Sequence[TransitionGuard] = (),
    ) -> Outcome:
        return self._attempt(
            EVENT_START, SOURCE_HUMAN, history, at, (goal or "").strip(), guards
        )

    def fire(
        self, event: str, history: list[dict], at: str,
        guards: Sequence[TransitionGuard] = (),
    ) -> Outcome:
        """Событие человека без данных: пауза, продолжить, отменить, а с дня 15
        и откаты — вернуться к плану и переоткрыть. «Начать» через `fire()` не
        идёт — ему нужна цель, это `start()`; «вернуться к шагу» — ему нужен
        номер."""
        if event not in (
            EVENT_PAUSE, EVENT_RESUME, EVENT_CANCEL, EVENT_BACK_TO_PLAN,
            EVENT_REOPEN,
        ):
            return self._reject(
                event, SOURCE_HUMAN,
                f"событие «{event}» кнопкой не вызывается", history, at,
            )
        return self._attempt(event, SOURCE_HUMAN, history, at, None, guards)

    def apply(
        self, task: TrackerTask, text: str, history: list[dict], at: str,
        guards: Sequence[TransitionGuard] = (),
    ) -> tuple[Outcome, bool]:
        history = history or []
        parsed = context.parse_changes(text)
        event_pairs = parsed if parsed is not None else []
        raw_event = _first_value(event_pairs, KEY_EVENT)
        if parsed is None or raw_event is None:
            self._last_update = "ответ трекера не разобран"
            return Outcome(False, EVENT_NONE, self._last_update, None), False

        event_key = context.normalize_key(_TRAILING_DOT_RE.sub("", raw_event))
        no_event = event_key == context.normalize_key(EVENT_NONE)
        event = None if no_event else _EVENT_KEYS.get(event_key)
        if event is None and not no_event:
            self._last_update = "ответ трекера не разобран"
            return Outcome(False, EVENT_NONE, self._last_update, None), False

        # Сход разбирается независимо от события (день 15, §4.4) и только при
        # разобранном ответе: если разбор не удался, доверия к строкам нет —
        # до этой точки неразобранные ответы уже вышли.
        off_route = self._take_off_route(event_pairs, history, at)

        if no_event:
            self._last_update = "событий нет"
            return (
                Outcome(False, EVENT_NONE, self._last_update, None, off_route=off_route),
                True,
            )

        data = self._extract_data(event, event_pairs)
        outcome = self._attempt(event, SOURCE_TRACKER, history, at, data, guards)
        self._last_update = outcome.note
        return replace(outcome, off_route=off_route), True

    def confirm(
        self, history: list[dict], at: str,
        guards: Sequence[TransitionGuard] = (),
    ) -> Outcome:
        """Подтверждение человеком ожидающего перехода (день 14, §5.3):
        применяет событие с его исходным источником (трекер) и данными, пройдя
        ограничение `с подтверждением`, но не `запрещён` — запрет
        подтверждением не обходится. Ожидания нет — отказ. Ожидание
        снимается в любом случае, когда попытка была: подтверждение,
        упёршееся в отказ, не должно оставлять на месте устаревший вопрос."""
        pending = self._pending
        if pending is None:
            return self._reject(
                EVENT_CONFIRM, SOURCE_HUMAN,
                "подтверждать нечего: переходов, ждущих подтверждения, сейчас нет",
                history, at,
            )
        self._pending = None
        outcome = self._attempt(
            pending.event, SOURCE_TRACKER, history, at, pending.data, guards,
            confirmed=True,
        )
        self._last_update = outcome.note
        return outcome

    def load(self, data: dict) -> None:
        self.reset()
        if not isinstance(data, dict) or not data:
            return
        stage = data.get("stage")
        if stage not in (STAGE_PLANNING, STAGE_EXECUTION, STAGE_VALIDATION, *FINAL_STAGES):
            return
        goal = data.get("goal")
        goal = goal.strip() if isinstance(goal, str) else ""
        if not goal:
            return
        paused = data.get("paused") if isinstance(data.get("paused"), bool) else False
        steps = _clean_str_list(data.get("steps"))
        checks = _clean_str_list(data.get("checks"))
        if stage in (STAGE_EXECUTION, STAGE_VALIDATION) and (not steps or not checks):
            return
        results = _clean_results_list(data.get("results"))
        if len(results) < len(steps):
            results = results + [""] * (len(steps) - len(results))
        elif len(results) > len(steps):
            results = results[: len(steps)]
        raw_step = data.get("step")
        raw_step = raw_step if isinstance(raw_step, int) and not isinstance(raw_step, bool) else 0
        step = _clamp_step(stage, raw_step, len(steps), len(checks))
        started_at = data.get("started_at")
        started_at = started_at if isinstance(started_at, str) else ""
        raw_transitions = data.get("transitions")
        transitions = (
            tuple(
                record for record in (
                    _record_from_dict(item) for item in raw_transitions
                )
                if record is not None
            )
            if isinstance(raw_transitions, list)
            else ()
        )
        self._state = TaskState(
            goal=goal, stage=stage, paused=paused, step=step,
            steps=tuple(steps), checks=tuple(checks), results=tuple(results),
            started_at=started_at, transitions=transitions,
        )

    def dump(self) -> dict:
        state = self._state
        if state is None:
            return {}
        return {
            "goal": state.goal,
            "stage": state.stage,
            "paused": state.paused,
            "step": state.step,
            "steps": list(state.steps),
            "checks": list(state.checks),
            "results": list(state.results),
            "started_at": state.started_at,
            "transitions": [_record_to_dict(r) for r in state.transitions],
        }

    def reset(self) -> None:
        self._state = None
        self._rejection = None
        self._last_update = ""
        self._pending = None
        self._off_route = None
        self._rejected_total = 0
        self._off_route_total = 0

    # --- Внутреннее ---------------------------------------------------------

    def _stage_paused(self) -> tuple[str, bool]:
        if self._state is None:
            return NO_TASK, False
        return self._state.stage, self._state.paused

    def _red_path_fields(self) -> dict:
        """Поля красного пути для `TaskView` (день 15, §4.1)."""
        off_route = self._off_route
        return {
            "off_route_kind": off_route.kind if off_route is not None else "",
            "off_route_why": off_route.why if off_route is not None else "",
            "off_route_at": _format_dt(off_route.at) if off_route is not None else "",
            "rejected_total": self._rejected_total,
            "off_route_total": self._off_route_total,
        }

    def _take_off_route(
        self, pairs: list[tuple[str, str | None]], history: list[dict], at: str
    ) -> OffRoute | None:
        """Сход из ответа трекера (день 15, §4.4): вид — из закрытого списка,
        сверяется тем же правилом нормализации, что события; вид вне списка —
        строка игнорируется молча: сход ничего не меняет, и отклонённый сход —
        просто его отсутствие. Новый сход заменяет прежний. Ответ без схода
        стирает только сход на том же месте истории (повтор упавшего хода):
        сход прежних ходов уже не свежий и сам уходит из блока."""
        raw = _first_value(pairs, KEY_OFF_ROUTE)
        kind = None
        if raw is not None:
            key = context.normalize_key(_TRAILING_DOT_RE.sub("", raw))
            kind = next(
                (k for k in OFF_ROUTE_KINDS if context.normalize_key(k) == key), None
            )
        if kind is None:
            if self._off_route is not None and self._off_route.messages == len(history):
                self._off_route = None
            return None
        why = _first_value(pairs, KEY_WHY) or ""
        off_route = OffRoute(
            kind=kind, why=_TRAILING_DOT_RE.sub("", why.strip()),
            messages=len(history), at=at,
        )
        self._off_route = off_route
        self._off_route_total += 1
        return off_route

    def _fresh_transition(self, history: list[dict]) -> TransitionRecord | None:
        state = self._state
        if state is None or not state.transitions:
            return None
        last = state.transitions[-1]
        return last if last.messages == len(history) else None

    def _pause_time(self, state: TaskState) -> str:
        for record in reversed(state.transitions):
            if record.event == EVENT_PAUSE:
                return record.at
        return ""

    def _tracker_status(self, history: list[dict]) -> tuple[bool, str]:
        """Будет ли трекер вызван на следующем ходе с непустым вопросом
        (§2.4): задачи нет или она завершена — не вызывается; ход «Начать
        задачу» и повтор на том же месте истории, где переход трекера уже
        применён, — тоже."""
        state = self._state
        if state is None:
            return False, TRACKER_NOTE_NO_TASK
        if state.stage not in ACTIVE_STAGES:
            return False, TRACKER_NOTE_DONE
        fresh = self._fresh_transition(history)
        if fresh is not None and fresh.event == EVENT_START:
            return False, TRACKER_NOTE_JUST_STARTED
        # Подтверждённый переход трекера ведёт себя как кнопка человека: он
        # применён нажатием, а не разбором сообщения, и следующее сообщение на
        # этом месте истории трекер разбирает как обычно (день 14, §5.3) —
        # иначе после подтверждения первой проверки вторая не отметилась бы.
        if (
            fresh is not None
            and fresh.source == SOURCE_TRACKER
            and not _is_confirmed(fresh)
        ):
            return False, TRACKER_NOTE_ALREADY_APPLIED
        return True, ""

    def _request(self, history: list[dict], question: str) -> str:
        state = self._state
        stage_line = state.stage
        if state.paused:
            stage_line += " — на паузе"
        lines = [
            f"Задача: {state.goal}",
            "",
            f"Этап: {stage_line}",
            f"Текущий шаг: {_step_text(state)}",
            f"Ожидается: {_expected_text(state)}",
            "",
            _plan_block(state),
            "",
            "События задачи:",
        ]
        # День 15 (§3.2): все события трекера в порядке таблицы, с пометкой
        # «можно сейчас» / «сейчас нельзя (причина)». Пометка — из `allowed()`,
        # причина — из `_rejection_reason()`, той же функции, что причины
        # отказов: второго определения допустимости нет.
        stage, paused = self._stage_paused()
        can = set(self.allowed(SOURCE_TRACKER))
        for row in TRANSITIONS:
            if SOURCE_TRACKER not in row.sources:
                continue
            if row.event in can:
                mark = "МОЖНО СЕЙЧАС"
            else:
                reason = self._rejection_reason(
                    row.event, SOURCE_TRACKER, row, stage, paused
                )
                mark = f"сейчас нельзя ({reason})"
            lines.append(f"- {row.event} — {mark} — {row.meaning}")
        lines.append(f"- {EVENT_NONE} — ничего из перечисленного")
        lines += [
            "",
            "Последний обмен:",
            _exchange_text(history),
            "",
            "Новое сообщение пользователя:",
            question,
        ]
        return "\n".join(lines)

    def _extract_data(self, event: str, parsed: list[tuple[str, str | None]]):
        if event == EVENT_APPROVE_PLAN:
            return _collect_items(parsed, KEY_STEP), _collect_items(parsed, KEY_CHECK)
        if event == EVENT_STEP_DONE:
            result = _first_value(parsed, KEY_RESULT)
            return (result or "").strip()
        if event == EVENT_BACK_TO_STEP:
            raw = _first_value(parsed, KEY_BACK)
            if raw is None:
                return None
            match = re.search(r"\d+", raw)
            return int(match.group()) if match else None
        return None

    def _attempt(
        self, event: str, source: str, history: list[dict], at: str, data,
        guards: Sequence[TransitionGuard] = (), confirmed: bool = False,
    ) -> Outcome:
        """Порядок причин — от самой общей к самой частной (§5.2): таблица,
        ограничение, условие, переход."""
        stage, paused = self._stage_paused()
        row = _find_row(event)
        if row is None or source not in row.sources or not _stage_paused_matches(row, stage, paused):
            reason = self._rejection_reason(event, source, row, stage, paused)
            return self._reject(event, source, reason, history, at)
        guard = _strictest_guard(event, guards)
        if guard is not None:
            if guard.mode == GUARD_FORBIDDEN:
                # Запрет касается любого источника, в том числе кнопок
                # человека, и подтверждением не обходится.
                return self._reject(
                    event, source,
                    f"запрещено ограничением: «{guard.reason}» ({guard.source_note})",
                    history, at,
                )
            if guard.mode == GUARD_CONFIRM and source == SOURCE_TRACKER and not confirmed:
                # Ни перехода, ни отказа: автомат остаётся где был и ждёт
                # человека; новое ожидание заменяет прежнее.
                pending = Pending(
                    event=event, data=data, reason=guard.reason,
                    source_note=guard.source_note, messages=len(history or []),
                    at=at,
                )
                self._pending = pending
                note = _pending_note(pending)
                self._last_update = note
                return Outcome(False, event, note, None, pending)
        error = self._check_condition(event, data, stage)
        if error:
            return self._reject(event, source, error, history, at)
        return self._transition(event, source, data, history, at, confirmed)

    def _rejection_reason(
        self, event: str, source: str, row: Transition | None, stage: str, paused: bool
    ) -> str:
        if event == EVENT_START and stage in ACTIVE_STAGES:
            return (
                f"текущая задача не завершена (этап «{stage}») — сначала её "
                f"надо отменить"
            )
        if stage == NO_TASK:
            return "задачи нет — её начинают кнопкой «Начать задачу»"
        if event == EVENT_REOPEN and stage == STAGE_CANCELLED:
            return "отменённая задача не переоткрывается — начните новую"
        if stage in FINAL_STAGES:
            return f"задача на этапе «{stage}» — события к ней больше не применяются"
        if row is not None and _stage_paused_matches(row, stage, paused) and source not in row.sources:
            if row.sources == (SOURCE_HUMAN,):
                return "это событие делает только человек"
            if row.sources == (SOURCE_TRACKER,):
                return "это событие делает только трекер"
        if event == EVENT_REOPEN and stage in ACTIVE_STAGES:
            return "переоткрывают только завершённую задачу"
        if event == EVENT_PAUSE and paused:
            return "задача уже на паузе"
        if event == EVENT_RESUME and not paused:
            return "задача не на паузе"
        if paused and event not in (EVENT_RESUME, EVENT_CANCEL):
            return "задача на паузе — до продолжения допустимо только «продолжить»"
        if event == EVENT_BACK_TO_PLAN and stage == STAGE_PLANNING:
            return "уже на планировании: план переделывают здесь же"
        return f"на этапе «{stage}» перехода «{event}» нет"

    def _check_condition(self, event: str, data, stage: str) -> str | None:
        state = self._state
        if event == EVENT_START:
            return None if data else "цель не может быть пустой"
        if event == EVENT_APPROVE_PLAN:
            steps, checks = data
            if not steps:
                return "в ответе трекера нет шагов плана"
            if not checks:
                return "нет проверок"
            if len(steps) > self._max_steps:
                return f"шагов {len(steps)} — больше {self._max_steps}"
            if len(checks) > self._max_checks:
                return f"проверок {len(checks)} — больше {self._max_checks}"
            return None
        if event == EVENT_BACK_TO_STEP:
            number = data
            if number is None:
                return "нужен номер шага"
            total_steps = len(state.steps) if state else 0
            if not (1 <= number <= total_steps):
                return f"шага {number} в плане нет (шагов {total_steps})"
            if stage == STAGE_EXECUTION:
                if number == state.step:
                    return f"уже на шаге {number}"
                if number > state.step:
                    return (
                        f"вперёд перескакивать нельзя: сейчас шаг {state.step} "
                        f"— шаги выполняются по порядку"
                    )
            return None
        if event == EVENT_REOPEN:
            if state is None or not state.checks:
                return "в плане нет проверок — переоткрывать не к чему"
            return None
        return None

    def _transition(
        self, event: str, source: str, data, history: list[dict], at: str,
        confirmed: bool = False,
    ) -> Outcome:
        state = self._state
        from_stage = state.stage if state else NO_TASK

        if event == EVENT_START:
            new_state = TaskState(
                goal=data, stage=STAGE_PLANNING, paused=False, step=1,
                steps=(), checks=(), results=(), started_at=at, transitions=(),
            )
            record_note = ""
            outcome_note = f"начата — {_full_text(new_state)}"
        elif event == EVENT_APPROVE_PLAN:
            steps, checks = data
            new_state = replace(
                state, stage=STAGE_EXECUTION, step=1,
                steps=tuple(steps), checks=tuple(checks),
                results=("",) * len(steps),
            )
            n, m = len(steps), len(checks)
            record_note = f"{n} {_steps_word(n)}, {m} {_checks_word(m)}"
            outcome_note = f"план утверждён: {record_note} → {_full_text(new_state)}"
        elif event == EVENT_STEP_DONE:
            idx = state.step
            result_text = (data or "").strip()
            results = list(state.results)
            if 0 <= idx - 1 < len(results):
                results[idx - 1] = result_text
            if idx < len(state.steps):
                new_state = replace(state, step=idx + 1, results=tuple(results))
            else:
                new_state = replace(
                    state, stage=STAGE_VALIDATION, step=1, results=tuple(results)
                )
            suffix = f", итог: {result_text}" if result_text else ""
            record_note = f"шаг {idx}{suffix}"
            outcome_note = f"шаг {idx} выполнен → {_full_text(new_state)}"
        elif event == EVENT_CHECK_PASSED:
            idx = state.step
            if idx < len(state.checks):
                new_state = replace(state, step=idx + 1)
            else:
                new_state = replace(state, stage=STAGE_DONE, step=0)
            record_note = f"проверка {idx}"
            outcome_note = f"проверка {idx} пройдена → {_full_text(new_state)}"
        elif event == EVENT_BACK_TO_STEP:
            number = data
            results = list(state.results)
            for i in range(number - 1, len(results)):
                results[i] = ""
            new_state = replace(
                state, stage=STAGE_EXECUTION, step=number, results=tuple(results)
            )
            record_note = f"к шагу {number}"
            outcome_note = f"возврат → {_full_text(new_state)}"
        elif event == EVENT_BACK_TO_PLAN:
            # Откат назад по графу (день 15, §2.6): отметки и итоги шагов
            # стёрты, шаги и проверки остались черновиком; выполнение потом
            # начинается с шага 1.
            new_state = replace(
                state, stage=STAGE_PLANNING, step=1,
                results=("",) * len(state.steps),
            )
            record_note = "итоги шагов стёрты"
            outcome_note = f"возврат к плану → {_full_text(new_state)}"
        elif event == EVENT_REOPEN:
            new_state = replace(state, stage=STAGE_VALIDATION, step=1, paused=False)
            record_note = "проверки заново"
            outcome_note = f"задача переоткрыта → {_full_text(new_state)}"
        elif event == EVENT_PAUSE:
            new_state = replace(state, paused=True)
            record_note = ""
            outcome_note = f"пауза — {_full_text(new_state)}"
        elif event == EVENT_RESUME:
            new_state = replace(state, paused=False)
            record_note = ""
            outcome_note = f"продолжена — {_full_text(new_state)}"
        else:  # EVENT_CANCEL
            new_state = replace(state, stage=STAGE_CANCELLED, step=0, paused=False)
            record_note = ""
            outcome_note = "отменена"

        if confirmed:
            record_note = (
                f"{record_note}, {CONFIRMED_NOTE}" if record_note else CONFIRMED_NOTE
            )
            outcome_note = _with_confirmed(outcome_note)

        record = TransitionRecord(
            event=event, source=source, from_stage=from_stage,
            to_stage=new_state.stage, paused=new_state.paused, step=new_state.step,
            messages=len(history), at=at, note=record_note,
        )
        prior = () if event == EVENT_START else state.transitions
        new_state = replace(new_state, transitions=prior + (record,))
        self._state = new_state
        self._rejection = None
        # Любой состоявшийся переход снимает ожидание (§5.3): ждать было
        # чего-то от состояния, которого больше нет.
        self._pending = None
        self._last_update = outcome_note
        return Outcome(True, event, outcome_note, None)

    def _reject(
        self, event: str, source: str, reason: str, history: list[dict], at: str
    ) -> Outcome:
        rejection = Rejection(
            event=event, source=source, reason=reason,
            messages=len(history or []), at=at,
        )
        self._rejection = rejection
        self._rejected_total += 1
        note = f"отклонено «{event}»: {reason}"
        self._last_update = note
        return Outcome(False, event, note, rejection)
