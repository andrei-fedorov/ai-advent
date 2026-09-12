# TooManyRules — стратегии управления контекстом (день 9, неделя 2).
#
# Модуль знает только про списки сообщений: ни Gradio, ни Too Many Bones, ни
# вызовов API здесь нет и быть не может. Из проекта не импортируется ничего —
# это второй лист графа зависимостей рядом с `tokens.py`
# (`agent.py` → `tokens.py`, `context.py`), см.
# `docs/TooManyRules — Неделя 2 архитектура.md`, §3, и спецификацию дня 9, §4.
#
# Главное решение дня: стратегия решает не что хранить, а что отправлять.
# Стек сообщений агента и файл сессии всегда полные, сводка живёт отдельным
# полем рядом с историей, а не вместо неё (спецификация дня 9, §2.1). Отсюда
# и разделение на две фазы:
#
#   prepare() — «что надо сделать до запроса» (чистая, описывает работу),
#   apply()   — «вот результат» (единственный метод, который меняет память),
#   build()   — «что уходит в модель» (чистая).
#
# `prepare()` и `build()` зовёт дебаг-панель на каждый свой рендер. Если бы
# свёртка жила внутри `build()`, каждое движение в интерфейсе стоило бы
# денег — поэтому служебный вызов делает агент, а здесь только сборка
# сообщений для него.
#
# Логов у модуля нет, как и у `tokens.py`: чистые функции зовут слишком
# часто. Всё, что нужно видеть в терминале, логирует агент.

from dataclasses import dataclass
from typing import Protocol

# Пометка перед текстом сводки в запросе: модель должна понимать, что читает
# пересказ, а не сообщение собеседника (спецификация дня 9, §3.4). Это про
# механику, а не про настолки, — поэтому константа здесь, а не в `presets.py`.
SUMMARY_HEADER = (
    "Краткая сводка предыдущей части разговора; исходных сообщений в "
    "контексте больше нет."
)

# Имя стратегии «Вся история» — константа модуля, а не проекта: это поведение
# дней 6-8, и `agent.py` поднимает её сам, когда набор стратегий ему не дали.
FULL_HISTORY_NAME = "Вся история"

# Разметка сворачиваемых сообщений для служебного вызова. Нейтральная: модуль
# не знает ни про игроков, ни про правила.
_ROLE_LABELS = {"user": "[пользователь]", "assistant": "[ассистент]"}
_NO_PREVIOUS_SUMMARY = "предыдущей сводки нет — это первая свёртка"


@dataclass(frozen=True)
class ContextTask:
    """Служебный вызов модели, который стратегия просит сделать до основного
    запроса. Сам вызов делает агент: в `context.py` сети нет."""

    kind: str                 # "summary"; день 10 добавит свои виды
    label: str                # человеческая строка для лога, статуса и панели
    messages: list[dict]      # готовый список сообщений для служебного вызова
    max_tokens: int | None
    # Граница, а не количество: результат покроет `history[:covers]`, включая
    # то, что уже было свёрнуто раньше. Сколько сообщений свернёт именно этот
    # вызов, агент считает сам (`ServiceCall.covers` — уже количество).
    covers: int


@dataclass(frozen=True)
class StrategyState:
    """Что стратегия помнит и что про неё показывает панель. Панель не знает,
    какие бывают стратегии, — она рисует то, что вернули отсюда."""

    name: str
    # Параметры словами: «последние 6 сообщений, свёртка каждые 6».
    description: str
    memory_label: str         # как называется память в панели: «Сводка»
    memory_text: str          # сама память текстом; "" — памяти нет
    covered: int              # сколько сообщений истории уже свёрнуто
    updated_turns: int        # сколько раз память обновлялась за жизнь агента
    note: str                 # что случится на следующем ходе, человеческим языком


class ContextStrategy(Protocol):
    """Как собрать запрос из системного промпта, истории и нового вопроса.

    Протокол объявлен здесь, а не в `agent.py`, как `HistoryStore` на дне 6:
    хранилище обменивается с агентом простыми списками и словарями, а
    стратегия — своими типами (`ContextTask`, `StrategyState`). Тип, который
    конструирует `context.py`, должен в `context.py` и жить, иначе модулю
    пришлось бы импортировать `agent.py` — модуль с клиентом API — и лист
    графа перестал бы быть листом.
    """

    name: str

    def describe(self, history: list[dict]) -> StrategyState: ...
    def prepare(self, history: list[dict], question: str) -> ContextTask | None: ...
    def apply(self, task: ContextTask, text: str) -> None: ...
    def build(
        self, system_prompt: str, history: list[dict], question: str
    ) -> list[dict]: ...
    def dump(self) -> dict: ...
    def load(self, data: dict) -> None: ...
    def reset(self) -> None: ...


class FullHistory:
    """«Вся история» — поведение дней 6-8: в модель уходит весь стек.

    Не заглушка, а рабочая стратегия и база для сравнения: `build()`
    возвращает ровно то, что `Agent._build_messages()` возвращал на дне 8,
    и дни 6-8 должны воспроизводиться на ней без оговорок. Памяти у неё нет,
    служебных вызовов она не просит.
    """

    name = FULL_HISTORY_NAME

    def describe(self, history: list[dict]) -> StrategyState:
        return StrategyState(
            name=self.name,
            description="в модель уходит весь стек сообщений целиком, как на днях 6-8",
            memory_label="Память стратегии",
            memory_text="",
            covered=0,
            updated_turns=0,
            note=(
                f"сжатия нет: следующий запрос унесёт весь стек "
                f"({len(history or [])} сообщ.) и будет тем дороже, чем длиннее диалог"
            ),
        )

    def prepare(self, history: list[dict], question: str) -> ContextTask | None:
        return None

    def apply(self, task: ContextTask, text: str) -> None:
        return None

    def build(
        self, system_prompt: str, history: list[dict], question: str
    ) -> list[dict]:
        return (
            [{"role": "system", "content": system_prompt}]
            + [dict(message) for message in history or []]
            + [{"role": "user", "content": question}]
        )

    def dump(self) -> dict:
        return {}

    def load(self, data: dict) -> None:
        return None

    def reset(self) -> None:
        return None


class SummaryStrategy:
    """«Сводка + последние N»: старая часть диалога уходит в модель одной
    сводкой, всё несвёрнутое — как есть.

    Память — это текст сводки и `covered`, индекс в историю агента: сколько
    первых сообщений она покрывает. Сами сообщения стратегия у себя не
    хранит — историю ей передают параметром на каждый вызов.

    Промпт свёртки и числа параметров приходят снаружи (из `presets.py`):
    про Too Many Bones этот модуль не знает. Ограничение длины сводки словами
    живёт внутри самого промпта, поэтому отдельным параметром сюда не приходит.
    """

    def __init__(
        self,
        name: str,
        prompt: str,            # промпт свёртки, приходит из presets.py
        keep_last: int,
        fold_every: int,
        max_tokens: int,
    ) -> None:
        self.name = name
        self._prompt = prompt
        # Чётность `keep_last` (спецификация, §3.1) обеспечивает `presets.py`,
        # здесь она не проверяется: модуль про ходы не знает. Нечётное число
        # ничего не потеряет — хвост просто начнётся с ответа, а не с вопроса.
        self._keep_last = max(0, int(keep_last))
        self._fold_every = max(1, int(fold_every))
        self._max_tokens = max_tokens
        self._summary = ""
        self._covered = 0
        self._updated_turns = 0

    # --- Чистые методы: их зовут на каждый рендер панели -----------------

    def describe(self, history: list[dict]) -> StrategyState:
        history = history or []
        covered = self._clamped_covered(history)
        pending = self._pending(history)
        if pending is not None:
            note = _fold_label(*pending)
        else:
            # Сколько сообщений осталось до свёртки: столько же ходов до
            # момента, который на видео и надо показать.
            left = self._fold_every - (len(history) - covered - self._keep_last)
            if self._summary:
                note = f"сводка есть; до следующей свёртки ещё {left} сообщ."
            else:
                note = f"свёртки ещё не было: до неё ещё {left} сообщ."
        return StrategyState(
            name=self.name,
            description=(
                f"последние {self._keep_last} сообщений уходят как есть, "
                f"свёртка каждые {self._fold_every} сообщений, "
                f"потолок сводки {self._max_tokens} токенов"
            ),
            memory_label="Сводка",
            memory_text=self._summary,
            covered=covered,
            updated_turns=self._updated_turns,
            note=note,
        )

    def prepare(self, history: list[dict], question: str) -> ContextTask | None:
        """Описание работы, а не сама работа: `None` — сворачивать нечего.

        Того же вызова панель спрашивает, чтобы показать «свёртка назрела», —
        строка в панели и то, что произойдёт на следующем ходе, считаются
        одним и тем же кодом. `question` сегодня не используется: триггер
        свёртки — по числу сообщений (§3.1); параметр есть, потому что он
        часть протокола.
        """
        history = history or []
        pending = self._pending(history)
        if pending is None:
            return None
        start, end = pending
        return ContextTask(
            kind="summary",
            label=_fold_label(start, end),
            messages=[
                {"role": "system", "content": self._prompt},
                {"role": "user", "content": self._fold_request(history[start:end])},
            ],
            max_tokens=self._max_tokens,
            covers=end,
        )

    def build(
        self, system_prompt: str, history: list[dict], question: str
    ) -> list[dict]:
        """Что уйдёт в модель: системный промпт, сводка ведущим `system`-
        сообщением и **весь** несвёрнутый хвост истории.

        Хвост — это `history[covered:]`, а не «последние `keep_last`»:
        `keep_last` определяет, сколько остаётся *после* свёртки, а не сколько
        отправляется. Сообщение не должно выпадать из запроса, не попав перед
        этим в сводку (§3.4).
        """
        history = history or []
        covered = self._clamped_covered(history)
        memory: list[dict] = []
        if self._summary:
            memory.append(
                {
                    "role": "system",
                    "content": f"{SUMMARY_HEADER}\n\n{self._summary}",
                }
            )
        return (
            [{"role": "system", "content": system_prompt}]
            + memory
            + [dict(message) for message in history[covered:]]
            + [{"role": "user", "content": question}]
        )

    # --- Единственный метод, который меняет память -----------------------

    def apply(self, task: ContextTask, text: str) -> None:
        """Результат служебного вызова. Зовёт только агент и только после
        успешной свёртки.

        Пустой ответ модели памятью не считается: сводку не сохраняем и
        `covered` не двигаем — свёртки не было, следующий ход попробует снова.
        """
        summary = (text or "").strip()
        if not summary:
            return
        self._summary = summary
        self._covered = max(0, int(task.covers))
        self._updated_turns += 1

    # --- Память на диск и обратно ----------------------------------------

    def dump(self) -> dict:
        return {
            "summary": self._summary,
            "covered": self._covered,
            "updated_turns": self._updated_turns,
        }

    def load(self, data: dict) -> None:
        """Память из файла сессии. Переживает мусор: чужой словарь,
        отсутствующие ключи, неожиданные типы.

        `covered` без сводки — это состояние, в котором сообщения пропали бы
        из запроса молча, ничем не заменённые. Такого не бывает: нет сводки —
        нет и покрытия.
        """
        self.reset()
        if not isinstance(data, dict):
            return
        summary = data.get("summary")
        self._summary = summary.strip() if isinstance(summary, str) else ""
        if not self._summary:
            return
        self._covered = _non_negative_int(data.get("covered"))
        self._updated_turns = _non_negative_int(data.get("updated_turns"))

    def reset(self) -> None:
        """Сброс диалога: сводка удалённого диалога в запрос попасть не может."""
        self._summary = ""
        self._covered = 0
        self._updated_turns = 0

    # --- Внутреннее ------------------------------------------------------

    def _clamped_covered(self, history: list[dict]) -> int:
        """`covered`, зажатый в `0 <= covered <= len(history) - keep_last`.

        Память могла приехать из файла прошлой жизни (диалог сбросили, файл
        остался) или из чужого словаря. Рассыпаться на битой памяти стратегия
        не должна: лишнее покрытие просто игнорируется, и в запрос уходит
        больше истории, а не меньше.
        """
        return min(max(self._covered, 0), max(0, len(history) - self._keep_last))

    def _pending(self, history: list[dict]) -> tuple[int, int] | None:
        """Границы среза, который пора свернуть, или `None`.

        Свёртка назрела, когда несвёрнутого хвоста набралось на `fold_every`
        сверх тех `keep_last`, которые остаются как есть.
        """
        covered = self._clamped_covered(history)
        end = len(history) - self._keep_last
        if end - covered < self._fold_every:
            return None
        return covered, end

    def _fold_request(self, messages: list[dict]) -> str:
        """Сворачиваемые сообщения уходят текстом внутри одного сообщения
        `user`, а не настоящим списком сообщений (§3.3): иначе модель с
        большой вероятностью ответит на последний вопрос игрока вместо того,
        чтобы сворачивать диалог."""
        lines = [
            f"{_ROLE_LABELS.get(message.get('role'), '[реплика]')} "
            f"{message.get('content') or ''}"
            for message in messages
            if isinstance(message, dict)
        ]
        return (
            "Предыдущая сводка:\n"
            f"{self._summary or _NO_PREVIOUS_SUMMARY}\n\n"
            "Новые сообщения диалога (свернуть вместе с предыдущей сводкой):\n"
            + "\n".join(lines)
        )


def _fold_label(start: int, end: int) -> str:
    """Одна строка про назревшую свёртку — и для панели, и для статуса хода,
    и для лога: «назрела» и «произошла» описываются одним и тем же кодом."""
    return (
        f"свёртка назрела: сообщения #{start}-#{end - 1} "
        f"({end - start} шт.) уедут в сводку"
    )


def _non_negative_int(value) -> int:
    """Целое из файла сессии. `True` — тоже `int` в Python, и в счётчик
    сообщений ему попадать незачем."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0
