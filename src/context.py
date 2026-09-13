# TooManyRules — стратегии управления контекстом (день 9, неделя 2; день 10 —
# «Скользящее окно», «Факты + последние N» и правки протокола под общий
# служебный вызов).
#
# Модуль знает только про списки сообщений: ни Gradio, ни Too Many Bones, ни
# вызовов API здесь нет и быть не может. Из проекта не импортируется ничего —
# это второй лист графа зависимостей рядом с `tokens.py`
# (`agent.py` → `tokens.py`, `context.py`), см.
# `docs/TooManyRules — Неделя 2 архитектура.md`, §3, и спецификации дня 9, §4,
# и дня 10, §4.
#
# Главное решение дня 9, в силе и сегодня: стратегия решает не что хранить, а
# что отправлять. Стек сообщений агента и файл сессии всегда полные, память
# стратегии (сводка, факты) живёт отдельным полем рядом с историей, а не
# вместо неё. Отсюда и разделение на две фазы:
#
#   prepare() — «что надо сделать до запроса» (чистая, описывает работу),
#   apply()   — «вот результат» (единственный метод, который меняет память),
#   build()   — «что уходит в модель» (чистая).
#
# `prepare()` и `build()` зовёт дебаг-панель на каждый свой рендер. Если бы
# служебная работа жила внутри `build()`, каждое движение в интерфейсе стоило
# бы денег — поэтому служебный вызов делает агент, а здесь только сборка
# сообщений для него.
#
# Логов у модуля нет, как и у `tokens.py`: чистые функции зовут слишком
# часто. Всё, что нужно видеть в терминале, логирует агент.
#
# День 10 добавляет «Скользящее окно» и «Факты + последние N» — обе стратегии
# без summary — и три правки протокола (спецификация дня 10, §4.1): `apply()`
# теперь возвращает `bool` (применился ли результат — на дне 9 единственным
# неприменимым результатом была пустая сводка, у фактов неприменимым бывает и
# непустой ответ), у `ContextTask` появилось поле `sends`, у `StrategyState` —
# `last_update`. Ветки диалога сюда не входят: это операция над историей
# агента, а не способ собрать запрос, и живёт она в `agent.py`.

import re
from dataclasses import dataclass
from typing import Protocol

# Пометка перед текстом сводки в запросе: модель должна понимать, что читает
# пересказ, а не сообщение собеседника (спецификация дня 9, §3.4). Это про
# механику, а не про настолки, — поэтому константа здесь, а не в `presets.py`.
SUMMARY_HEADER = (
    "Краткая сводка предыдущей части разговора; исходных сообщений в "
    "контексте больше нет."
)

# То же самое для блока фактов (спецификация дня 10, §3.5): факты — тоже
# выжимка, а не сообщение собеседника, и исходных сообщений за ними может уже
# не быть в контексте.
FACTS_HEADER = (
    "Факты из разговора с пользователем в виде «ключ — значение»; это "
    "выжимка, исходных сообщений в контексте может не быть."
)

# Имя стратегии «Вся история» — константа модуля, а не проекта: это поведение
# дней 6-8, и `agent.py` поднимает её сам, когда набор стратегий ему не дали.
FULL_HISTORY_NAME = "Вся история"

# Разметка сворачиваемых сообщений для служебного вызова. Нейтральная: модуль
# не знает ни про игроков, ни про правила. Общая для сводки и фактов — обе
# передают модели куски диалога одним и тем же приёмом (спецификации дня 9,
# §3.3, и дня 10, §3.3).
_ROLE_LABELS = {"user": "[пользователь]", "assistant": "[ассистент]"}
_NO_PREVIOUS_SUMMARY = "предыдущей сводки нет — это первая свёртка"

# Разбор ответа фактов (спецификация дня 10, §3.4): маркеры списков и
# нумерации перед ключом, схлопывание пробелов, распознавание «нет изменений»
# и «удалить» без учёта регистра и завершающей точки. Обрамляющие выделение и
# кавычки срезаются: модель может ответить markdown'ом или повторить
# оформление промпта, и «**тиран**» не должен встать отдельным ключом рядом с
# «тиран» — слияние такой дубль никогда бы не убрало.
_LIST_MARKER_RE = re.compile(r"^(?:[-*•]|\d+[.)])\s*")
_WHITESPACE_RE = re.compile(r"\s+")
_EDGE_MARKUP = "`*«»\""
_NO_CHANGES_TEXT = "нет изменений"
_DELETE_VALUE = "удалить"


@dataclass(frozen=True)
class ContextTask:
    """Служебный вызов модели, который стратегия просит сделать до основного
    запроса. Сам вызов делает агент: в `context.py` сети нет."""

    kind: str                  # "summary" | "facts"
    # Что за работа — существительным, а не будущим временем (день 10, §4.1):
    # «свёртка сообщений #0-#5 (6 шт.) в сводку». Одна и та же строка стоит и
    # в панели («⏳ На следующем ходе: …»), и в статусе хода, и в логе уже
    # состоявшегося вызова — будущее время в логе читалось бы как противоречие.
    label: str
    messages: list[dict]       # готовый список сообщений для служебного вызова
    max_tokens: int | None
    # Граница, а не количество: результат покроет `history[:covers]`, включая
    # то, что уже было учтено раньше. Сколько сообщений покроет именно этот
    # вызов, агент считает сам (`ServiceCall.covers` — уже количество).
    covers: int
    # Сколько сообщений истории уйдёт в запрос, когда результат применится
    # (день 10, §4.1). У сводки — `len(history) - covers`, у фактов —
    # `min(len(history), keep_last)`. Без значения по умолчанию: `ContextTask`
    # создаётся только в `context.py`, и забытое число должно упасть сразу, а
    # не молча показать ноль.
    sends: int


@dataclass(frozen=True)
class StrategyState:
    """Что стратегия помнит и что про неё показывает панель. Панель не знает,
    какие бывают стратегии, — она рисует то, что вернули отсюда."""

    name: str
    # Параметры словами: «последние 6 сообщений, свёртка каждые 6».
    description: str
    memory_label: str          # как называется память в панели: «Сводка», «Факты»
    memory_text: str           # сама память текстом; "" — памяти нет
    # Сколько сообщений истории память уже учла (свёрнуто в сводку, разобрано
    # в факты); 0 — у стратегии без памяти.
    covered: int
    updated_turns: int         # сколько раз память обновлялась за жизнь агента
    note: str                  # что случится на следующем ходе, человеческим языком
    # Чем последнее обновление памяти отчиталось, словами (день 10, §4.1): у
    # фактов «+тиран; ~договорённости», у сводки «сводка переписана, учитывает
    # 12 сообщ.». Живёт только в памяти процесса: в `dump()` не входит, после
    # перезапуска пустое.
    last_update: str = ""


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
    def apply(self, task: ContextTask, text: str) -> bool: ...
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

    def apply(self, task: ContextTask, text: str) -> bool:
        return False

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


class SlidingWindow:
    """«Скользящее окно» (день 10): в модель уходят только последние
    `keep_last` сообщений истории, всё старше — отбрасывается из запроса без
    замены.

    Памяти у стратегии нет вовсе: она ничего не помнит о том, что осталось за
    бортом, и служебных вызовов не просит — единственная плата за это,
    деталями, а не токенами и не временем ответа (спецификация дня 10, §2.4).
    """

    def __init__(self, name: str, keep_last: int) -> None:
        self.name = name
        self._keep_last = max(0, int(keep_last))

    def describe(self, history: list[dict]) -> StrategyState:
        history = history or []
        dropped = max(0, len(history) - self._keep_last)
        if dropped:
            note = (
                f"за бортом {dropped} сообщ. из {len(history)}: отброшены из "
                f"запроса без замены, в стеке и в файле сессии остались"
            )
        else:
            note = (
                f"история помещается целиком: {len(history)} сообщ. из "
                f"{self._keep_last}"
            )
        return StrategyState(
            name=self.name,
            description=(
                f"в модель уходят только последние {self._keep_last} "
                f"сообщений; всё, что старше, отбрасывается из запроса без замены"
            ),
            memory_label="Память стратегии",
            memory_text="",
            covered=0,
            updated_turns=0,
            note=note,
        )

    def prepare(self, history: list[dict], question: str) -> ContextTask | None:
        return None

    def apply(self, task: ContextTask, text: str) -> bool:
        return False

    def build(
        self, system_prompt: str, history: list[dict], question: str
    ) -> list[dict]:
        # `history[-0:]` вернул бы весь список, а не пустой: при `keep_last=0`
        # хвост нужно собрать явным пустым списком.
        history = history or []
        tail = history[-self._keep_last:] if self._keep_last else []
        return (
            [{"role": "system", "content": system_prompt}]
            + [dict(message) for message in tail]
            + [{"role": "user", "content": question}]
        )

    def dump(self) -> dict:
        return {}

    def load(self, data: dict) -> None:
        return None

    def reset(self) -> None:
        return None


class StickyFacts:
    """«Факты + последние N» (день 10): блок «ключ — значение», который
    служебный вызов обновляет перед каждым ответом, плюс хвост истории.

    Память — упорядоченный словарь `ключ → значение`, `covered` (сколько
    сообщений истории учтено) и `updated_turns`. Сами сообщения стратегия у
    себя не хранит, как и сводка (`SummaryStrategy`) — историю ей передают
    параметром на каждый вызов.

    Промпт извлечения фактов и числа параметров приходят снаружи (из
    `presets.py`): про Too Many Bones этот модуль не знает. Лимиты числа
    фактов и длины значения живут внутри самого промпта, поэтому отдельными
    параметрами сюда не приходят (как `SUMMARY_WORDS` у сводки).
    """

    def __init__(
        self,
        name: str,
        prompt: str,             # промпт фактов, приходит из presets.py
        keep_last: int,
        max_tokens: int,
    ) -> None:
        self.name = name
        self._prompt = prompt
        self._keep_last = max(0, int(keep_last))
        self._max_tokens = max_tokens
        self._facts: dict[str, str] = {}
        self._covered = 0
        self._updated_turns = 0
        self._last_update = ""

    # --- Чистые методы: их зовут на каждый рендер панели -----------------

    def describe(self, history: list[dict]) -> StrategyState:
        history = history or []
        covered = self._clamped_covered(history)
        count = len(self._facts)
        if count:
            note = f"фактов {count}; обновляются перед каждым ответом"
        else:
            pending = len(history) - covered
            note = (
                f"фактов пока нет: перед первым ответом служебный вызов "
                f"прочитает {pending} сообщ."
                if pending
                else "фактов пока нет: обновятся перед первым ответом"
            )
        return StrategyState(
            name=self.name,
            description=(
                f"факты «ключ — значение» + последние {self._keep_last} "
                f"сообщений; факты обновляются служебным вызовом перед каждым "
                f"ответом, потолок {self._max_tokens} токенов"
            ),
            memory_label="Факты",
            memory_text=self._facts_text(),
            covered=covered,
            updated_turns=self._updated_turns,
            note=note,
            last_update=self._last_update,
        )

    def prepare(self, history: list[dict], question: str) -> ContextTask | None:
        """Задача на обновление фактов — почти всегда, пока не учтено
        абсолютно всё и вопроса нет (спецификация дня 10, §4.4): в штатном
        режиме неучтённым остаётся последний обмен, и назревшая задача здесь —
        нормальное состояние стратегии, а не предупреждение."""
        history = history or []
        question = question or ""
        covered = self._clamped_covered(history)
        new_messages = history[covered:]
        if not new_messages and not question:
            return None
        return ContextTask(
            kind="facts",
            label=_facts_label(covered, len(history), bool(question)),
            messages=[
                {"role": "system", "content": self._prompt},
                {
                    "role": "user",
                    "content": self._facts_request(new_messages, question),
                },
            ],
            max_tokens=self._max_tokens,
            covers=len(history),
            sends=min(len(history), self._keep_last),
        )

    def build(
        self, system_prompt: str, history: list[dict], question: str
    ) -> list[dict]:
        """Системный промпт, блок фактов ведущим `system`-сообщением (если
        факты есть) и хвост — последние `keep_last` сообщений, но не короче
        неучтённого (спецификация дня 10, §3.5): если обновления фактов
        несколько ходов подряд падали, в запрос уходит всё, что в факты ещё не
        попало, — деградация в сторону «без сжатия», а не в сторону потери."""
        history = history or []
        covered = self._clamped_covered(history)
        start = min(covered, len(history) - self._keep_last)
        start = max(0, min(start, len(history)))
        memory: list[dict] = []
        if self._facts:
            memory.append(
                {
                    "role": "system",
                    "content": f"{FACTS_HEADER}\n\n{self._facts_block()}",
                }
            )
        return (
            [{"role": "system", "content": system_prompt}]
            + memory
            + [dict(message) for message in history[start:]]
            + [{"role": "user", "content": question}]
        )

    # --- Единственный метод, который меняет память -----------------------

    def apply(self, task: ContextTask, text: str) -> bool:
        """Разбор и слияние ответа служебного вызова (спецификация дня 10,
        §3.4). Возвращает, применился ли результат: `apply()` — единственный,
        кто знает, разобралась ли хоть одна строка.

        Слияние, а не замена: новые ключи добавляются в конец, изменившиеся
        заменяют значение на месте, остальные остаются как были — это и
        делает факты «липкими». Сливается копия, и факты подменяются целиком
        в конце: панель соседней вкладки читает их без замков, и словарь не
        должен меняться у неё под руками (спецификация дня 10, §4.4).
        """
        facts = dict(self._facts)
        changes: list[str] = []
        understood = False
        for raw_line in (text or "").splitlines():
            line = _unwrap(raw_line)
            if not line:
                continue
            # «Нет изменений» — и одной строкой, и с пояснением через
            # двоеточие или тире, и пунктом списка: ключом такая строка не
            # становится.
            bare = _unwrap(_LIST_MARKER_RE.sub("", line, count=1))
            if bare.casefold().startswith(_NO_CHANGES_TEXT):
                understood = True
                continue
            if ":" not in line:
                continue
            raw_key, raw_value = line.split(":", 1)
            key = _LIST_MARKER_RE.sub("", raw_key.strip(), count=1)
            key = _WHITESPACE_RE.sub(" ", _unwrap(key)).lower()
            value = _unwrap(raw_value)
            if not key or not value:
                continue
            understood = True
            if _strip_trailing_dot(value).casefold() == _DELETE_VALUE:
                if key in facts:
                    del facts[key]
                    changes.append(f"−{key}")
                continue  # удаление несуществующего ключа — не ошибка
            if facts.get(key) == value:
                continue  # повтор факта слово в слово — не изменение
            marker = "~" if key in facts else "+"
            facts[key] = value
            changes.append(f"{marker}{key}")

        if not understood:
            return False
        self._facts = facts
        self._covered = max(0, int(task.covers))
        self._updated_turns += 1
        self._last_update = "; ".join(changes) if changes else "без изменений"
        return True

    # --- Память на диск и обратно ----------------------------------------

    def dump(self) -> dict:
        if not self._facts:
            return {}
        return {
            "facts": dict(self._facts),
            "covered": self._covered,
            "updated_turns": self._updated_turns,
        }

    def load(self, data: dict) -> None:
        """Память из файла сессии. В факты попадают только пары «непустая
        строка → непустая строка», ключи берутся как есть, без нормализации.
        Нет ни одного факта — нет и покрытия: покрытие без фактов на диск не
        пишется, и отличить его от мусора было бы нельзя."""
        self.reset()
        if not isinstance(data, dict):
            return
        facts = data.get("facts")
        if isinstance(facts, dict):
            for key, value in facts.items():
                if isinstance(key, str) and key and isinstance(value, str) and value:
                    self._facts[key] = value
        if not self._facts:
            return
        self._covered = _non_negative_int(data.get("covered"))
        self._updated_turns = _non_negative_int(data.get("updated_turns"))

    def reset(self) -> None:
        self._facts = {}
        self._covered = 0
        self._updated_turns = 0
        self._last_update = ""

    # --- Внутреннее ------------------------------------------------------

    def _clamped_covered(self, history: list[dict]) -> int:
        """`covered`, зажатый в `0 <= covered <= len(history)` — шире, чем у
        сводки: факты могут учесть всю историю без остатка, хвост для запроса
        считается отдельно, в `build()`."""
        return min(max(self._covered, 0), len(history))

    def _facts_text(self) -> str:
        """Факты строками `ключ: значение», в порядке словаря — для панели."""
        return "\n".join(f"{key}: {value}" for key, value in self._facts.items())

    def _facts_block(self) -> str:
        """Факты строками со маркером списка — для блока в основном запросе."""
        return "\n".join(f"- {key}: {value}" for key, value in self._facts.items())

    def _facts_request(self, new_messages: list[dict], question: str) -> str:
        """Вход служебного вызова текстом внутри одного сообщения `user`, тем
        же приёмом, что у свёртки (спецификация дня 10, §3.3): иначе модель с
        большой вероятностью ответит на вопрос вместо того, чтобы вести факты."""
        parts = ["Текущие факты:", self._facts_text() or "фактов пока нет"]
        if new_messages:
            parts += [
                "",
                "Новые сообщения диалога с прошлого обновления:",
                *[
                    f"{_ROLE_LABELS.get(message.get('role'), '[реплика]')} "
                    f"{message.get('content') or ''}"
                    for message in new_messages
                    if isinstance(message, dict)
                ],
            ]
        if question:
            parts += [
                "",
                "Новое сообщение пользователя (ассистент на него ещё не ответил):",
                question,
            ]
        return "\n".join(parts)


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
        self._last_update = ""

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
            last_update=self._last_update,
        )

    def prepare(self, history: list[dict], question: str) -> ContextTask | None:
        """Описание работы, а не сама работа: `None` — сворачивать нечего.

        Того же вызова панель спрашивает, чтобы показать назревшую свёртку, —
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
            sends=len(history) - end,
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

    def apply(self, task: ContextTask, text: str) -> bool:
        """Результат служебного вызова. Зовёт только агент и только после
        ответа модели.

        Пустой ответ модели памятью не считается: сводку не сохраняем и
        `covered` не двигаем — свёртки не было, следующий ход попробует снова.
        """
        summary = (text or "").strip()
        if not summary:
            return False
        self._summary = summary
        self._covered = max(0, int(task.covers))
        self._updated_turns += 1
        self._last_update = f"сводка переписана, учитывает {self._covered} сообщ."
        return True

    # --- Память на диск и обратно ----------------------------------------

    def dump(self) -> dict:
        if not self._summary:
            return {}
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
        self._last_update = ""

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


def _facts_label(covered: int, total: int, has_question: bool) -> str:
    """Строка про обновление фактов — назревшее или уже состоявшееся, одна и
    та же что в панели, что в статусе хода, что в логе (спецификация дня 10,
    §4.1). Новое сообщение упоминается, только если вопрос непустой: без
    вопроса (превью в панели) в задаче участвуют только неучтённые сообщения."""
    bits: list[str] = []
    if covered < total:
        bits.append(f"сообщениям #{covered}-#{total - 1} ({total - covered} шт.)")
    if has_question:
        bits.append("новому сообщению")
    if not bits:
        return "обновление фактов"
    return f"обновление фактов по {' и '.join(bits)}"


def _strip_trailing_dot(text: str) -> str:
    text = text.strip()
    return text[:-1].strip() if text.endswith(".") else text


def _unwrap(text: str) -> str:
    """Строка без обрамляющих пробелов, выделения и кавычек: «`тиран: Nom`»,
    «**тиран**» и «"Nom"» разбираются так же, как «тиран: Nom»."""
    return text.strip().strip(_EDGE_MARKUP).strip()


def _fold_label(start: int, end: int) -> str:
    """Строка про свёртку — существительным, а не будущим временем: она стоит
    и в панели («⏳ На следующем ходе: …»), и в статусе хода, и в логе уже
    состоявшегося вызова (спецификация дня 10, §4.1) — будущее время в логе
    завершённого вызова читалось бы как противоречие."""
    return f"свёртка сообщений #{start}-#{end - 1} ({end - start} шт.) в сводку"


def _non_negative_int(value) -> int:
    """Целое из файла сессии. `True` — тоже `int` в Python, и в счётчик
    сообщений ему попадать незачем."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0
