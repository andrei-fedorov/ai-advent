# TooManyRules — модель памяти агента (день 11, неделя 3).
#
# Память агента разведена по времени жизни на три слоя (спецификация дня 11,
# §2.1):
#
#   краткосрочная  — разговор этой сессии: стек сообщений агента и память
#                    стратегий дней 9-10. Здесь её нет — она уже живёт в
#                    агенте и в `context.py`, и сводка с фактами не отдельный
#                    слой, а способ отправить разговор короче;
#   рабочая        — данные текущей задачи: живёт столько же, сколько сессия,
#                    пишется разбором памяти сразу, без подтверждения;
#   долговременная — сведения о пользователе, его решения и знания: от сессий
#                    и агентов не зависит, и запись в неё — только решение
#                    человека. Разбор предлагает **кандидата**, а не запись.
#
# Что в каком слое, решает не модель, а карта памяти (`MemorySlot`) —
# закрытый список ключей, написанный человеком (`presets.MEMORY_MAP`). Модель
# выбирает только ключ, слой ключа берётся из карты, ключ вне карты
# отклоняется (§2.3).
#
# Модуль знает только про слои, карту и строки «ключ: значение»: ни сети, ни
# диска, ни логов, ни Gradio, ни Too Many Bones. Карта, промпт разбора и числа
# приходят из `presets.py`; долговременную память модуль получает параметром —
# словарём «ключ → значение» или записями хранилища, а читает и пишет её файл
# `storage.py` по команде агента. Всё, что нужно видеть в терминале, логирует
# агент.
#
# Из проекта импортируется только `context.py` — ради общего разбора ответа
# служебного вызова (`parse_changes()`, `normalize_key()`, спецификация дня 11,
# §4.5): правила разбора выстраданы ревью дня 10, и вторая копия разошлась бы
# с первой на первой же правке. `context.py` при этом остаётся листом графа:
# presets.py → agent.py → memory.py → context.py, и presets.py → memory.py.
#
# Две фазы, как у стратегий: `describe()`, `prepare()` и сборка блоков —
# чистые (их зовёт панель на каждый рендер), а рабочую память и кандидатов
# меняют только `apply()`, `take_candidates()`, `drop_candidates()`, `load()` и
# `reset()` — их зовёт только агент под своим замком. Состояние подменяется
# копиями целиком: панель соседней вкладки читает его без замков.
#
# Отдельный модуль, а не ещё один класс в `context.py`: стратегия отвечает на
# вопрос «что из разговора отправить» и выбирается одна, модель памяти — на
# вопрос «что и где агент знает помимо разговора», и слои работают вместе с
# любой стратегией.

from collections.abc import Sequence
from dataclasses import dataclass, replace

import context

# --- Слои ------------------------------------------------------------------

LAYER_SHORT_TERM = "краткосрочная"
LAYER_WORKING = "рабочая"
LAYER_LONG_TERM = "долговременная"
# Слои, которые переключаются в запросе, — в порядке их блоков в запросе
# (§3.5): от самого устойчивого к самому изменчивому. Краткосрочный слой
# переключателем не выключается: что из разговора отправить, решает стратегия.
REQUEST_LAYERS = (LAYER_LONG_TERM, LAYER_WORKING)

# Слои, которые бывают у ключа карты. Краткосрочного среди них нет: разговор
# пишется каждым ходом целиком, выбирать в нём нечего.
_MAP_LAYERS = (LAYER_WORKING, LAYER_LONG_TERM)

# Заголовки блоков слоёв в основном запросе (§3.5). Нейтральные — про
# пользователя и задачу, а не про игрока и партию: модуль про настолки не
# знает. Правила приоритета в них не украшение: профиль говорит «только база»,
# а сегодня друг принёс дополнение — ответ должен идти от сегодняшнего.
#
# Заголовки — ещё и метка для счёта токенов: `Agent._count_messages()`
# узнаёт блоки слоёв по началу `system`-сообщения, поэтому оба заголовка не
# должны начинаться одинаково ни друг с другом, ни с заголовками сводки и
# фактов из `context.py`.
LONG_TERM_HEADER = (
    "Долговременная память: сведения о пользователе, его решения и знания из "
    "прошлых разговоров, которые он подтвердил для будущих разговоров. Если "
    "текущий разговор им противоречит, верен текущий разговор."
)
WORKING_HEADER = (
    "Рабочая память: данные текущей задачи из этого разговора; исходных "
    "сообщений в контексте может уже не быть. Для текущей задачи рабочая "
    "память важнее долговременной."
)

# Разметка входа служебного вызова (§3.4) — та же, что у свёртки и фактов в
# `context.py`: модель видит куски диалога одним и тем же приёмом.
_ROLE_LABELS = {"user": "[пользователь]", "assistant": "[ассистент]"}
_EMPTY_MEMORY = "пусто"
_NO_CANDIDATES = "нет"
# Как кандидат на удаление выглядит во входе разбора: тем же словом, которым
# модель его предлагает (формат ответа дня 10).
_DELETE_WORD = "удалить"
_UNUSED_HEADER = "В файле, не используется — ключей нет в долговременной части карты:"


@dataclass(frozen=True)
class MemorySlot:
    """Строка карты памяти."""

    key: str                 # как в ответе модели: нижний регистр, пробелы схлопнуты
    layer: str               # LAYER_WORKING | LAYER_LONG_TERM
    section: str             # «партия», «профиль», «решения», «знания»
    description: str         # что в ключе хранится — уходит в промпт разбора


@dataclass(frozen=True)
class MemoryTask:
    """Служебный вызов «разбор памяти». Сам вызов делает агент."""

    # Существительным, как `ContextTask.label` дня 10: одна и та же строка
    # стоит и в логе состоявшегося вызова, и в панели, и в статусе хода.
    label: str               # «разбор памяти по сообщениям #4-#5 (2 шт.) и новому сообщению»
    messages: list[dict]
    max_tokens: int | None
    covers: int              # граница: результат учтёт history[:covers]


@dataclass(frozen=True)
class Candidate:
    """Запись, которую разбор отнёс к долговременной памяти и которая ждёт
    решения человека. Не память, а вопрос к человеку (§2.4): в запрос не
    уходит, на диск не пишется, в ветку не копируется и перезапуск процесса
    не переживает."""

    key: str
    value: str | None        # None — предложено удалить ключ
    previous: str | None     # значение в долговременной памяти на момент предложения; None — ключа не было
    section: str


@dataclass(frozen=True)
class MemoryState:
    """Что про модель памяти показывает панель. Панель не знает, какие ключи
    бывают, — рисует это."""

    working_text: str        # строки «ключ: значение» рабочей памяти, в порядке карты; "" — пусто
    working_items: int
    covered: int             # сколько сообщений истории разбор уже учёл
    updated_turns: int       # сколько разборов применилось за жизнь агента
    last_update: str         # что изменил последний применённый разбор, словами; "" — в этом процессе не было
    candidates: list[Candidate]
    rejected: list[str]      # ключи вне карты из последнего применённого разбора


# --- Долговременный блок (§4.2) --------------------------------------------
# `entries` — то, что отдаёт хранилище долговременной памяти (`storage.py`):
# «ключ → {"value", "session_id", "updated_at"}». Все три функции чистые и
# переживают мусор: не словарь, запись не словарь, пустое значение — строка
# пропускается.

def long_term_values(entries: dict, slots: Sequence[MemorySlot]) -> dict[str, str]:
    """«Ключ → значение» только для ключей долговременной части карты, в
    порядке карты. Запись с ключом вне карты (карту поправили, а файл
    остался) сюда не попадает: в запрос и во вход разбора уходит только то,
    что карта считает долговременным."""
    stored = _clean_entries(entries)
    return {
        slot.key: stored[slot.key]["value"]
        for slot in slots
        if slot.layer == LAYER_LONG_TERM and slot.key in stored
    }


def long_term_block(entries: dict, slots: Sequence[MemorySlot]) -> dict | None:
    """Блок долговременной памяти для запроса (§3.5): заголовок и строки
    `- ключ: значение`, сгруппированные подзаголовками разделов карты; пустой
    раздел не выводится. `None` — значений нет, и блока в запросе не будет."""
    values = long_term_values(entries, slots)
    if not values:
        return None
    return {
        "role": "system",
        "content": f"{LONG_TERM_HEADER}\n\n{_sections_text(values, {}, slots)}",
    }


def long_term_text(entries: dict, slots: Sequence[MemorySlot]) -> str:
    """То же, что блок, но для панели: у каждой строки происхождение —
    сессия и время, когда человек запись сохранил; записи вне долговременной
    части карты — отдельно, с пометкой, что они лежат в файле, но не
    используются. "" — записей нет."""
    stored = _clean_entries(entries)
    values = long_term_values(stored, slots)
    parts = []
    text = _sections_text(values, stored, slots)
    if text:
        parts.append(text)
    known = {slot.key for slot in slots if slot.layer == LAYER_LONG_TERM}
    unused = [
        f"- {key}: {entry['value']}{_origin(entry)}"
        for key, entry in stored.items()
        if key not in known
    ]
    if unused:
        parts.append("\n".join([_UNUSED_HEADER, *unused]))
    return "\n\n".join(parts)


class AgentMemory:
    """Модель памяти одного агента: рабочая память сессии, кандидаты в
    долговременную и разбор ответа служебного вызова по карте.

    Память — упорядоченный словарь рабочих записей «ключ → значение» (в
    порядке карты), `covered`, `updated_turns`, `last_update`, упорядоченный
    словарь кандидатов «ключ → `Candidate`» и список ключей вне карты из
    последнего разбора. Сами сообщения модуль у себя не хранит — историю ему
    передают параметром, как стратегиям.

    Экземпляр свой у каждого агента (его раздаёт `presets.make_memory()`):
    рабочая память и кандидаты относятся к конкретной сессии.
    """

    def __init__(
        self,
        slots: Sequence[MemorySlot],
        prompt: str,              # промпт разбора, приходит из presets.py
        max_tokens: int,
    ) -> None:
        # Карта — код, и ошибку в ней надо видеть до первого вопроса, а не на
        # разборе ответа (§4.3): конструктор проверяет её сразу. Ключи
        # нормализуются тем же правилом, что ключи ответа модели, — иначе
        # «Состав партии» в карте никогда не совпал бы с «состав партии» в
        # ответе.
        normalized: list[MemorySlot] = []
        seen: set[str] = set()
        for slot in slots:
            key = context.normalize_key(slot.key)
            if not key:
                raise ValueError(
                    f"карта памяти: пустой ключ в строке {slot!r}"
                )
            if key in seen:
                raise ValueError(f"карта памяти: ключ «{key}» повторяется")
            if slot.layer not in _MAP_LAYERS:
                raise ValueError(
                    f"карта памяти: у ключа «{key}» слой «{slot.layer}» — "
                    f"допустимы только «{LAYER_WORKING}» и «{LAYER_LONG_TERM}»"
                )
            seen.add(key)
            normalized.append(replace(slot, key=key))
        self._slots: tuple[MemorySlot, ...] = tuple(normalized)
        self._by_key = {slot.key: slot for slot in self._slots}
        self._prompt = prompt
        self._max_tokens = max_tokens
        self._working: dict[str, str] = {}
        self._covered = 0
        self._updated_turns = 0
        self._last_update = ""
        self._candidates: dict[str, Candidate] = {}
        self._rejected: list[str] = []

    @property
    def slots(self) -> tuple[MemorySlot, ...]:
        """Карта с нормализованными ключами — её агент передаёт функциям
        долговременного блока."""
        return self._slots

    def layer_of(self, key: str) -> str | None:
        """Слой ключа по карте; `None` — ключа в карте нет."""
        slot = self._by_key.get(context.normalize_key(key))
        return slot.layer if slot is not None else None

    # --- Чистые методы: их зовут на каждый рендер панели -----------------

    def describe(self, history: list[dict]) -> MemoryState:
        working = self._working
        return MemoryState(
            working_text=_lines(working, marker=""),
            working_items=len(working),
            covered=self._clamped_covered(history or []),
            updated_turns=self._updated_turns,
            last_update=self._last_update,
            candidates=list(self._candidates.values()),
            rejected=list(self._rejected),
        )

    def prepare(
        self, history: list[dict], question: str, long_term: dict[str, str]
    ) -> MemoryTask | None:
        """Задача на разбор памяти, если есть неучтённые сообщения
        `history[covered:]` или непустой вопрос; `None` — только когда учтено
        всё и вопроса нет. Как у фактов дня 10, в штатном режиме задача есть
        почти всегда: неучтённым остаётся последний обмен — без ответа
        ассистента реплика «да, берём» не значит ничего.

        `long_term` — значения долговременной части карты
        (`long_term_values()`): их модель видит, чтобы не предлагать то, что
        уже сохранено.
        """
        history = history or []
        question = question or ""
        covered = self._clamped_covered(history)
        new_messages = history[covered:]
        if not new_messages and not question:
            return None
        return MemoryTask(
            label=_task_label(covered, len(history), bool(question)),
            messages=[
                {"role": "system", "content": self._prompt},
                {
                    "role": "user",
                    "content": self._request(new_messages, question, long_term),
                },
            ],
            max_tokens=self._max_tokens,
            covers=len(history),
        )

    def working_block(self) -> dict | None:
        """Блок рабочей памяти для запроса: заголовок и строки
        `- ключ: значение` в порядке карты; `None` — рабочая память пуста.
        Порядок карты, а не порядок записи: блок, чьи строки переставляются
        от хода к ходу, сбивал бы кэш промпта без причины (§4.3)."""
        working = self._working
        if not working:
            return None
        return {
            "role": "system",
            "content": f"{WORKING_HEADER}\n\n{_lines(working, marker='- ')}",
        }

    # --- Меняют состояние: зовёт только агент под своим замком -----------

    def apply(self, task: MemoryTask, text: str, long_term: dict[str, str]) -> bool:
        """Раскладка ответа служебного вызова по карте (§4.4). Возвращает,
        применился ли разбор.

        Каждая пара «ключ, значение» идёт по карте в порядке строк ответа:
        ключ вне карты отклоняется; рабочий ключ сливается в рабочую память
        по правилам фактов дня 10; долговременный становится кандидатом, а не
        записью, — записать его может только человек.

        Разбор, в котором разобрались только ключи вне карты, —
        применившийся: модель ответила по формату, а карта отказалась. Иначе
        такой ответ повторялся бы каждый ход за деньги с тем же итогом.
        """
        parsed = context.parse_changes(text)
        if parsed is None:
            return False

        stored = _clean_values(long_term)
        working = dict(self._working)
        candidates = dict(self._candidates)
        working_changes: list[str] = []
        proposed: list[str] = []
        rejected: list[str] = []
        for key, value in parsed:
            slot = self._by_key.get(key)
            if slot is None:
                # Решение карты, а не сбой: в рабочую память такой ключ «на
                # всякий случай» не попадает.
                if key not in rejected:
                    rejected.append(key)
                continue

            if slot.layer == LAYER_WORKING:
                if value is None:
                    if key in working:
                        del working[key]
                        working_changes.append(f"−{key}")
                    continue  # удаление несуществующего ключа — не ошибка
                if working.get(key) == value:
                    continue  # повтор значения слово в слово — не изменение
                working_changes.append(f"{'~' if key in working else '+'}{key}")
                working[key] = value
                continue

            # Долговременный ключ — кандидат. Предлагать нечего, если значение
            # уже лежит в долговременной памяти (или предложено удалить то,
            # чего там нет): тогда и ждущий кандидат по ключу больше не нужен.
            previous = stored.get(key)
            if value == previous:
                candidates.pop(key, None)
                continue
            candidate = Candidate(
                key=key, value=value, previous=previous, section=slot.section
            )
            if candidates.get(key) == candidate:
                continue  # тот же кандидат ещё ждёт решения — не изменение
            # Заменяет ждущего кандидата по тому же ключу: человек решает про
            # последнее сказанное, а не про историю предложений.
            candidates[key] = candidate
            if key not in proposed:
                proposed.append(key)

        self._working = self._in_map_order(working)
        self._candidates = candidates
        self._rejected = rejected
        self._covered = max(0, int(task.covers))
        self._updated_turns += 1
        self._last_update = _update_label(
            working_changes, [key for key in proposed if key in candidates], rejected
        )
        return True

    def take_candidates(self, keys: Sequence[str]) -> list[Candidate]:
        """Убирает кандидатов с этими ключами и возвращает их — агент пишет их
        в долговременную память (и зовёт это только после успешной записи).
        Ключи, которых среди кандидатов нет, — не ошибка."""
        return self._remove_candidates(keys)

    def drop_candidates(self, keys: Sequence[str]) -> list[Candidate]:
        """Убирает и возвращает кандидатов с этими ключами — отклонение. В
        память ничего не пишется; сказанное остаётся в истории разговора."""
        return self._remove_candidates(keys)

    def dump(self) -> dict:
        """Рабочая память для `context.working` файла сессии; `{}` — нет ни
        записей, ни покрытия.

        В отличие от фактов дня 10, покрытие без записей пишется: разбор,
        перечитавший историю после перезапуска, заново предложил бы
        кандидатов, которые человек уже сохранил или отклонил. Кандидатов в
        выгрузке нет никогда."""
        if not self._working and not self._covered:
            return {}
        return {
            "entries": dict(self._working),
            "covered": self._covered,
            "updated_turns": self._updated_turns,
        }

    def load(self, data: dict) -> None:
        """Рабочая память из файла сессии или из снимка checkpoint'а. Переживает
        мусор по правилам дня 10: не словарь, чужие типы, пустые строки. В
        рабочую память попадают только пары «непустая строка → непустая
        строка» с ключом **рабочей** части карты; остальное пропускается —
        что прочитано не всё, агент узнаёт сам, сравнив `dump()` с
        прочитанным."""
        self.reset()
        if not isinstance(data, dict):
            return
        entries = data.get("entries")
        working: dict[str, str] = {}
        if isinstance(entries, dict):
            for key, value in entries.items():
                slot = self._by_key.get(key) if isinstance(key, str) else None
                if slot is None or slot.layer != LAYER_WORKING:
                    continue
                if isinstance(value, str) and value.strip():
                    working[key] = value
        self._working = self._in_map_order(working)
        self._covered = _non_negative_int(data.get("covered"))
        self._updated_turns = _non_negative_int(data.get("updated_turns"))

    def reset(self) -> None:
        """Сброс диалога: рабочая память, покрытие, счётчик, последнее
        изменение, кандидаты и ключи вне карты пришли из диалога, которого
        больше нет. Долговременную память модуль не хранит — и не трогает."""
        self._working = {}
        self._covered = 0
        self._updated_turns = 0
        self._last_update = ""
        self._candidates = {}
        self._rejected = []

    # --- Внутреннее ------------------------------------------------------

    def _clamped_covered(self, history: list[dict]) -> int:
        """`covered`, зажатый в `0 <= covered <= len(history)`: память могла
        приехать из файла прошлой жизни или из снимка чужой длины."""
        return min(max(self._covered, 0), len(history))

    def _in_map_order(self, working: dict[str, str]) -> dict[str, str]:
        return {
            slot.key: working[slot.key] for slot in self._slots if slot.key in working
        }

    def _remove_candidates(self, keys: Sequence[str]) -> list[Candidate]:
        wanted = {
            context.normalize_key(key) for key in keys or () if isinstance(key, str)
        }
        removed = [c for key, c in self._candidates.items() if key in wanted]
        if removed:
            self._candidates = {
                key: c for key, c in self._candidates.items() if key not in wanted
            }
        return removed

    def _request(
        self, new_messages: list[dict], question: str, long_term: dict[str, str]
    ) -> str:
        """Вход разбора текстом внутри одного сообщения `user` (§3.4), тем же
        приёмом, что у свёртки и фактов: иначе модель ответит на вопрос
        пользователя вместо того, чтобы разбирать память. Обе памяти уходят
        целиком: значение пишется со всеми прежними пунктами, и модели нужно
        видеть, что в ключе уже лежит."""
        candidates = "\n".join(
            f"{c.key}: {c.value if c.value is not None else _DELETE_WORD}"
            for c in self._candidates.values()
        )
        parts = [
            "Рабочая память (текущая задача):",
            _lines(self._working, marker="") or _EMPTY_MEMORY,
            "",
            "Долговременная память (подтверждена пользователем):",
            _lines(_clean_values(long_term), marker="") or _EMPTY_MEMORY,
            "",
            "Ждут подтверждения пользователя:",
            candidates or _NO_CANDIDATES,
        ]
        if new_messages:
            parts += [
                "",
                "Новые сообщения диалога с прошлого разбора:",
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


def _lines(values: dict[str, str], marker: str) -> str:
    return "\n".join(f"{marker}{key}: {value}" for key, value in values.items())


def _sections_text(
    values: dict[str, str], origins: dict, slots: Sequence[MemorySlot]
) -> str:
    """Строки долговременной части карты, сгруппированные по разделам в
    порядке карты; пустой раздел не выводится. С `origins` (записи
    хранилища) у строк появляется происхождение — это вид для панели."""
    sections: dict[str, list[str]] = {}
    for slot in slots:
        if slot.layer != LAYER_LONG_TERM:
            continue
        lines = sections.setdefault(slot.section, [])
        if slot.key in values:
            origin = _origin(origins[slot.key]) if slot.key in origins else ""
            lines.append(f"- {slot.key}: {values[slot.key]}{origin}")
    # Пустая строка между разделами обязательна: без неё в markdown панели
    # подзаголовок следующего раздела приклеился бы к последнему пункту
    # предыдущего.
    return "\n\n".join(
        "\n".join([f"{section[:1].upper()}{section[1:]}:", *lines])
        for section, lines in sections.items()
        if lines
    )


def _origin(entry: dict) -> str:
    """Происхождение записи для панели: « · s002 · 2026-09-15 19:40:12»."""
    bits = []
    session_id = entry.get("session_id")
    if isinstance(session_id, str) and session_id:
        bits.append(session_id)
    updated_at = entry.get("updated_at")
    if isinstance(updated_at, str) and updated_at:
        bits.append(updated_at.replace("T", " "))
    return "".join(f" · {bit}" for bit in bits)


def _clean_entries(entries: object) -> dict[str, dict]:
    """Записи хранилища, у которых есть что показать: словарь с непустой
    строкой в `value`. Остальное пропускается молча — логов у модуля нет,
    битые записи отмечает хранилище при чтении файла."""
    if not isinstance(entries, dict):
        return {}
    return {
        key: entry
        for key, entry in entries.items()
        if isinstance(key, str) and key
        and isinstance(entry, dict)
        and isinstance(entry.get("value"), str) and entry["value"].strip()
    }


def _clean_values(values: object) -> dict[str, str]:
    """Значения долговременной памяти, пришедшие параметром: только пары
    «непустая строка → непустая строка»."""
    if not isinstance(values, dict):
        return {}
    return {
        key: value
        for key, value in values.items()
        if isinstance(key, str) and key and isinstance(value, str) and value.strip()
    }


def _task_label(covered: int, total: int, has_question: bool) -> str:
    """Строка про разбор памяти — существительным, как у фактов дня 10: она
    стоит и в логе состоявшегося вызова, и в панели, и в статусе хода. Новое
    сообщение упоминается, только если вопрос непустой."""
    bits: list[str] = []
    if covered < total:
        bits.append(f"сообщениям #{covered}-#{total - 1} ({total - covered} шт.)")
    if has_question:
        bits.append("новому сообщению")
    if not bits:
        return "разбор памяти"
    return f"разбор памяти по {' и '.join(bits)}"


def _update_label(
    working_changes: list[str], proposed: list[str], rejected: list[str]
) -> str:
    """Что изменил применённый разбор, словами (§4.4). Все три части —
    всегда, если изменилось хоть что-то: по строке в логе видно и что не
    изменилось; ничего — «без изменений»."""
    if not (working_changes or proposed or rejected):
        return "без изменений"
    return "; ".join(
        [
            f"рабочая: {', '.join(working_changes) if working_changes else 'без изменений'}",
            f"ждут подтверждения: {', '.join(proposed) if proposed else 'нет'}",
            f"вне карты: {', '.join(rejected) if rejected else 'нет'}",
        ]
    )


def _non_negative_int(value) -> int:
    """Целое из файла сессии. `True` — тоже `int` в Python, и в счётчик
    сообщений ему попадать незачем."""
    if isinstance(value, int) and not isinstance(value, bool) and value >= 0:
        return value
    return 0
