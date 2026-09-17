# TooManyRules — профиль пользователя и выбор режима ответа (день 12,
# неделя 3).
#
# Профиль — не слой памяти и не пресет (спецификация дня 12, §2.1): это
# явные настройки, которые пользователь пишет сам в редакторе, — общая часть
# «обо мне» и режимы под задачи. Профиль уходит в каждый запрос, а режим на
# конкретный ход выбирает роутер (служебный вызов модели) или сам
# пользователь переключателем. Модель профиль не пишет и не предлагает
# правок — в этом отличие от долговременной памяти дня 11, где записи
# предлагает модель, а подтверждает человек.
#
# Модуль знает только про профиль и выбор режима: ни сети, ни диска, ни
# логов, ни Gradio, ни Too Many Bones. Профиль по умолчанию, промпт роутера и
# числа приходят из `presets.py`; сам профиль модуль получает словарём (как
# его хранит `storage.py`) или объектом `UserProfile`. Всё, что нужно видеть
# в терминале, логирует агент.
#
# Из проекта импортируется только `context.py` — ради `normalize_key()` при
# сравнении названий режимов и разборе ответа роутера: правила нормализации
# выстраданы днями 10-11, и вторая копия разошлась бы с первой на первой же
# правке. Направление: presets.py → agent.py → user_profile.py → context.py;
# presets.py → user_profile.py; app.py → user_profile.py.
#
# Имя модуля — `user_profile.py`, а не `profile.py`: `profile` — модуль
# стандартной библиотеки (профилировщик, его импортирует `cProfile`), и файл
# `src/profile.py` рядом с `app.py` подменил бы его всему процессу.
#
# Две фазы, как у стратегий и модели памяти: `choices()`, `preview()`,
# `prepare()`, сборка блока и текста — чистые, их зовёт панель на каждый
# рендер. Состояние `ModeRouter` (положение переключателя и режим прошлого
# хода) меняют только `set_choice()`, `settle()` и `reset()` — их зовёт
# только агент под своим замком.

import re
from dataclasses import dataclass

import context

# Заголовок блока профиля в запросе (§4.3) — сразу за системным промптом и
# перед слоями памяти. Нейтральный, без слов про настолки: смысл — это
# настройки, заданные самим пользователем, они выполняются в каждом ответе
# без ссылки на профиль, разговор и системный промпт важнее профиля
# (приоритеты §2.4).
#
# Заголовок — ещё и метка для счёта токенов (§5.6): `Agent._count_messages()`
# узнаёт блок профиля по началу `system`-сообщения, поэтому он не должен
# начинаться одинаково с заголовками слоёв памяти (`memory.py`), сводки и
# фактов (`context.py`).
PROFILE_HEADER = (
    "Профиль пользователя: настройки, которые пользователь задал сам — как "
    "к нему обращаться и как ему отвечать. Выполняй их в каждом ответе, не "
    "пересказывая профиль и не ссылаясь на него (никаких «согласно вашему "
    "профилю»). Если в текущем разговоре пользователь просит иначе — верен "
    "разговор. Если системный промпт требует особого формата ответа — он "
    "важнее профиля."
)

# Переключатель «Профиль в запросе» (§2.3): `авто` — режим выбирает роутер,
# `выключен` — профиль в запрос не уходит. Оба — зарезервированные названия:
# режим профиля не может называться так же, иначе выбор стал бы неоднозначным.
CHOICE_AUTO = "авто"
CHOICE_OFF = "выключен"
_RESERVED_CHOICES = (CHOICE_AUTO, CHOICE_OFF)

_NOT_SET = "не задано"

# Разметка последнего обмена для входа роутера (§3.4) — тот же приём, что у
# свёртки, фактов и разбора памяти.
_ROLE_LABELS = {"user": "[пользователь]", "assistant": "[ассистент]"}

# Подписи полей общей части и режима — константы модуля, нейтральные: слова
# «партия» и «игрок» живут только в текстах `presets.py`.
_COMMON_LABELS = {
    "address": "обращение",
    "language": "язык",
    "level": "уровень",
    "constraints": "ограничения",
}
_MODE_LABELS = {
    "when": "когда",
    "style": "стиль",
    "format": "формат",
    "constraints": "ограничения",
    "steps": "порядок ответа",
}


@dataclass(frozen=True)
class ProfileMode:
    """Профиль под одну задачу пользователя."""

    name: str                       # как в переключателе и в ответе роутера
    when: str                       # когда нужен — по этому описанию выбирает роутер
    style: str = ""
    format: str = ""
    constraints: str = ""
    steps: tuple[str, ...] = ()     # порядок ответа, по шагу


@dataclass(frozen=True)
class UserProfile:
    """Профиль пользователя: общая часть и режимы."""

    address: str = ""               # обращение
    language: str = ""              # язык ответов и названий
    level: str = ""                 # уровень пользователя в теме
    constraints: str = ""           # ограничения во всех ответах
    modes: tuple[ProfileMode, ...] = ()


@dataclass(frozen=True)
class ModeChoice:
    """Что из профиля уходит в запрос на этом ходе."""

    sent: bool                      # False — профиль выключен или его нет
    mode: str | None                # название режима; None — режима нет
    note: str                       # откуда, словами (§4.4)


@dataclass(frozen=True)
class RouteTask:
    """Служебный вызов «роутер профиля». Сам вызов делает агент."""

    label: str                      # существительным
    messages: list[dict]
    max_tokens: int | None


# --- Разбор, проверка и запись профиля (§4.2) ------------------------------

def parse_profile(data: object) -> tuple[UserProfile, list[str]]:
    """Профиль из словаря (файл хранилища или форма редактора) — переживает
    мусор по правилам дня 10 и возвращает профиль и замечания: что пропущено
    и почему.

    Не словарь — пустой профиль. Строковое поле не строка — пустое. `modes`
    не список — режимов нет. Режим не словарь, без названия, с повтором
    названия (после `context.normalize_key`) или с названием, совпадающим с
    `CHOICE_AUTO`/`CHOICE_OFF`, — пропускается. `steps` — список непустых
    строк или одна строка, разбитая по переводам строк (так его отдаёт
    многострочное поле формы). Строки обрезаются по краям.
    """
    if not isinstance(data, dict):
        return UserProfile(), ["профиль не словарь — профиль пустой"]

    notes: list[str] = []
    common = {
        field: _clean_field(data, field, notes) for field in _COMMON_LABELS
    }

    raw_modes = data.get("modes")
    modes: list[ProfileMode] = []
    if raw_modes is None:
        pass
    elif not isinstance(raw_modes, list):
        notes.append("modes не список — режимов нет")
    else:
        seen: set[str] = set()
        for index, item in enumerate(raw_modes):
            mode = _parse_mode(item, seen, notes, index)
            if mode is not None:
                modes.append(mode)
                seen.add(context.normalize_key(mode.name))

    return UserProfile(modes=tuple(modes), **common), notes


def validate_profile(profile: UserProfile) -> list[str]:
    """Ошибки, с которыми профиль сохранять нельзя (§4.2): в профиле нет
    режимов; пустое название; повтор названия; зарезервированное название;
    при двух режимах и больше — пустое «когда» у любого из них (по нему
    выбирает роутер). Все поля общей части и остальные поля режима могут
    быть пустыми."""
    errors: list[str] = []
    if not profile.modes:
        errors.append("в профиле нет ни одного режима")
        return errors

    reserved = {context.normalize_key(choice) for choice in _RESERVED_CHOICES}
    seen: set[str] = set()
    for mode in profile.modes:
        name = (mode.name or "").strip()
        if not name:
            errors.append("у режима пустое название")
            continue
        key = context.normalize_key(name)
        if key in reserved:
            errors.append(f"у режима «{name}» зарезервированное название")
        if key in seen:
            errors.append(f"название режима «{name}» повторяется")
        seen.add(key)

    if len(profile.modes) >= 2:
        for mode in profile.modes:
            if not (mode.when or "").strip():
                errors.append(
                    f"у режима «{mode.name}» пустое «когда» — по нему "
                    f"выбирает роутер"
                )
    return errors


def dump_profile(profile: UserProfile) -> dict:
    """Профиль для файла (§4.2): пустые поля пишутся пустыми строками — у
    формы профиля в файле один вид."""
    return {
        "address": profile.address,
        "language": profile.language,
        "level": profile.level,
        "constraints": profile.constraints,
        "modes": [
            {
                "name": mode.name,
                "when": mode.when,
                "style": mode.style,
                "format": mode.format,
                "constraints": mode.constraints,
                "steps": list(mode.steps),
            }
            for mode in profile.modes
        ],
    }


def profile_changes(old: UserProfile, new: UserProfile) -> list[str]:
    """Что изменилось между двумя профилями, словами для лога и статуса
    (§4.2): «обращение, уровень, режим «Учу новичка»: формат, порядок
    ответа». Пустой список — изменений нет. Режимы сравниваются по названию
    (после нормализации): режим, которого не стало или который появился,
    отмечается как удалённый/добавленный, а не как «переименован»."""
    changes: list[str] = [
        label
        for field, label in _COMMON_LABELS.items()
        if getattr(old, field) != getattr(new, field)
    ]

    old_by_key = {context.normalize_key(m.name): m for m in old.modes}
    new_by_key = {context.normalize_key(m.name): m for m in new.modes}
    seen: set[str] = set()
    for mode in (*old.modes, *new.modes):
        key = context.normalize_key(mode.name)
        if key in seen:
            continue
        seen.add(key)
        before = old_by_key.get(key)
        after = new_by_key.get(key)
        if before is None:
            changes.append(f"режим «{mode.name}»: добавлен")
            continue
        if after is None:
            changes.append(f"режим «{mode.name}»: удалён")
            continue
        fields = [
            label
            for field, label in _MODE_LABELS.items()
            if getattr(before, field) != getattr(after, field)
        ]
        if fields:
            changes.append(f"режим «{after.name}»: {', '.join(fields)}")
    return changes


def mode_by_name(profile: UserProfile, name: str | None) -> ProfileMode | None:
    """Режим профиля по названию, сравнение через `context.normalize_key`;
    `None` — имени нет, или режима с таким именем в профиле нет."""
    if not name:
        return None
    key = context.normalize_key(name)
    for mode in profile.modes:
        if context.normalize_key(mode.name) == key:
            return mode
    return None


# --- Блок профиля для запроса и текст для панели (§4.3) --------------------

def profile_block(profile: UserProfile, mode: str | None) -> dict | None:
    """`{"role": "system", ...}`; `None`, если общая часть пуста и режима
    нет. Пустые поля не выводятся, пустая общая часть не выводится вовсе.
    Режим — только переданный; неизвестное название ведёт себя как `None`."""
    parts: list[str] = []
    common = _common_lines(profile)
    if common:
        parts.append("\n".join(["О пользователе:", *common]))
    active = mode_by_name(profile, mode)
    if active is not None:
        parts.append(_mode_lines(active))
    if not parts:
        return None
    return {
        "role": "system",
        "content": f"{PROFILE_HEADER}\n\n" + "\n\n".join(parts),
    }


def profile_text(profile: UserProfile, active: str | None = None) -> str:
    """То же для панели: общая часть и **все** режимы, у активного пометка.
    Пустые поля — «не задано», чтобы было видно, что пользователь ещё не
    заполнил."""
    active_key = context.normalize_key(active) if active else None
    lines = [
        "О пользователе:",
        f"- обращение: {profile.address or _NOT_SET}",
        f"- язык: {profile.language or _NOT_SET}",
        f"- уровень: {profile.level or _NOT_SET}",
        f"- ограничения: {profile.constraints or _NOT_SET}",
        "",
        "Режимы:" if profile.modes else "Режимы: ни одного",
    ]
    for mode in profile.modes:
        marker = (
            " — активен"
            if active_key and context.normalize_key(mode.name) == active_key
            else ""
        )
        steps = (
            "; ".join(f"{i}. {step}" for i, step in enumerate(mode.steps, 1))
            if mode.steps
            else _NOT_SET
        )
        lines += [
            "",
            f"«{mode.name}»{marker}",
            f"- когда: {mode.when or _NOT_SET}",
            f"- стиль: {mode.style or _NOT_SET}",
            f"- формат: {mode.format or _NOT_SET}",
            f"- ограничения: {mode.constraints or _NOT_SET}",
            f"- порядок ответа: {steps}",
        ]
    return "\n".join(lines)


# --- ModeRouter (§4.4) ------------------------------------------------------

class ModeRouter:
    """Выбор режима профиля для одного агента: положение переключателя
    «Профиль в запросе», режим прошлого хода, вход роутера и разбор его
    ответа."""

    def __init__(self, prompt: str, max_tokens: int) -> None:
        self._prompt = prompt
        self._max_tokens = max_tokens
        self._choice: str = CHOICE_AUTO
        self._last: ModeChoice | None = None

    @property
    def choice(self) -> str:
        return self._choice

    @property
    def last(self) -> ModeChoice | None:
        """Что ушло в запрос прошлого хода; `None` — ходов не было."""
        return self._last

    # --- Чистые методы: их зовут на каждый рендер панели -------------------

    def choices(self, profile: UserProfile) -> list[str]:
        return [CHOICE_AUTO, *(mode.name for mode in profile.modes), CHOICE_OFF]

    def preview(self, profile: UserProfile) -> ModeChoice:
        """Что уйдёт в запрос без вызова роутера (§4.4)."""
        if self._choice == CHOICE_OFF:
            return ModeChoice(False, None, "профиль выключен")
        if not profile.modes:
            return ModeChoice(True, None, "в профиле нет режимов")
        if len(profile.modes) == 1:
            only = profile.modes[0]
            return ModeChoice(True, only.name, "единственный режим профиля")
        manual = self._manual_mode(profile)
        if manual is not None:
            return ModeChoice(True, manual.name, "выбран вручную")
        # `CHOICE_AUTO` или устаревшее название (режима с таким именем в
        # профиле больше нет) — ведёт себя одинаково: режим прошлого хода,
        # если он ещё есть в профиле, иначе первый; роутер решит на ходе.
        fallback = self._fallback_mode(profile)
        return ModeChoice(True, fallback.name, "роутер выберет на этом ходе")

    def prepare(
        self, profile: UserProfile, history: list[dict], question: str
    ) -> RouteTask | None:
        """Задача на выбор режима, только если он нужен: `CHOICE_AUTO` (или
        устаревшее название), режимов два и больше, вопрос непустой (§4.4)."""
        if self._choice == CHOICE_OFF:
            return None
        if len(profile.modes) < 2:
            return None
        if self._manual_mode(profile) is not None:
            return None
        question = question or ""
        if not question:
            return None
        return RouteTask(
            label="выбор режима по новому сообщению и последнему обмену",
            messages=[
                {"role": "system", "content": self._prompt},
                {
                    "role": "user",
                    "content": self._request(profile, history, question),
                },
            ],
            max_tokens=self._max_tokens,
        )

    # --- Меняют состояние: зовёт только агент под своим замком --------------

    def set_choice(self, choice: str, profile: UserProfile) -> bool:
        """Только значения из `choices(profile)`; `False` — значение
        неизвестно или не изменилось. `last` не трогает."""
        if choice not in self.choices(profile):
            return False
        if choice == self._choice:
            return False
        self._choice = choice
        return True

    def settle(
        self, profile: UserProfile, task: RouteTask | None, text: str | None
    ) -> tuple[ModeChoice, bool]:
        """Фиксирует, что ушло в запрос этого хода, и запоминает в `last`
        (§4.4). Возвращает решение и признак успеха (не запасной ли режим)."""
        if task is None:
            result = self.preview(profile)
            self._last = result
            return result, True
        if text is None:
            result = self._fallback_choice(profile, "роутер не ответил")
            self._last = result
            return result, False
        mode_name = _parse_router_answer(text, profile)
        if mode_name is None:
            result = self._fallback_choice(profile, "ответ роутера не разобран")
            self._last = result
            return result, False
        result = ModeChoice(True, mode_name, "выбран роутером")
        self._last = result
        return result, True

    def reset(self) -> None:
        """Сброс диалога: режим прошлого хода ушёл вместе с ним, переключатель
        не трогается — это состояние агента, а не диалога."""
        self._last = None

    # --- Внутреннее ---------------------------------------------------------

    def _manual_mode(self, profile: UserProfile) -> ProfileMode | None:
        """Режим, выбранный вручную (переключатель стоит на имени
        существующего режима); `None` — `CHOICE_AUTO`, `CHOICE_OFF` или
        устаревшее название."""
        if self._choice in (CHOICE_AUTO, CHOICE_OFF):
            return None
        return mode_by_name(profile, self._choice)

    def _fallback_mode(self, profile: UserProfile) -> ProfileMode:
        """Режим прошлого хода, если он ещё есть в профиле, иначе первый
        режим профиля. Вызывается только когда `profile.modes` уже проверено
        непустым (в частности — двумя и более)."""
        if self._last is not None and self._last.mode is not None:
            previous = mode_by_name(profile, self._last.mode)
            if previous is not None:
                return previous
        return profile.modes[0]

    def _fallback_choice(self, profile: UserProfile, reason: str) -> ModeChoice:
        previous = (
            mode_by_name(profile, self._last.mode)
            if self._last is not None and self._last.mode is not None
            else None
        )
        fallback = previous if previous is not None else profile.modes[0]
        suffix = "режим прошлого хода" if previous is not None else "первый режим профиля"
        return ModeChoice(True, fallback.name, f"{reason} — {suffix}")

    def _request(
        self, profile: UserProfile, history: list[dict], question: str
    ) -> str:
        """Вход роутера текстом внутри одного сообщения `user` (§3.4), тем
        же приёмом, что у свёртки, фактов и разбора памяти. Общая часть
        профиля во вход не идёт: выбор режима зависит от задачи, а не от
        того, кто спрашивает."""
        modes_lines = [f"- {mode.name} — {mode.when}" for mode in profile.modes]
        previous = self._last.mode if self._last is not None else None
        previous_line = (
            f"Режим прошлого хода: {previous}"
            if previous
            else "Режим прошлого хода: нет — это первый ход"
        )
        last_exchange = [m for m in (history or [])[-2:] if isinstance(m, dict)]
        exchange_text = (
            "\n".join(
                f"{_ROLE_LABELS.get(m.get('role'), '[реплика]')} "
                f"{m.get('content') or ''}"
                for m in last_exchange
            )
            if last_exchange
            else "разговор только начинается"
        )
        return "\n".join(
            [
                "Режимы:",
                *modes_lines,
                "",
                previous_line,
                "",
                "Последний обмен:",
                exchange_text,
                "",
                "Новое сообщение пользователя:",
                question,
            ]
        )


# --- Внутреннее: разбор профиля и ответа роутера ----------------------------

def _clean_field(data: dict, key: str, notes: list[str]) -> str:
    value = data.get(key)
    if value is None:
        return ""
    if not isinstance(value, str):
        notes.append(f"поле «{key}» не строка — пустое")
        return ""
    return value.strip()


def _parse_mode(
    item: object, seen: set[str], notes: list[str], index: int
) -> ProfileMode | None:
    if not isinstance(item, dict):
        notes.append(f"режим #{index + 1}: не словарь — пропущен")
        return None
    raw_name = item.get("name")
    if not isinstance(raw_name, str) or not raw_name.strip():
        notes.append(f"режим #{index + 1}: без названия — пропущен")
        return None
    name = raw_name.strip()
    key = context.normalize_key(name)
    if key in {context.normalize_key(choice) for choice in _RESERVED_CHOICES}:
        notes.append(f"режим «{name}»: зарезервированное название — пропущен")
        return None
    if key in seen:
        notes.append(f"режим «{name}»: название повторяется — пропущен")
        return None
    fields = {
        field: _clean_field(item, field, notes)
        for field in ("when", "style", "format", "constraints")
    }
    steps = _parse_steps(item.get("steps"), notes, name)
    return ProfileMode(name=name, steps=steps, **fields)


def _parse_steps(value: object, notes: list[str], mode_name: str) -> tuple[str, ...]:
    if value is None:
        return ()
    if isinstance(value, str):
        lines = value.splitlines()
    elif isinstance(value, list):
        lines = value
    else:
        notes.append(f"режим «{mode_name}»: steps не список и не строка — пусто")
        return ()
    return tuple(
        line.strip() for line in lines if isinstance(line, str) and line.strip()
    )


def _common_lines(profile: UserProfile) -> list[str]:
    lines = []
    if profile.address:
        lines.append(f"- обращение: {profile.address}")
    if profile.language:
        lines.append(f"- язык: {profile.language}")
    if profile.level:
        lines.append(f"- уровень: {profile.level}")
    if profile.constraints:
        lines.append(f"- ограничения: {profile.constraints}")
    return lines


def _mode_lines(mode: ProfileMode) -> str:
    lines = []
    if mode.style:
        lines.append(f"- стиль: {mode.style}")
    if mode.format:
        lines.append(f"- формат: {mode.format}")
    if mode.constraints:
        lines.append(f"- ограничения: {mode.constraints}")
    if mode.steps:
        lines.append("- порядок ответа:")
        lines += [f"  {i}. {step}" for i, step in enumerate(mode.steps, 1)]
    header = f"Режим ответа «{mode.name}»" + (
        f" — {mode.when}:" if mode.when else ":"
    )
    return "\n".join([header, *lines])


_TRAILING_DOT_RE = re.compile(r"\.\s*$")


def _parse_router_answer(text: str, profile: UserProfile) -> str | None:
    """Разбор ответа роутера (§4.5): первая непустая строка, часть после
    последнего двоеточия (если есть), без завершающей точки, выделения и
    кавычек, нормализованная `context.normalize_key`. Совпадение — ровно
    одно название; частичное совпадение, два названия в строке и название
    вне профиля не угадываются — неразобранный ответ."""
    line = next((raw.strip() for raw in (text or "").splitlines() if raw.strip()), "")
    if not line:
        return None
    if ":" in line:
        line = line.rsplit(":", 1)[1].strip()
    line = _TRAILING_DOT_RE.sub("", line).strip()
    candidate = context.normalize_key(line)
    if not candidate:
        return None
    matches = [
        mode.name for mode in profile.modes
        if context.normalize_key(mode.name) == candidate
    ]
    return matches[0] if len(matches) == 1 else None
