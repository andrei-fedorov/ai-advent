# TooManyRules — хранилище истории диалогов (день 7, неделя 2; день 9 —
# формат версии 2, память стратегии рядом с историей; день 11 — долговременная
# память в отдельном файле).
#
# Реализация протокола `HistoryStore`, объявленного в `agent.py` на дне 6:
# одна сессия — один JSON-файл в `src/data/sessions/`. День 6 объявил
# интерфейс, день 7 подставляет в него реализацию — публичный контракт агента
# при этом не меняется. Когда появится VPS и несколько пользователей, в эту же
# точку встанет SQLite (см. спецификацию дня 7, §2.1).
#
# Модуль ничего не знает ни про Gradio, ни про Too Many Bones и не импортирует
# `presets.py`: на диск едут история и имя пресета, а сами промпты, модель и
# параметры остаются в коде — правка промпта должна доезжать до восстановленных
# агентов, а не оставаться перекрытой копией в файле.
#
# Из `agent.py` тоже ничего не импортируется: протокол структурный, наследовать
# его не требуется. Направление зависимостей: app.py → storage.py.
#
# День 11 добавляет второе хранилище — `JsonLongTermStore`, долговременную
# память агента (спецификация дня 11, §6). Хранение следует за временем
# жизни (§2.2): рабочая память живёт ровно столько, сколько сессия, и едет в
# тот же непрозрачный блок `context` файла сессии — формат файла сессии не
# меняется, версия остаётся второй. Долговременная память с сессиями не
# связана, и её файл лежит вне каталога сессий — рядом с ним, в `memory/`:
# удалённая сессия не должна уносить профиль пользователя, а ветка — жить со
# своей копией. Карты памяти и слоёв хранилище не знает: ключи для него —
# непрозрачные строки.

import contextlib
import copy
import json
import logging
import os
import re
import threading
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

logger = logging.getLogger("toomanyrules.storage")

# Версия формата файла. День 9 — ровно тот случай, под который поле заводилось
# на дне 7: рядом с историей в файл легла память стратегии (`context`), и
# формат стал вторым.
#
# Миграций нет и не будет: версия 1 читается (нет ключа `context` — пустая
# память и стратегия по умолчанию), версия 2 пишется. Первая же запись делает
# старый файл вторым; «на месте» ничего не переписывается.
FORMAT_VERSION = 2

_PROJECT_ROOT = Path(__file__).resolve().parent.parent
_DEFAULT_DATA_DIR = Path(__file__).resolve().parent / "data" / "sessions"

# Каталог данных. Переопределяется переменной окружения — она нужна, чтобы
# поднять второй экземпляр приложения, не перемешав его историю с рабочей:
# межпроцессных блокировок здесь нет и не планируется.
DATA_DIR = Path(os.getenv("TOOMANYRULES_DATA_DIR") or _DEFAULT_DATA_DIR)

# Имя файла сессии: s001.json, s002.json, … — номер с ведущими нулями, чтобы
# файлы в каталоге шли в порядке создания, а сессия читалась в логе глазом.
_SESSION_ID_RE = re.compile(r"^s(\d+)$")

# Каталог долговременной памяти (день 11, §2.2) — сосед каталога сессий, а не
# его подкаталог: в каталог сессий он не попадает, и `sessions()` про него не
# знает. Правило одно на оба случая: по умолчанию это `src/data/memory/`
# рядом с `src/data/sessions/`, а экземпляр, поднятый с
# `TOOMANYRULES_DATA_DIR=<каталог>/sessions`, получает свою долговременную
# память в `<каталог>/memory/` и не пишет в память автора.
LONG_TERM_DIR = DATA_DIR.parent / "memory"

# Версия формата файла долговременной памяти. Своя, а не общая с файлом
# сессии: файлы разные и растут независимо.
LONG_TERM_FORMAT_VERSION = 1


class StorageError(Exception):
    """Файл сессии не прочитался или не записался."""


@dataclass(frozen=True)
class SessionInfo:
    """Строка каталога сессий: то, что видно про сессию, не поднимая агента."""

    session_id: str
    preset: str | None        # None, если в файле его нет
    messages: int             # сколько сообщений в файле
    created_at: str
    updated_at: str
    path: str


class JsonHistoryStore:
    """История диалогов в JSON-файлах, по файлу на сессию.

    Экземпляр в приложении один на процесс и общий для всех агентов:
    `session_id` разводит их по файлам. Разные сессии пишут в разные файлы,
    поэтому сериализовать сами записи не требуется — под `Lock` только выдача
    номеров и кэш метаданных (Gradio обрабатывает запросы в пуле потоков, то
    же обоснование, что у реестра агентов на дне 6).
    """

    def __init__(self, data_dir: Path | None = None) -> None:
        self._data_dir = Path(data_dir) if data_dir is not None else DATA_DIR
        self._lock = threading.Lock()
        # session_id → имя пресета: всё, что этот процесс успел увидеть или
        # выдать. Кэш решает две задачи сразу — не переиспользовать номера
        # сессий и не потерять имя пресета, если файл сессии исчез (сброс
        # диалога), а следующая запись его восстановит.
        self._known: dict[str, str | None] = {}

    @property
    def data_dir(self) -> Path:
        """Каталог сессий — показывается в дебаг-панели."""
        return self._data_dir

    # --- Протокол HistoryStore (объявлен на дне 6, дополнен на дне 9) -----

    def load(self, session_id: str) -> list[dict]:
        """История сессии с диска.

        Файла нет → пустой список: это нормальная ситуация (сессию завели,
        но ещё ни разу не ответили), а не ошибка. Файл не читается или в нём
        не JSON нужной формы → `StorageError`: агент в этом случае выключит
        себе запись и не затрёт файл, который сегодня не прочитался.
        """
        path = self.path_for(session_id)
        if not path.exists():
            logger.info(
                "сессия %s: файла нет, стек пустой (%s)",
                session_id, display_path(path),
            )
            return []

        data = self._read(path)
        messages = _clean_messages(data.get("messages"), session_id)
        self._remember(session_id, _preset_of(data))
        logger.info(
            "загружено %s: %d сообщ. ← %s",
            session_id, len(messages), display_path(path),
        )
        return messages

    def save(
        self,
        session_id: str,
        messages: list[dict],
        context: dict | None = None,
    ) -> None:
        """Записывает историю сессии, переписывая файл целиком и атомарно.

        Инкрементальной дозаписи нет: файл маленький, а простота здесь дороже.
        Пустой список удаляет файл — пустых файлов сессий не бывает: нет
        контекста, нет и сессии.

        `context` (день 9) — непрозрачный блок от агента: память стратегий и
        имя активной. Хранилище не знает ни про стратегии, ни про сводки —
        что пришло, то и уедет обратно. Пишется одной записью вместе с
        историей: разъехаться сводка с диалогом не должна.
        """
        if not messages:
            self.delete_session(session_id)
            return

        path = self.path_for(session_id)
        # Имя пресета и дату создания забираем из файла, если он есть, иначе
        # из кэша процесса: пересоздание файла после сброса диалога не должно
        # терять пресет.
        existing = self._existing_meta(path)
        now = _now()
        payload = {
            "version": FORMAT_VERSION,
            "session_id": session_id,
            "preset": _preset_of(existing) or self._preset_for(session_id),
            "created_at": str(existing.get("created_at") or now),
            "updated_at": now,
            # Контекст идёт перед историей: в файле, открытом глазами, сводку
            # нужно видеть сразу, а не после сотни сообщений.
            "context": context if isinstance(context, dict) else {},
            "messages": [
                {"role": message["role"], "content": message["content"]}
                for message in messages
            ],
        }

        # Запись во временный файл рядом плюс os.replace: падение посреди
        # записи не оставит половины диалога.
        tmp_path = path.with_name(path.name + ".tmp")
        try:
            self._data_dir.mkdir(parents=True, exist_ok=True)
            tmp_path.write_text(
                # ensure_ascii=False обязателен: русский текст в файле должен
                # читаться как русский текст, а не как Ход.
                json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                encoding="utf-8",
            )
            os.replace(tmp_path, path)
        except OSError as exc:
            with contextlib.suppress(OSError):
                tmp_path.unlink(missing_ok=True)
            raise StorageError(
                f"не удалось записать {display_path(path)}: {exc}"
            ) from exc

        logger.info(
            "сохранено %s: %d сообщ.%s → %s",
            session_id, len(messages),
            _context_note(payload["context"]), display_path(path),
        )

    def load_context(self, session_id: str) -> dict:
        """Память стратегий из файла сессии — непрозрачный блок, каким его
        записал агент.

        Исключений не бросает и историю читать не мешает: файла нет, файл
        битый, `context` не словарь — пустой словарь и предупреждение в лог.
        Пустая память хуже полной, но лучше отказа читать диалог.
        """
        path = self.path_for(session_id)
        if not path.exists():
            return {}
        try:
            data = self._read(path)
        except StorageError as exc:
            logger.warning("сессия %s: контекст не прочитан — %s", session_id, exc)
            return {}
        context = data.get("context")
        if context is None:
            # Файл версии 1: ключа `context` в нём нет и не должно быть.
            return {}
        if not isinstance(context, dict):
            logger.warning(
                "сессия %s: в файле context имеет тип %s вместо словаря — "
                "память стратегий пустая, история прочитана как обычно",
                session_id, type(context).__name__,
            )
            return {}
        return context

    # --- Каталог сессий: этим пользуется только app.py -------------------

    def create_session(self, preset: str) -> str:
        """Выдаёт новый `session_id`. Файла при этом не создаёт — он появится
        с первой записью: у обычной сессии это первый успешный ответ, у
        ветки (день 10) — момент создания, сразу с копией истории.

        Номер берётся как максимум по файлам на диске и по всем номерам,
        которые процесс уже видел или выдал: номера сессий в пределах процесса
        не переиспользуются. Иначе новый агент мог бы получить `session_id`
        живого агента, у которого файл только что удалили сбросом, и они
        начали бы писать в один файл.
        """
        with self._lock:
            session_id = f"s{self._next_number():03d}"
            self._known[session_id] = preset
        logger.info(
            "заведена сессия %s (пресет «%s»); файл %s появится с первой записью",
            session_id, preset, display_path(self.path_for(session_id)),
        )
        return session_id

    def sessions(self) -> list[SessionInfo]:
        """Каталог сессий по возрастанию `session_id` (он же порядок создания).

        Исключений не бросает: нечитаемый или посторонний файл пропускается с
        предупреждением в лог. Приложение обязано подниматься при любом
        содержимом каталога данных.
        """
        infos: list[SessionInfo] = []
        for path in sorted(self._data_dir.glob("*")):
            if path.is_dir():
                continue
            if path.suffix != ".json" or not _SESSION_ID_RE.match(path.stem):
                logger.warning(
                    "посторонний файл в каталоге сессий пропущен: %s",
                    display_path(path),
                )
                continue

            session_id = path.stem
            try:
                data = self._read(path)
            except StorageError as exc:
                # Битый файл не читаем и не трогаем: сессия пропускается,
                # остальные восстанавливаются, файл остаётся как был.
                logger.warning("сессия %s пропущена: %s", session_id, exc)
                continue

            preset = _preset_of(data)
            self._remember(session_id, preset)
            infos.append(
                SessionInfo(
                    session_id=session_id,
                    preset=preset,
                    messages=len(data["messages"]),
                    created_at=str(data.get("created_at") or ""),
                    updated_at=str(data.get("updated_at") or ""),
                    path=display_path(path),
                )
            )

        infos.sort(key=lambda info: _session_number(info.session_id) or 0)
        logger.info(
            "каталог сессий %s: сессий на диске %d",
            display_path(self._data_dir), len(infos),
        )
        return infos

    def delete_session(self, session_id: str) -> bool:
        """Удаляет файл сессии; возвращает, был ли он там. Запись о сессии
        в кэше процесса остаётся — её номер не переиспользуется."""
        path = self.path_for(session_id)
        try:
            path.unlink()
        except FileNotFoundError:
            return False
        except OSError as exc:
            raise StorageError(
                f"не удалось удалить {display_path(path)}: {exc}"
            ) from exc
        logger.info(
            "файл сессии %s удалён (%s)", session_id, display_path(path)
        )
        return True

    def path_for(self, session_id: str) -> Path:
        return self._data_dir / f"{session_id}.json"

    def read_file(self, session_id: str) -> dict | None:
        """Сырое содержимое файла для дебаг-панели: `None`, если файла нет
        или он не читается (причина в этом случае уходит в лог).

        Панель перечитывает файл на каждом событии интерфейса, поэтому чтение
        логируется на уровне `debug`, а не `info`: иначе лог заплывёт строками
        на каждый клик, и в нём перестанут быть видны загрузка, запись и
        удаление.
        """
        path = self.path_for(session_id)
        if not path.exists():
            return None
        try:
            data = self._read(path)
        except StorageError as exc:
            logger.warning("сессия %s: файл не показан в панели — %s", session_id, exc)
            return None
        logger.debug("прочитан файл сессии %s (%s)", session_id, display_path(path))
        return data

    # --- Внутреннее ------------------------------------------------------

    def _read(self, path: Path) -> dict:
        """Разбор файла сессии. Всё, что не похоже на файл сессии, — это
        `StorageError` с причиной: она попадёт и в лог, и в дебаг-панель."""
        try:
            raw = path.read_text(encoding="utf-8")
        except (OSError, UnicodeDecodeError) as exc:
            raise StorageError(
                f"файл {display_path(path)} не читается: {exc}"
            ) from exc
        try:
            data = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise StorageError(
                f"файл {display_path(path)} — не JSON: {exc}"
            ) from exc
        if not isinstance(data, dict) or not isinstance(data.get("messages"), list):
            raise StorageError(
                f"файл {display_path(path)} — не файл сессии: нет списка messages"
            )
        return data

    def _existing_meta(self, path: Path) -> dict:
        """Метаданные уже лежащего файла (пресет, дата создания). Файла нет
        или он битый — пустой словарь, запись от этого не отменяется."""
        try:
            return self._read(path)
        except StorageError:
            return {}

    def _next_number(self) -> int:
        """Следующий свободный номер сессии. Вызывается под `self._lock`."""
        numbers = {
            number
            for name in self._file_stems() + list(self._known)
            if (number := _session_number(name)) is not None
        }
        return max(numbers, default=0) + 1

    def _file_stems(self) -> list[str]:
        """Имена файлов сессий на диске без расширения. Битый файл тоже
        считается занятым номером — его не перезапишет новая сессия."""
        return [path.stem for path in self._data_dir.glob("*.json")]

    def _remember(self, session_id: str, preset: str | None) -> None:
        """Запоминает увиденную сессию и её пресет. Известное имя пресета
        неизвестным не затирается."""
        with self._lock:
            if preset is not None or session_id not in self._known:
                self._known[session_id] = preset

    def _preset_for(self, session_id: str) -> str | None:
        with self._lock:
            return self._known.get(session_id)


class JsonLongTermStore:
    """Долговременная память в одном JSON-файле, общем для всех сессий и
    агентов процесса. Хранилище не знает ни карты памяти, ни слоёв: ключи
    для него — непрозрачные строки.

    Экземпляр в приложении один на процесс и передаётся всем агентам тем же
    объектом, как `JsonHistoryStore`. Файл читается один раз при создании,
    дальше записи живут в памяти процесса, а файл — источник при старте. Два
    процесса с одним каталогом не синхронизируются (правило дня 7): второй
    экземпляр поднимается с другим `TOOMANYRULES_DATA_DIR`, и у него своя
    долговременная память.
    """

    def __init__(self, path: Path | None = None) -> None:
        self._path = (
            Path(path) if path is not None else LONG_TERM_DIR / "long_term.json"
        )
        # Замок нужен, потому что хранилище общее: два агента, сохраняющие
        # кандидатов одновременно, без него записали бы файл в обратном
        # порядке и потеряли бы одно решение человека.
        self._lock = threading.Lock()
        self._entries: dict[str, dict] = {}
        self._error: str | None = None
        # Запись выключается ровно в одном случае — файл есть, но не
        # прочитался: не затираем то, что сегодня не прочиталось (день 7).
        self._writable = True
        self._load()

    @property
    def path(self) -> str:
        """Путь к файлу для логов и панели — от корня репозитория."""
        return display_path(self._path)

    @property
    def error(self) -> str | None:
        """Почему файл не прочитался или не записался; `None` — всё в порядке."""
        return self._error

    def entries(self) -> dict:
        """Копия записей: «ключ → {"value", "session_id", "updated_at"}».

        Глубокая копия, без замка: словарь записей подменяется целиком после
        каждой записи и после этого не меняется, поэтому панель и агенты
        читают его, не дожидаясь чужой записи, — и не должны получить
        словарь, который меняется у них под руками.
        """
        return copy.deepcopy(self._entries)

    def apply(self, changes: dict[str, str | None], session_id: str) -> dict:
        """Слияние изменений и запись файла; возвращает новую копию записей.

        `None` удаляет ключ (удаление отсутствующего — не ошибка), значение
        слово в слово запись не трогает — ни значение, ни происхождение.
        Сливается копия, файл пишется атомарно (временный файл рядом плюс
        `os.replace`, как у сессий), и только после успешной записи
        подменяется кэш. Сбой записи — `StorageError`, кэш остаётся прежним:
        решение человека не должно считаться сохранённым, если его нет на
        диске.

        Пустой набор записей файл **не** удаляет: в отличие от сессии, пустая
        долговременная память — законное состояние «человек всё отклонил и
        всё удалил», и `updated_at` у файла это показывает.
        """
        with self._lock:
            if not self._writable:
                raise StorageError(
                    f"запись долговременной памяти выключена: {self._error}"
                )
            entries = copy.deepcopy(self._entries)
            now = _now()
            changed: list[str] = []
            for key, value in (changes or {}).items():
                if not isinstance(key, str) or not key:
                    continue
                if value is None:
                    if key in entries:
                        del entries[key]
                        changed.append(f"−{key}")
                    continue
                current = entries.get(key)
                if current is not None and current.get("value") == value:
                    continue
                changed.append(f"{'~' if current is not None else '+'}{key}")
                entries[key] = {
                    "value": value,
                    "session_id": session_id,
                    "updated_at": now,
                }
            if not changed:
                # Писать нечего: всё уже лежит в файле в этом виде.
                return copy.deepcopy(entries)

            payload = {
                "version": LONG_TERM_FORMAT_VERSION,
                "updated_at": now,
                "entries": entries,
            }
            tmp_path = self._path.with_name(self._path.name + ".tmp")
            try:
                self._path.parent.mkdir(parents=True, exist_ok=True)
                tmp_path.write_text(
                    json.dumps(payload, ensure_ascii=False, indent=2) + "\n",
                    encoding="utf-8",
                )
                os.replace(tmp_path, self._path)
            except OSError as exc:
                with contextlib.suppress(OSError):
                    tmp_path.unlink(missing_ok=True)
                self._error = f"не удалось записать {self.path}: {exc}"
                logger.warning("долговременная память: %s", self._error)
                raise StorageError(self._error) from exc

            self._entries = entries
            self._error = None
            logger.info(
                "долговременная память: записано %s (сессия %s), записей %d → %s",
                ", ".join(changed), session_id, len(entries), self.path,
            )
            return copy.deepcopy(entries)

    def read_file(self) -> dict | None:
        """Сырое содержимое файла для дебаг-панели: `None`, если файла нет
        или он не читается. Панель перечитывает файл на каждом событии
        интерфейса, поэтому успешное чтение логируется на уровне `debug`."""
        if not self._path.exists():
            return None
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            logger.warning(
                "долговременная память: файл не показан в панели — %s: %s",
                self.path, exc,
            )
            return None
        logger.debug("прочитан файл долговременной памяти (%s)", self.path)
        return data if isinstance(data, dict) else None

    # --- Внутреннее ------------------------------------------------------

    def _load(self) -> None:
        """Чтение файла при создании хранилища. Нет файла — пустая память, это
        нормально. Файл не читается, не JSON, не той формы — пустая память,
        запись выключена, файл не трогается. Отдельные битые записи
        пропускаются с предупреждением, остальные читаются."""
        if not self._path.exists():
            logger.info(
                "долговременная память: файла нет, память пустая — %s появится "
                "с первым сохранённым кандидатом",
                self.path,
            )
            return
        try:
            data = json.loads(self._path.read_text(encoding="utf-8"))
        except (OSError, UnicodeDecodeError, json.JSONDecodeError) as exc:
            self._disable(f"файл {self.path} не прочитался: {exc}")
            return
        if not isinstance(data, dict) or not isinstance(data.get("entries"), dict):
            self._disable(
                f"файл {self.path} — не файл долговременной памяти: нет "
                f"словаря entries"
            )
            return
        version = data.get("version")
        if version is not None and version != LONG_TERM_FORMAT_VERSION:
            self._disable(
                f"файл {self.path} — формат версии {version!r}, этот код "
                f"читает версию {LONG_TERM_FORMAT_VERSION}"
            )
            return

        entries: dict[str, dict] = {}
        dropped: list[str] = []
        for key, entry in data["entries"].items():
            value = entry.get("value") if isinstance(entry, dict) else None
            if not key or not isinstance(value, str) or not value.strip():
                dropped.append(str(key))
                continue
            entries[key] = {
                "value": value,
                "session_id": _str_or_empty(entry.get("session_id")),
                "updated_at": _str_or_empty(entry.get("updated_at")),
            }
        if dropped:
            logger.warning(
                "долговременная память: пропущено битых записей %d (%s), "
                "прочитано %d — %s",
                len(dropped), ", ".join(dropped), len(entries), self.path,
            )
        self._entries = entries
        logger.info(
            "долговременная память: загружено %d записей ← %s",
            len(entries), self.path,
        )

    def _disable(self, reason: str) -> None:
        self._writable = False
        self._error = f"{reason}; память пустая, запись выключена, файл не тронут"
        logger.warning("долговременная память: %s", self._error)


def _str_or_empty(value: object) -> str:
    return value if isinstance(value, str) else ""


def _clean_messages(messages: object, session_id: str) -> list[dict]:
    """Оставляет только пары user/assistant со строковым содержимым.

    Битые записи отбрасываются с предупреждением: одна испорченная строка не
    должна стоить всего диалога.
    """
    if not isinstance(messages, list):
        return []
    clean: list[dict] = []
    dropped = 0
    for message in messages:
        if (
            isinstance(message, dict)
            and message.get("role") in ("user", "assistant")
            and isinstance(message.get("content"), str)
        ):
            clean.append({"role": message["role"], "content": message["content"]})
        else:
            dropped += 1
    if dropped:
        logger.warning(
            "сессия %s: отброшено битых записей в messages: %d (осталось %d)",
            session_id, dropped, len(clean),
        )
    return clean


def _context_note(context: dict) -> str:
    """Кусок строки лога про записанный контекст: по логу должно быть видно,
    что на диск уехала не только история."""
    if not context:
        return ""
    memory = context.get("memory")
    kinds = len(memory) if isinstance(memory, dict) else 0
    return f" + контекст (стратегия «{context.get('strategy')}», памяти: {kinds})"


def _preset_of(data: dict) -> str | None:
    preset = data.get("preset")
    return preset if isinstance(preset, str) else None


def _session_number(name: str) -> int | None:
    match = _SESSION_ID_RE.match(name)
    return int(match.group(1)) if match else None


def _now() -> str:
    return datetime.now().isoformat(timespec="seconds")


def display_path(path: Path) -> str:
    """Путь для логов и дебаг-панели: относительный от корня репозитория
    (`src/data/sessions/s001.json`), чтобы его можно было скопировать в
    соседний терминал. Каталог за пределами репозитория показывается как есть.
    """
    try:
        return str(path.resolve().relative_to(_PROJECT_ROOT))
    except ValueError:
        return str(path)
