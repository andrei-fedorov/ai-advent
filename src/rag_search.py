# TooManyRules — поиск по индексу правил (день 22, неделя 5; второй этап — день 23).
#
# Индекс строит `rag_index.py` (день 21), читает — этот модуль: первый RAG-запрос
# ассистента и `--probe` программы индексации ищут одной функцией
# (`RulesIndex.nearest()`). Модуль про Too Many Bones не знает: путь индекса,
# стратегия, язык, число кусков и ожидаемая модель — параметрами. LLM не
# вызывает; `sentence_transformers` не импортирует — вектор вопроса считает
# `embedder.py`. Сюда переехали версия схемы, набор таблиц и проверка файла
# индекса (`check_index()`): их знает и тот, кто пишет индекс, и тот, кто его
# читает. Схема (DDL) и запись остаются в `rag_index.py`.
#
# Куски индекса живут в памяти процесса целиком (сотни строк, около мегабайта
# векторов): фильтр по стратегии и языку — маской, поиск — перебором скалярных
# произведений (векторы нормированы — это косинусная близость). Индекс
# читается `mode=ro` и перечитывается, если файл сменился: `./run.sh index`
# ставит новый файл через `os.replace()`, поэтому пересборка во время работы
# приложения подхватывается следующим вопросом. Модель эмбеддингов берётся из
# `meta` индекса — векторы разных моделей несравнимы — и загружается при первом
# поиске, а не при создании: старт приложения не ждёт `torch`. Всё, что меняет
# состояние (перечитывание, загрузка модели, вектор вопроса), — под замком:
# Gradio выполняет обработчики параллельно, а модель одна на процесс.
#
# С дня 23 (спецификация, §4) у поиска есть второй этап: этап 1 берёт
# `candidates_k` ближайших кусков, этап 2 — `Reranker` из `embedder.py` —
# оценивает каждую пару «запрос, кусок» (0..1); куски ниже порога отсеиваются, из
# прошедших уходит не больше `top_k`. Реранкер грузится лениво при первом
# поиске со вторым этапом, из кэша и без скачивания, под тем же замком, что и
# модель эмбеддингов. Сбой второго этапа — выдача этапа 1 с пометкой, а не сбой
# поиска; пустая выдача после порога — штатный ответ (`ok=True`, `hits=[]`).
#
# `search()` не бросает: сбой возвращается словарём (`ok=False`, `error`), ход
# агента из-за него не отменяется (правило каталога инструментов, день 17).
# Логи — logging, `[RAG]`; каждый ход агент пишет свою строку.

import logging
import sqlite3
import threading
import time
from collections.abc import Mapping, Sequence
from contextlib import closing
from dataclasses import asdict, dataclass, replace
from pathlib import Path

import numpy as np

from embedder import Embedder, Reranker, in_cache

logger = logging.getLogger("toomanyrules.rag")

ROOT = Path(__file__).resolve().parent.parent
SCHEMA_VERSION = 1
TABLES = {"meta", "documents", "chunks"}


class IndexFileError(Exception):
    """По пути индекса лежит чужой или битый файл — он не затирается."""


class RagError(Exception):
    """Поиск не удался; текст понятен человеку (для лога и панели)."""


def shown(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def check_index(path: Path) -> int:
    """Версия схемы файла по пути индекса; 0 — файла нет или он пустой.
    Только чтение (`mode=ro`): отказ не оставляет следов в файле (правило
    базы сервера FAQ, день 18). Своя версия с чужим набором таблиц — тоже
    чужой файл: версия 1 бывает и у базы сервера FAQ дня 18."""
    if not path.exists() or path.stat().st_size == 0:
        return 0
    try:
        uri = path.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            version = conn.execute("PRAGMA user_version").fetchone()[0]
            tables = {row[0] for row in conn.execute("SELECT name FROM sqlite_master WHERE type = 'table'")}
    except sqlite3.Error as exc:
        raise IndexFileError(f"файл не открывается как база SQLite: {exc}") from exc
    if version == 0 and tables:
        raise IndexFileError("версия схемы 0, но в базе уже есть таблицы — это не индекс")
    if version not in (0, SCHEMA_VERSION):
        raise IndexFileError(f"версия схемы {version}, программа знает только {SCHEMA_VERSION}")
    if version == SCHEMA_VERSION and tables != TABLES:
        raise IndexFileError(f"версия схемы {version}, но таблицы не индекса: {', '.join(sorted(tables)) or 'нет'}")
    return version


@dataclass(frozen=True)
class Hit:
    """Кусок выдачи. `tokens` — токенизатором модели эмбеддингов (из индекса),
    `score` — косинусная близость к вопросу, `rerank_score` (с дня 23) —
    оценка реранкера 0..1; `None` — второго этапа не было."""

    chunk_id: str
    doc: str
    lang: str
    title: str
    section: str
    part: str | None
    page_from: int
    page_to: int
    text: str
    tokens: int
    score: float
    rerank_score: float | None = None

    def as_dict(self) -> dict:
        return asdict(self)


def sources_found(
    hits: Sequence[Hit | Mapping], sources: Sequence[tuple[str, str, int]],
) -> list[int | None]:
    """Для каждого ожидаемого источника («документ, раздел, страница») — место
    в выдаче (с единицы) или `None`. Найден — значит, в выдаче есть кусок того
    же документа и раздела, покрывающий эту страницу; часть длинного раздела
    не важна. Одно определение «найден» для панели и программы сравнения; про
    ответ модели оно ничего не говорит."""

    def field(hit: Hit | Mapping, name: str):
        return hit[name] if isinstance(hit, Mapping) else getattr(hit, name)

    places: list[int | None] = []
    for doc, section, page in sources:
        place = None
        for number, hit in enumerate(hits, 1):
            if (
                field(hit, "doc") == doc and field(hit, "section") == section
                and field(hit, "page_from") <= page <= field(hit, "page_to")
            ):
                place = number
                break
        places.append(place)
    return places


class RulesIndex:
    """Индекс правил для поиска: один на процесс, общий для агентов."""

    def __init__(
        self, path: Path, strategy: str, lang: str, top_k: int,
        candidates_k: int = 0, rerank_model: str = "", threshold: float = 0.0,
        rerank_max_length: int = 1024, rerank_batch: int = 4,
        allow_download: bool = False, expected_model: str = "",
    ) -> None:
        self.path = Path(path)
        self.strategy = strategy
        self.lang = lang
        self.top_k = top_k
        # Второй этап (день 23, §4.1): `rerank_model=""` — его нет,
        # `candidates_k=0` — кандидатов столько же, сколько выдачи.
        self.candidates_k = candidates_k or top_k
        self.rerank_model = rerank_model
        self.threshold = threshold
        self.rerank_max_length = rerank_max_length
        self.rerank_batch = rerank_batch
        self.allow_download = allow_download
        self.expected_model = expected_model
        self._lock = threading.RLock()
        self._stamp: tuple[int, int, int] | None = None
        self._meta: dict[str, str] = {}
        self._rows: list[tuple] = []
        self._matrix = np.zeros((0, 0), dtype=np.float32)
        self._strategies = np.zeros(0, dtype=object)
        self._langs = np.zeros(0, dtype=object)
        self._embedder: Embedder | None = None
        self._reranker: Reranker | None = None
        self._warned_model = ""
        self._last_info: dict | None = None

    @property
    def name(self) -> str:
        if not self.rerank_model:
            return f"{self.strategy}/{self.lang}, top-{self.top_k} · {shown(self.path)}"
        return (
            f"{self.strategy}/{self.lang}: {self.candidates_k} → {self.rerank_model.rsplit('/', 1)[-1]} "
            f"≥ {self.threshold:.2f} → ≤{self.top_k} · {shown(self.path)}"
        )

    @property
    def device(self) -> str:
        with self._lock:
            return self._embedder.device if self._embedder is not None and self._embedder.loaded else "н/д"

    # --- Чтение индекса ------------------------------------------------------

    def _stat(self) -> tuple[int, int, int]:
        try:
            st = self.path.stat()
        except OSError:
            self._stamp = None
            raise RagError(f"индекса нет: {shown(self.path)} — сначала ./run.sh index") from None
        return (st.st_mtime_ns, st.st_size, st.st_ino)

    def _check(self) -> None:
        try:
            version = check_index(self.path)
        except IndexFileError as exc:
            raise RagError(f"по пути индекса {shown(self.path)}: {exc}") from exc
        if version == 0:
            raise RagError(f"индекса нет: {shown(self.path)} — сначала ./run.sh index")

    def _reload(self) -> None:
        """Индекс с диска в память. Под замком; зовёт `_refresh()`."""
        self._check()
        uri = self.path.resolve().as_uri() + "?mode=ro"
        with closing(sqlite3.connect(uri, uri=True)) as conn:
            conn.row_factory = sqlite3.Row
            meta = {row["key"]: row["value"] for row in conn.execute("SELECT key, value FROM meta")}
            rows = conn.execute(
                "SELECT chunk_id, strategy, doc, lang, title, section, part, page_from, page_to, "
                "text, tokens, embedding FROM chunks ORDER BY strategy, lang, ordinal"
            ).fetchall()
        if "embed_model" not in meta:
            raise RagError(f"по пути индекса {shown(self.path)}: в meta нет модели эмбеддингов")
        self._meta = meta
        self._rows = [tuple(row)[:11] for row in rows]
        self._matrix = (
            np.stack([np.frombuffer(row["embedding"], dtype="<f4") for row in rows]).astype(np.float32)
            if rows else np.zeros((0, 0), dtype=np.float32)
        )
        self._strategies = np.array([row["strategy"] for row in rows], dtype=object)
        self._langs = np.array([row["lang"] for row in rows], dtype=object)
        model = meta["embed_model"]
        if self._embedder is not None and self._embedder.model_name != model:
            self._embedder = None
        if self.expected_model and model != self.expected_model and self._warned_model != model:
            self._warned_model = model
            logger.warning(
                "[RAG] индекс собран моделью %s, ожидается %s: вопросы считаю моделью индекса "
                "(векторы разных моделей несравнимы)", model, self.expected_model,
            )

    def _refresh(self) -> None:
        """Перечитывает индекс, если файл сменился (или его ещё не читали)."""
        stamp = self._stat()
        if stamp == self._stamp:
            return
        again = self._stamp is not None
        step = time.perf_counter()
        self._stamp = None
        self._reload()
        self._stamp = stamp
        logger.info(
            "[RAG] %s · %d кусков · %.2f с", "индекс сменился — перечитан" if again else "индекс прочитан",
            len(self._rows), time.perf_counter() - step,
        )

    def _mask(self, strategy: str, lang: str) -> np.ndarray:
        return (self._strategies == strategy) & (self._langs == lang)

    # --- Модель ----------------------------------------------------------------

    def load_model(self) -> float:
        """Загружает модель индекса (если ещё нет); секунды загрузки, 0.0 —
        если модель уже была в памяти. Скачивания нет, пока `allow_download`
        не разрешён."""
        with self._lock:
            self._refresh()
            return self._load_model()

    def _load_model(self) -> float:
        """`load_model()` без перечитывания индекса: под замком, индекс уже
        перечитан. `nearest()` и `search()` считают маску до загрузки модели:
        если бы загрузка перечитала сменившийся файл ещё раз, маска
        разошлась бы с матрицей (правка по ревью дня 22)."""
        model = self._meta["embed_model"]
        if self._embedder is None:
            self._embedder = Embedder(model)
        if self._embedder.loaded:
            return 0.0
        step = time.perf_counter()
        try:
            self._embedder.load(
                lambda: logger.info(
                    "[RAG] модели %s нет в кэше — скачиваю ≈1,1 ГБ в кэш Hugging Face (один раз)", model,
                ),
                allow_download=self.allow_download,
            )
        except OSError as exc:
            raise RagError(str(exc)) from exc
        loaded = time.perf_counter() - step
        logger.info("[RAG] модель %s загружена за %.1f с · устройство %s", model, loaded, self._embedder.device)
        return loaded

    def _load_reranker(self) -> float:
        """Загрузка реранкера (если ещё нет); секунды загрузки, 0.0 — уже был в
        памяти. Под замком; скачивания нет, пока `allow_download` не разрешён."""
        if not self.rerank_model:
            raise RagError("второй этап не настроен")
        if self._reranker is None:
            self._reranker = Reranker(self.rerank_model, self.rerank_max_length, self.rerank_batch)
        if self._reranker.loaded:
            return 0.0
        step = time.perf_counter()
        try:
            self._reranker.load(
                lambda: logger.info(
                    "[RAG] модели реранкера %s нет в кэше — скачиваю ≈1,5 ГБ в кэш Hugging Face (один раз)",
                    self.rerank_model,
                ),
                allow_download=self.allow_download,
            )
        except OSError as exc:
            raise RagError(str(exc)) from exc
        loaded = time.perf_counter() - step
        logger.info(
            "[RAG] реранкер %s загружен за %.1f с · устройство %s · %s",
            self.rerank_model, loaded, self._reranker.device, self._reranker.dtype,
        )
        return loaded

    def _rerank(self, query: str, hits: Sequence[Hit]) -> tuple[list[Hit], float, float]:
        """Второй этап над готовыми кусками: `(куски с оценками по убыванию,
        секунды загрузки, секунды оценки)`. Под замком (зовут `rerank()` и
        `search()`); бросает `RagError`."""
        load_s = self._load_reranker()
        step = time.perf_counter()
        scores = self._reranker.score(query, [f"{hit.section}\n{hit.text}" for hit in hits])
        score_s = time.perf_counter() - step
        order = sorted(range(len(hits)), key=lambda i: (-float(scores[i]), i))
        return [replace(hits[i], rerank_score=float(scores[i])) for i in order], load_s, score_s

    # --- Публичное ---------------------------------------------------------------

    def rerank(self, query: str, hits: Sequence[Hit]) -> list[Hit]:
        """Второй этап над готовыми кусками (день 23, §4.1): оценки реранкера,
        порядок по убыванию (при равенстве — исходный), `rerank_score` у
        каждого. Отсев по порогу и потолок делает `search()`. Бросает
        `RagError`."""
        with self._lock:
            try:
                return self._rerank(query, hits)[0]
            except RagError:
                raise
            except Exception as exc:  # noqa: BLE001
                logger.exception("[RAG] реранкер не сработал")
                raise RagError(f"второй этап: {exc}") from exc

    def info(self) -> dict:
        """Состояние индекса без модели: для строки при старте и шапки блока.
        `{"ok", "error", ...}`; не бросает. Рендер панели не ждёт поиска: замок
        занят (идёт поиск или загрузка модели) — отдаётся последнее известное
        состояние."""
        if not self._lock.acquire(blocking=False):
            if self._last_info is not None:
                return self._last_info
            self._lock.acquire()
        try:
            self._last_info = self._info()
            return self._last_info
        finally:
            self._lock.release()

    def _info(self) -> dict:
        try:
            self._refresh()
        except RagError as exc:
            return {"ok": False, "error": str(exc), "path": shown(self.path), **self._stage2_info()}
        except Exception as exc:  # noqa: BLE001 — панель не должна падать от индекса
            logger.exception("[RAG] info() не удался")
            return {"ok": False, "error": f"индекс: {exc}", "path": shown(self.path), **self._stage2_info()}
        own = int(self._mask(self.strategy, self.lang).sum())
        counts: dict[str, int] = {}
        for strategy in self._strategies:
            counts[strategy] = counts.get(strategy, 0) + 1
        return {
            "ok": True, "error": "", "path": shown(self.path), "built_at": self._meta.get("built_at", ""),
            "model": self._meta["embed_model"], "device": self._meta.get("device", ""),
            "chunks": own, "total": len(self._rows), "by_strategy": counts,
            "strategy": self.strategy, "lang": self.lang, "top_k": self.top_k,
            **self._stage2_info(),
        }

    def _stage2_info(self) -> dict:
        """Условия второго этапа для строки при старте и панели (день 23, §4.1):
        модели не грузит."""
        loaded = self._reranker is not None and self._reranker.loaded
        info = {
            "candidates_k": self.candidates_k, "rerank_model": self.rerank_model,
            "threshold": self.threshold,
            "rerank_cached": in_cache(self.rerank_model) if self.rerank_model else False,
            "rerank_loaded": loaded,
        }
        if loaded:
            info["rerank_device"] = self._reranker.device
            info["rerank_dtype"] = self._reranker.dtype
        return info

    def nearest(
        self, question: str, strategy: str | None = None, lang: str | None = None, top: int | None = None,
        vector: np.ndarray | None = None,
    ) -> list[Hit]:
        """Ближайшие куски стратегии и языка; бросает `RagError`. Вопрос как
        есть — без истории и переписывания. `vector` — уже посчитанный вектор
        вопроса (`--probe` считает его один раз на обе стратегии)."""
        strategy = strategy or self.strategy
        lang = lang or self.lang
        top = top or self.top_k
        with self._lock:
            self._refresh()
            mask = self._mask(strategy, lang)
            if not mask.any():
                raise RagError(f"в индексе нет кусков {strategy}/{lang}")
            if vector is None:
                self._load_model()
                vector = self._embedder.embed_query(question)
            return self._ranked(mask, vector, top)

    def _ranked(self, mask: np.ndarray, vector: np.ndarray, top: int) -> list[Hit]:
        """Перебор: `top` ближайших к вектору среди кусков маски. Под замком,
        индекс уже перечитан — зовут `nearest()` и `search()`."""
        indexes = np.flatnonzero(mask)
        scores = self._matrix[indexes] @ np.asarray(vector, dtype=np.float32)
        order = np.argsort(-scores)[:top]
        hits: list[Hit] = []
        for position in order:
            row = self._rows[indexes[position]]
            hits.append(Hit(
                chunk_id=row[0], doc=row[2], lang=row[3], title=row[4], section=row[5], part=row[6],
                page_from=row[7], page_to=row[8], text=row[9], tokens=row[10], score=float(scores[position]),
            ))
        return hits

    def query_vector(self, question: str) -> np.ndarray:
        """Вектор вопроса моделью индекса (для `--probe`); бросает `RagError`."""
        with self._lock:
            self.load_model()
            return self._embedder.embed_query(question)

    def search(self, question: str, rerank: bool = False) -> dict:
        """Протокол `Retriever` агента: ближайшие куски со стратегией, языком и
        `top_k` экземпляра, словарём, без исключений. `rerank=False` — день 22:
        `top_k` ближайших. `rerank=True` (день 23, §4.2) — этап 1 берёт
        `candidates_k` ближайших, этап 2 оценивает их реранкером; в `hits` —
        прошедшие порог, не больше `top_k`, по убыванию оценки. **`ok=True` с
        пустыми `hits` — штатный ответ**: второй этап удался и отсёк всех. Сбой
        этапа 2 — `hits` этапа 1 (`top_k` ближайших), `rerank_ok=False` и
        причина. `elapsed` — вся работа поиска без обеих загрузок (`load_s` —
        модель эмбеддингов, `rerank_load_s` — реранкер)."""
        started = time.perf_counter()
        result = {
            "ok": False, "error": "", "elapsed": 0.0, "load_s": 0.0, "embed_s": 0.0, "total": 0, "hits": [],
            "query": question, "rerank": bool(rerank), "rerank_ok": False, "rerank_error": "",
            "rerank_model": self.rerank_model, "threshold": self.threshold,
            "candidates_k": self.candidates_k, "top_k": self.top_k,
            "rerank_s": 0.0, "rerank_load_s": 0.0, "candidates": [],
        }
        try:
            with self._lock:
                self._refresh()
                mask = self._mask(self.strategy, self.lang)
                result["total"] = int(mask.sum())
                if not result["total"]:
                    raise RagError(f"в индексе нет кусков {self.strategy}/{self.lang}")
                result["load_s"] = self._load_model()
                step = time.perf_counter()
                vector = self._embedder.embed_query(question)
                result["embed_s"] = time.perf_counter() - step
                if not rerank:
                    result["hits"] = [hit.as_dict() for hit in self._ranked(mask, vector, self.top_k)]
                else:
                    self._second_stage(result, question, self._ranked(mask, vector, self.candidates_k))
            result["ok"] = True
        except RagError as exc:
            result["error"] = str(exc)
        except Exception as exc:  # noqa: BLE001 — сбой поиска не должен отменять ход
            logger.exception("[RAG] поиск не удался")
            result["error"] = f"поиск: {exc}"
        result["elapsed"] = max(
            time.perf_counter() - started - result["load_s"] - result["rerank_load_s"], 0.0,
        )
        return result

    def _second_stage(self, result: dict, query: str, stage1: list[Hit]) -> None:
        """Этап 2 поиска (день 23, §4.2): дописывает в `result` выдачу,
        кандидатов и условия. Под замком; не бросает — сбой реранкера
        превращается в выдачу этапа 1 с причиной (§2.3)."""
        try:
            ranked, load_s, score_s = self._rerank(query, stage1)
        except RagError as exc:
            result["rerank_error"] = str(exc)
        except Exception as exc:  # noqa: BLE001 — реранкер не должен отменять поиск
            logger.exception("[RAG] реранкер не сработал")
            result["rerank_error"] = f"второй этап: {exc}"
        else:
            result["rerank_load_s"] = load_s
            result["rerank_s"] = score_s
            result["rerank_ok"] = True
            kept = [hit for hit in ranked if hit.rerank_score >= self.threshold][:self.top_k]
            kept_ids = {hit.chunk_id for hit in kept}
            result["hits"] = [hit.as_dict() for hit in kept]
            by_id = {hit.chunk_id: (place, hit) for place, hit in enumerate(ranked, 1)}
            candidates = []
            for hit in stage1:
                place, scored = by_id[hit.chunk_id]
                if hit.chunk_id in kept_ids:
                    fate = "в выдаче"
                elif scored.rerank_score < self.threshold:
                    fate = "ниже порога"
                else:
                    fate = "сверх top-K"
                candidates.append({
                    "chunk_id": hit.chunk_id, "doc": hit.doc, "title": hit.title, "section": hit.section,
                    "part": hit.part, "page_from": hit.page_from, "page_to": hit.page_to,
                    "tokens": hit.tokens, "score": hit.score, "rerank_score": scored.rerank_score,
                    "rerank_place": place, "fate": fate,
                })
            result["candidates"] = candidates
            return
        # Сбой: в запрос уходят `top_k` ближайших этапа 1, без оценок.
        result["hits"] = [hit.as_dict() for hit in stage1[:self.top_k]]
        logger.warning("[RAG] второй этап не удался: %s — выдача без фильтра", result["rerank_error"])


__all__ = [
    "Hit", "IndexFileError", "RagError", "RulesIndex", "SCHEMA_VERSION", "TABLES", "check_index", "shown", "sources_found",
]
