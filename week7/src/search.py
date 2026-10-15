"""Проверка собственного индекса, векторный поиск и реранк."""

from dataclasses import dataclass, replace
import fnmatch
import hashlib
from pathlib import Path
import sqlite3
import time

import numpy as np

from embedder import Embedder, Reranker

SCHEMA = 1
TABLE_COLUMNS = {
    "meta": {"key", "value"},
    "files": {"doc", "sha256", "size", "chunks"},
    "chunks": {"id", "doc", "day", "kind", "section", "part", "parts", "line", "text", "tokens", "vec"},
}
META_KEYS = {"schema", "embed_model", "dim", "repo", "globs", "head", "built_at", "chunk_max_chars"}


class IndexInvalid(ValueError):
    pass


def read_index(path: Path):
    """Открывает строго на чтение; ни чужая, ни пустая база не становится нашей."""
    try:
        with sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True) as db:
            if db.execute("PRAGMA user_version").fetchone()[0] != SCHEMA:
                raise IndexInvalid("чужой файл индекса: неподдерживаемая версия")
            tables = {row[0] for row in db.execute("SELECT name FROM sqlite_master WHERE type='table'")}
            if tables != set(TABLE_COLUMNS):
                raise IndexInvalid("чужой файл индекса: другая схема таблиц")
            for table, columns in TABLE_COLUMNS.items():
                if {row[1] for row in db.execute(f"PRAGMA table_info({table})")} != columns:
                    raise IndexInvalid("чужой файл индекса: другие столбцы")
            meta = dict(db.execute("SELECT key,value FROM meta"))
            if not META_KEYS <= meta.keys() or int(meta["schema"]) != SCHEMA or int(meta["dim"]) < 1:
                raise IndexInvalid("чужой файл индекса: нет метаданных")
            db.row_factory = sqlite3.Row
            files = {row["doc"]: dict(row) for row in db.execute("SELECT * FROM files")}
            chunks = [dict(row) for row in db.execute("SELECT * FROM chunks ORDER BY id")]
            for row in chunks:
                if len(row["vec"]) != int(meta["dim"]) * 4 or row["doc"] not in files:
                    raise IndexInvalid("повреждённый индекс: векторы или файлы")
            return meta, files, chunks
    except (sqlite3.Error, OSError, ValueError, TypeError) as exc:
        if isinstance(exc, IndexInvalid):
            raise
        raise IndexInvalid(f"чужой или повреждённый файл индекса: {exc}") from exc


def corpus_files(repo: Path, globs, exclude=()) -> tuple[list[Path], list[str]]:
    found = set()
    warnings = []
    for pattern in globs:
        matches = []
        excluded = False
        for path in repo.glob(pattern):
            if path.name in {"AGENTS.md", "CLAUDE.md"} or not path.is_file():
                continue
            if not path.resolve().is_relative_to(repo.resolve()):
                continue
            if any(fnmatch.fnmatch(path.relative_to(repo).as_posix(), rule) for rule in exclude):
                excluded = True
                continue
            matches.append(path)
        # Шаблон, чьи файлы целиком сняты DOCS_EXCLUDE, — осознанное решение, не пропажа.
        if not matches and not excluded:
            warnings.append(f"шаблон {pattern} не нашёл файлов")
        found.update(matches)
    return sorted(found), warnings


def file_hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


@dataclass(frozen=True)
class Hit:
    doc: str
    day: int | None
    kind: str
    section: str
    part: int
    parts: int
    line: int
    text: str
    sim: float
    score: float | None
    rank: int

    def label(self) -> str:
        note = " · текущее состояние" if self.kind == "current" else f" · день {self.day}" if self.day else ""
        part = f" · часть {self.part}/{self.parts}" if self.parts > 1 else ""
        return f"{self.doc}:{self.line} · {self.section}{part}{note}"


@dataclass
class SearchResult:
    hits: list[Hit]
    elapsed: float
    rerank_elapsed: float
    rerank_error: str | None = None


class DocsIndex:
    def __init__(self, db_path: Path, models_dir: Path, *, rerank_model: str,
                 candidates: int = 20, top_k: int = 5):
        if not db_path.exists():
            raise IndexInvalid("индекс не найден — ./week7/run.sh index")
        self.meta, self.files, self.chunks = read_index(db_path)
        if not self.chunks:
            raise IndexInvalid("индекс пуст — ./week7/run.sh index")
        self.matrix = np.stack([np.frombuffer(row["vec"], dtype=np.float32) for row in self.chunks])
        if not np.isfinite(self.matrix).all():
            raise IndexInvalid("повреждённый индекс: нечисловые векторы")
        self.embedder = Embedder(self.meta["embed_model"], models_dir)
        self.reranker = Reranker(self.meta.get("rerank_model", rerank_model), models_dir)
        self.candidates, self.top_k = candidates, top_k

    def load_models(self, allow_download: bool = False):
        self.embedder.load(allow_download)
        self.reranker.load(allow_download)

    def warnings(self, repo: Path, globs, exclude=()) -> list[str]:
        warnings = []
        if self.meta["repo"] != str(repo.resolve()):
            warnings.append(f"индекс собран для другого репозитория: {self.meta['repo']}")
        paths, missing = corpus_files(repo, globs, exclude)
        warnings.extend(missing)
        hashes = {path.relative_to(repo).as_posix(): file_hash(path) for path in paths}
        changed = sum(hashes.get(doc) != self.files.get(doc, {}).get("sha256") for doc in hashes.keys() | self.files.keys())
        if changed:
            warnings.append(f"корпус менялся после сборки индекса ({changed} файлов) — ./week7/run.sh index")
        return warnings

    def search(self, question: str, kinds=None, exclude_docs=()) -> SearchResult:
        started = time.monotonic()
        eligible = np.array([i for i, row in enumerate(self.chunks)
                             if (kinds is None or row["kind"] in kinds) and row["doc"] not in exclude_docs],
                            dtype=np.intp)
        if not len(eligible):
            return SearchResult([], time.monotonic() - started, 0)
        similarities = self.matrix[eligible] @ self.embedder.embed_query(question)
        nearest = np.argsort(-similarities, kind="stable")[:self.candidates]
        hits = []
        for rank, index in enumerate(nearest, 1):
            row = self.chunks[int(eligible[index])]
            hits.append(Hit(**{key: row[key] for key in ("doc", "day", "kind", "section", "part", "parts", "line", "text")},
                            sim=float(similarities[index]), score=None, rank=rank))
        rerank_started = time.monotonic()
        error = None
        try:
            scores = self.reranker.score(question, [hit.text for hit in hits])
            if len(scores) != len(hits) or not np.isfinite(scores).all():
                raise ValueError("реранкер вернул некорректные оценки")
            hits = sorted([replace(hit, score=score) for hit, score in zip(hits, scores)],
                          key=lambda hit: hit.score, reverse=True)
        except Exception as exc:
            error = str(exc)
        return SearchResult(hits[:self.top_k], time.monotonic() - started,
                            time.monotonic() - rerank_started, error)
