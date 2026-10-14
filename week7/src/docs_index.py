"""Программа индексации Markdown: пересчёт изменённых файлов, атомарная запись."""

import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import sqlite3
import subprocess
import time

from embedder import Embedder, Reranker
from md_chunks import chunk_markdown
from search import SCHEMA, DocsIndex, IndexInvalid, corpus_files, read_index
import settings as cfg


def log(text):
    print(f"[Индекс] {text}", flush=True)


def repo_metadata(repo: Path) -> str:
    env = {**os.environ, "GIT_OPTIONAL_LOCKS": "0", "GIT_TERMINAL_PROMPT": "0", "LC_ALL": "C"}
    def git(*args):
        result = subprocess.run(["git", "-C", str(repo), "--no-pager", "rev-parse", *args],
                                env=env, capture_output=True, text=True, timeout=10)
        if result.returncode:
            raise ValueError(result.stderr.strip())
        return result.stdout.strip()
    if not repo.is_dir() or Path(git("--show-toplevel")).resolve() != repo:
        raise ValueError("--repo должен быть корнем git-репозитория")
    return git("HEAD")


def write_index(path, meta, files, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    # Не трогаем уже существующий временный файл другого запуска.
    fd = os.open(temporary, os.O_CREAT | os.O_EXCL | os.O_WRONLY, 0o600)
    os.close(fd)
    try:
        with sqlite3.connect(temporary) as db:
            db.execute("BEGIN")
            db.execute(f"PRAGMA user_version={SCHEMA}")
            db.execute("CREATE TABLE meta(key TEXT PRIMARY KEY, value TEXT NOT NULL)")
            db.execute("CREATE TABLE files(doc TEXT PRIMARY KEY, sha256 TEXT, size INTEGER, chunks INTEGER)")
            db.execute("""CREATE TABLE chunks(id INTEGER PRIMARY KEY, doc TEXT, day INTEGER,
                       kind TEXT, section TEXT, part INTEGER, parts INTEGER, line INTEGER,
                       text TEXT, tokens INTEGER, vec BLOB)""")
            db.executemany("INSERT INTO meta VALUES (?,?)", [(key, str(value)) for key, value in meta.items()])
            db.executemany("INSERT INTO files VALUES (:doc,:sha256,:size,:chunks)", files)
            db.executemany("""INSERT INTO chunks(doc,day,kind,section,part,parts,line,text,tokens,vec)
                           VALUES (:doc,:day,:kind,:section,:part,:parts,:line,:text,:tokens,:vec)""", rows)
        os.replace(temporary, path)
    finally:
        temporary.unlink(missing_ok=True)


def build(args, paths, head, previous):
    started = time.monotonic()
    old_meta, old_files, old_rows = previous or ({}, {}, [])
    expected = {"schema": str(SCHEMA), "embed_model": cfg.EMBED_MODEL,
                "chunk_max_chars": str(cfg.CHUNK_MAX_CHARS), "repo": str(args.repo)}
    reasons = {"schema": "другая схема", "embed_model": "другая модель",
               "chunk_max_chars": "другая нарезка", "repo": "другой репозиторий"}
    reason = "--full" if args.full else "индекса нет" if not previous else next(
        (reasons[key] for key, value in expected.items() if old_meta.get(key) != value), None)
    if reason:
        log(f"пересчёт всего корпуса: {reason}")
    else:
        log("пересчёт только изменённых файлов (SHA-256)")
    reusable = {}
    for row in old_rows:
        reusable.setdefault(row["doc"], []).append(row)
    files, rows, pending = [], [], []
    kept = recalculated = 0
    for path in paths:
        raw = path.read_bytes()
        doc = path.relative_to(args.repo).as_posix()
        digest = hashlib.sha256(raw).hexdigest()
        if reason is None and old_files.get(doc, {}).get("sha256") == digest:
            chunks = reusable.get(doc, [])
            rows.extend(chunks)
            kept += 1
        else:
            chunks = chunk_markdown(raw.decode("utf-8"), doc, cfg.CHUNK_MAX_CHARS)
            pending.extend(chunks)
            recalculated += 1
        files.append({"doc": doc, "sha256": digest, "size": len(raw), "chunks": len(chunks)})
    slicing_time = time.monotonic() - started
    if not rows and not pending:
        raise ValueError("в корпусе нет непустых разделов")
    loading = time.monotonic()
    embedder = Embedder(cfg.EMBED_MODEL, cfg.MODELS_DIR).load(allow_download=True)
    # Обычная сборка тоже готовит реранкер: help никогда не скачивает модели.
    Reranker(cfg.RERANK_MODEL, cfg.MODELS_DIR).load(allow_download=True)
    loading_time = time.monotonic() - loading
    embedding = time.monotonic()
    last_progress = embedding
    for start in range(0, len(pending), cfg.EMBED_BATCH_SIZE):
        batch = pending[start:start + cfg.EMBED_BATCH_SIZE]
        texts = [chunk.embedding_text for chunk in batch]
        vectors = embedder.embed_passages(texts)
        for chunk, vector, text in zip(batch, vectors, texts):
            day = re.search(r"День (\d+)", chunk.doc, re.IGNORECASE)
            rows.append({"doc": chunk.doc, "day": int(day[1]) if day else None,
                         "kind": "current" if chunk.doc in cfg.CURRENT_DOCS else "docs",
                         "section": chunk.section, "part": chunk.part, "parts": chunk.parts,
                         "line": chunk.line, "text": chunk.text,
                         "tokens": embedder.count_tokens(text), "vec": vector.astype("float32").tobytes()})
        done = min(start + len(batch), len(pending))
        now = time.monotonic()
        if now - last_progress >= 10 or done == len(pending):
            remaining = (now - embedding) / done * (len(pending) - done)
            log(f"{done} из {len(pending)} кусков, ≈{remaining:.0f} с осталось")
            last_progress = now
    embedding_time = time.monotonic() - embedding
    # Подписи обновляются даже у сохранённых векторов, если изменили CURRENT_DOCS.
    for row in rows:
        row["kind"] = "current" if row["doc"] in cfg.CURRENT_DOCS else "docs"
    meta = {**expected, "dim": len(rows[0]["vec"]) // 4, "globs": json.dumps(cfg.DOCS_GLOBS),
            "head": head, "built_at": datetime.now(timezone.utc).isoformat(),
            "rerank_model": cfg.RERANK_MODEL}
    writing = time.monotonic()
    write_index(args.db, meta, files, rows)
    writing_time = time.monotonic() - writing
    deleted = len(old_files.keys() - {file["doc"] for file in files})
    lengths = [len(row["text"]) for row in rows]
    overlong = [row for row in rows if row["tokens"] > 512]
    log(f"файлов {len(files)}: сохранено {kept}, пересчитано {recalculated}, удалено {deleted}; кусков {len(rows)}")
    log(f"пересчитано: {len(pending)} кусков, {sum(len(chunk.text) for chunk in pending)} символов")
    log(f"длина кусков: средняя {sum(lengths) / len(lengths):.0f}, максимум {max(lengths)}; "
        f"токены: максимум {max(row['tokens'] for row in rows)}, длиннее 512 — {len(overlong)}")
    if overlong:
        log("⚠️ модель обрезает куски длиннее 512 токенов: " + ", ".join(sorted({row['doc'] for row in overlong})))
    log(f"нарезка {slicing_time:.2f} с · загрузка моделей {loading_time:.2f} с · "
        f"эмбеддинги {embedding_time:.2f} с · запись {writing_time:.2f} с · всего {time.monotonic() - started:.2f} с")
    log(f"HEAD {head}; индекс {args.db}")


def probe(args):
    index = DocsIndex(args.db, cfg.MODELS_DIR, rerank_model=cfg.RERANK_MODEL,
                      candidates=cfg.SEARCH_CANDIDATES, top_k=cfg.SEARCH_TOP_K)
    index.load_models(allow_download=True)
    for warning in index.warnings(args.repo, cfg.DOCS_GLOBS, cfg.DOCS_EXCLUDE):
        log("⚠️ " + warning)
    for number, question in enumerate(cfg.PROBE_QUESTIONS, 1):
        result = index.search(question)
        log(f"В{number}: {question} · поиск {result.elapsed:.2f} с, реранк {result.rerank_elapsed:.2f} с")
        if result.rerank_error:
            log(f"⚠️ реранкер: {result.rerank_error}")
        for position, hit in enumerate(result.hits, 1):
            score = "—" if hit.score is None else f"{hit.score:.3f}"
            log(f"[{position}] sim={hit.sim:.3f}, score={score}, до реранка #{hit.rank}: {hit.label()}")


def main():
    parser = argparse.ArgumentParser(description="Индекс документации проекта")
    parser.add_argument("--repo", type=Path, default=cfg.DEFAULT_REPO)
    parser.add_argument("--db", type=Path, default=cfg.INDEX_DB)
    parser.add_argument("--full", action="store_true", help="пересчитать все файлы")
    parser.add_argument("--probe", action="store_true", help="поиск по готовому индексу без пересборки")
    args = parser.parse_args()
    args.repo, args.db = args.repo.expanduser().resolve(), args.db.expanduser().resolve()
    try:
        head = repo_metadata(args.repo)
        paths, warnings = corpus_files(args.repo, cfg.DOCS_GLOBS, cfg.DOCS_EXCLUDE)
        for warning in warnings:
            log("⚠️ " + warning)
        if not paths:
            raise ValueError("ни один шаблон не нашёл документации")
        previous = read_index(args.db) if args.db.exists() else None
        if args.probe and previous is None:
            raise IndexInvalid("индекс не найден — ./week7/run.sh index")
    except (ValueError, OSError, subprocess.SubprocessError) as exc:
        log(f"отказ: {exc}")
        return 2
    try:
        probe(args) if args.probe else build(args, paths, head, previous)
        return 0
    except KeyboardInterrupt:
        log("прервано; прежний индекс сохранён")
        return 130
    except Exception as exc:
        log(f"сбой: {exc}")
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
