# TooManyRules — индекс правил: программа индексации (день 21, неделя 5).
#
# **Отдельная программа**, как серверы дней 17-20, но запускает её человек
# (`./run.sh index`), и работает она до конца, а не живёт. Приложение её не
# импортирует; индекс читает `rag_search.py` (приложение и `--probe`).
#
# Пайплайн (спецификация дня 21, §2.1):
#   1. извлечение — PDF → строки с номером страницы, шрифтом и кеглем;
#   2. очистка   — правила документа из `presets.py` (`rag_chunks.clean()`);
#   3. нарезка   — одна очищенная копия → «фиксированный размер» и «по
#                  структуре»;
#   4. эмбеддинги — локальная многоязычная модель (`embedder.py`);
#   5. сохранение — SQLite, одной транзакцией во временный файл и
#                  `os.replace()`;
#   6. отчёт     — числа сравнения стратегий в терминале, без суждений.
# `--probe` — пробные вопросы по готовому индексу: три ближайших куска каждой
# стратегии, без LLM, порогов и сборки ответа.
#
# **Единственное место, где импортируется `pymupdf`**, и единственное место
# записи базы индекса; читает и проверяет файл (`check_index()`) с дня 22
# `rag_search.py`, поэтому `--probe` ищет той же `RulesIndex.nearest()`, что и
# ассистент. Про Too Many Bones знает только через `presets.py`
# (какие документы, их правила, модель, параметры нарезки, пробные вопросы).
# LLM API не вызывается вовсе; сеть — только однократное скачивание модели.
#
# Логи — stdout, префикс `[Индекс]`, строка на шаг: что сделал, сколько и за
# сколько. Отчёт и ответы `--probe` — обычным текстом без префикса.

import argparse
import hashlib
import json
import logging
import os
import sqlite3
import sys
import time
from collections import Counter
from contextlib import closing
from datetime import datetime
from pathlib import Path

import numpy as np
import pymupdf

import presets
import rag_chunks
from embedder import Embedder, package_version
from rag_search import SCHEMA_VERSION, IndexFileError, RagError, RulesIndex, check_index

logger = logging.getLogger("rag_index")

ROOT = Path(__file__).resolve().parent.parent
PROBE_TOP = 3
PREVIEW_CHARS = 120
# Текст и шрифты — да, картинки — нет: их блоки в `dict` не нужны.
TEXT_FLAGS = pymupdf.TEXTFLAGS_DICT & ~pymupdf.TEXT_PRESERVE_IMAGES

SCHEMA = """
CREATE TABLE meta (
    key   TEXT PRIMARY KEY,
    value TEXT NOT NULL
);
CREATE TABLE documents (
    doc         TEXT NOT NULL,
    lang        TEXT NOT NULL,
    publisher   TEXT NOT NULL,
    game        TEXT NOT NULL,
    edition     TEXT NOT NULL,
    title       TEXT NOT NULL,
    structure   TEXT NOT NULL,
    source      TEXT NOT NULL,
    sha256      TEXT NOT NULL,
    pages       INTEGER NOT NULL,
    chars_raw   INTEGER NOT NULL,
    chars_clean INTEGER NOT NULL,
    dropped     TEXT NOT NULL,
    PRIMARY KEY (doc, lang)
);
CREATE TABLE chunks (
    chunk_id         TEXT PRIMARY KEY,
    strategy         TEXT NOT NULL,
    doc              TEXT NOT NULL,
    lang             TEXT NOT NULL,
    ordinal          INTEGER NOT NULL,
    source           TEXT NOT NULL,
    title            TEXT NOT NULL,
    section          TEXT NOT NULL,
    part             TEXT,
    sections_spanned INTEGER NOT NULL,
    page_from        INTEGER NOT NULL,
    page_to          INTEGER NOT NULL,
    char_start       INTEGER NOT NULL,
    char_end         INTEGER NOT NULL,
    publisher        TEXT NOT NULL,
    game             TEXT NOT NULL,
    edition          TEXT NOT NULL,
    text             TEXT NOT NULL,
    chars            INTEGER NOT NULL,
    tokens           INTEGER NOT NULL,
    embedding        BLOB NOT NULL,
    FOREIGN KEY (doc, lang) REFERENCES documents (doc, lang)
);
CREATE INDEX chunks_strategy_lang ON chunks (strategy, lang);
"""


# --- Мелочи вывода -------------------------------------------------------------

def shown(path: Path) -> str:
    try:
        return str(path.resolve().relative_to(ROOT))
    except ValueError:
        return str(path)


def num(value: int) -> str:
    return f"{value:,}".replace(",", " ")


def chunks_word(count: int) -> str:
    if count % 10 == 1 and count % 100 != 11:
        word = "кусок"
    elif count % 10 in (2, 3, 4) and count % 100 not in (12, 13, 14):
        word = "куска"
    else:
        word = "кусков"
    return f"{num(count)} {word}"


def pages_text(first: int, last: int) -> str:
    return f"стр. {first}" if first == last else f"стр. {first}–{last}"


# --- Индекс на диске -----------------------------------------------------------

def write_index(
    path: Path,
    meta: dict[str, str],
    cleans: list[rag_chunks.CleanDoc],
    shas: dict[str, str],
    chunks: list[rag_chunks.Chunk],
    tokens: dict[str, int],
    vectors: dict[str, np.ndarray],
) -> None:
    """Индекс целиком — во временный файл рядом, одной транзакцией, и на
    место через `os.replace()`: прерванная сборка оставляет прежний индекс
    целым (§4.3)."""
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(path.name + ".tmp")
    temporary.unlink(missing_ok=True)
    try:
        with closing(sqlite3.connect(temporary, isolation_level=None)) as conn:
            conn.execute("BEGIN")
            # Схема — по одному оператору, а не `executescript()`: тот сам
            # коммитит, и индекс перестал бы быть одной транзакцией.
            for statement in SCHEMA.split(";"):
                if statement.strip():
                    conn.execute(statement)
            conn.executemany("INSERT INTO meta (key, value) VALUES (?, ?)", sorted(meta.items()))
            conn.executemany(
                "INSERT INTO documents VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        c.source.doc, c.source.lang, c.source.publisher, c.source.game, c.source.edition,
                        c.source.title, c.source.structure, c.source.path, shas[c.source.key], c.pages,
                        c.chars_raw, len(c.text), json.dumps(c.dropped, ensure_ascii=False),
                    )
                    for c in cleans
                ],
            )
            conn.executemany(
                "INSERT INTO chunks VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)",
                [
                    (
                        ch.chunk_id, ch.strategy, ch.doc, ch.lang, ch.ordinal, ch.source, ch.title,
                        ch.section, ch.part, ch.sections_spanned, ch.page_from, ch.page_to,
                        ch.char_start, ch.char_end, ch.publisher, ch.game, ch.edition, ch.text,
                        ch.chars, tokens[ch.chunk_id],
                        vectors[ch.chunk_id].astype("<f4").tobytes(),
                    )
                    for ch in chunks
                ],
            )
            conn.execute(f"PRAGMA user_version = {SCHEMA_VERSION}")
            conn.execute("COMMIT")
        os.replace(temporary, path)
    except BaseException:
        temporary.unlink(missing_ok=True)
        raise


# --- Извлечение ------------------------------------------------------------------

def extract(path: Path) -> tuple[list[rag_chunks.Line], int]:
    """Строки PDF в порядке PyMuPDF и число страниц. Шрифт строки — тот,
    которым набрано больше всего её символов; кегль — самого крупного
    фрагмента; `first_font` — шрифт первого непустого фрагмента."""
    lines: list[rag_chunks.Line] = []
    with pymupdf.open(path) as pdf:
        pages = pdf.page_count
        for number, page in enumerate(pdf, 1):
            for block in page.get_text("dict", flags=TEXT_FLAGS)["blocks"]:
                if block["type"] != 0:
                    continue
                for line in block["lines"]:
                    spans = line["spans"]
                    text = "".join(span["text"] for span in spans)
                    solid = [span for span in spans if span["text"].strip()]
                    if not solid:
                        continue
                    weight: Counter[str] = Counter()
                    for span in solid:
                        weight[span["font"]] += len(span["text"].strip())
                    x, y = line["bbox"][0], line["bbox"][1]
                    lines.append(rag_chunks.Line(
                        page=number, block=block["number"], text=text,
                        font=weight.most_common(1)[0][0],
                        size=round(max(span["size"] for span in solid), 2),
                        first_font=solid[0]["font"], x=round(x, 1), y=round(y, 1),
                    ))
    return lines, pages


def file_sha256(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def dropped_text(dropped: dict[str, int]) -> str:
    parts = [f"{rule}: {num(count)}" for rule, count in dropped.items() if count]
    return "; ".join(parts) or "ничего"


def write_dump(directory: Path, cleans: list[rag_chunks.CleanDoc]) -> list[Path]:
    """Выгрузка для глаз (§5.1): очищенный текст — абзацы, заголовки с `## `,
    страницы строкой `--- стр. N ---`; выброшенное — по правилам."""
    directory.mkdir(parents=True, exist_ok=True)
    written: list[Path] = []
    for clean_doc in cleans:
        base = f"{clean_doc.source.doc}.{clean_doc.source.lang}"
        out: list[str] = []
        page = 0
        for paragraph in clean_doc.paragraphs:
            if paragraph.page_from != page:
                out.append(f"--- стр. {paragraph.page_from} ---")
                out.append("")
            page = paragraph.page_to
            out.append(("## " if paragraph.kind == "heading" else "") + paragraph.text)
            out.append("")
        clean_path = directory / f"{base}.clean.txt"
        clean_path.write_text("\n".join(out), encoding="utf-8")
        grouped: dict[str, list[tuple[int, str]]] = {rule: [] for rule in clean_doc.dropped}
        for rule, item_page, text in clean_doc.dropped_items:
            grouped[rule].append((item_page, text))
        out = []
        for rule, items in grouped.items():
            out.append(f"# {rule} — {num(clean_doc.dropped[rule])} симв.")
            out.extend(f"стр. {item_page}: {text}" for item_page, text in items)
            out.append("")
        dropped_path = directory / f"{base}.dropped.txt"
        dropped_path.write_text("\n".join(out), encoding="utf-8")
        written += [clean_path, dropped_path]
    return written


# --- Отчёт -------------------------------------------------------------------

def _share(count: int, total: int) -> str:
    return f"{num(count)} ({count / total:.0%})" if total else "—"


def print_report(metrics: list[rag_chunks.Metrics], timing: dict[str, tuple[float, int]]) -> None:
    """Таблица §7: по стратегиям (колонки) и документам (разделы). Только
    числа — слова «лучше» здесь нет, выводы пишет автор."""
    by_scope: dict[str, dict[str, rag_chunks.Metrics]] = {}
    for item in metrics:
        by_scope.setdefault(item.doc, {})[item.strategy] = item
    strategies = list(rag_chunks.STRATEGIES)
    width = 44
    print()
    print("Сравнение стратегий нарезки (только числа; выводы — в документе сравнения)")
    print(" " * width + "".join(f"{name:>22}" for name in strategies))
    for scope, row in by_scope.items():
        print(scope or "все документы")

        def line(label: str, render) -> None:
            print(f"  {label:<{width - 2}}" + "".join(f"{render(row[name]):>22}" for name in strategies))

        line("кусков", lambda m: num(m.chunks))
        line("длина, символов (мин / мед / макс)", lambda m: " / ".join(num(v) for v in m.chars))
        line("длина, токенов модели (мин / мед / макс)", lambda m: " / ".join(num(v) for v in m.tokens))
        line(f"сверх {row[strategies[0]].token_limit} токенов", lambda m: num(m.over_limit))
        line("оборван посреди предложения", lambda m: _share(m.cut_end, m.chunks))
        line("начат посреди предложения", lambda m: _share(m.cut_start, m.chunks))
        line("задел несколько разделов", lambda m: _share(m.multi_section, m.chunks))
        line("разделов разрезано между кусками", lambda m: f"{num(m.units_split)} из {num(m.units)}")
        if row[strategies[0]].entries:
            line("записей глоссария целиком в одном куске", lambda m: f"{num(m.entries_whole)} из {num(m.entries)}")
        line("объём индекса / очищенный текст", lambda m: f"{m.volume:.2f}")
        if not scope:
            line(
                "эмбеддинги, с (кусков в секунду)",
                lambda m: f"{timing[m.strategy][0]:.1f} ({timing[m.strategy][1] / timing[m.strategy][0]:.1f})"
                if timing[m.strategy][0] else "—",
            )
    print()


# --- Сборка ----------------------------------------------------------------------

def build(db: Path, sources: Path, dump: Path | None) -> int:
    started = time.perf_counter()
    documents = presets.RAG_DOCUMENTS
    missing = [source for source in documents if not (sources / source.path).is_file()]
    if missing:
        for source in missing:
            logger.error("нет файла документа %s: %s", source.key, shown(sources / source.path))
        logger.error("сборка отменена: индекс без всех документов хуже прежнего — %s не изменён", shown(db))
        return 2
    try:
        check_index(db)
    except IndexFileError as exc:
        logger.error("по пути индекса %s: %s — сборка отменена, файл не изменён", shown(db), exc)
        return 2

    langs = Counter(source.lang for source in documents)
    logger.info(
        "документы: %d (%s) · %s/", len(documents),
        ", ".join(f"{lang}: {count}" for lang, count in langs.items()), shown(sources),
    )
    cleans: list[rag_chunks.CleanDoc] = []
    shas: dict[str, str] = {}
    for source in documents:
        step = time.perf_counter()
        path = sources / source.path
        shas[source.key] = file_sha256(path)
        lines, pages = extract(path)
        clean_doc = rag_chunks.clean(source, lines, pages)
        cleans.append(clean_doc)
        logger.info(
            "%s: %d стр. · %s → %s симв. · абзацев %d · выброшено: %s · %.2f с",
            source.key, pages, num(clean_doc.chars_raw), num(len(clean_doc.text)),
            len(clean_doc.paragraphs), dropped_text(clean_doc.dropped), time.perf_counter() - step,
        )
    if dump is not None:
        written = write_dump(dump, cleans)
        logger.info("выгрузка --dump: %d файлов в %s/", len(written), shown(dump))

    step = time.perf_counter()
    fixed = [
        chunk for clean_doc in cleans
        for chunk in rag_chunks.fixed_chunks(clean_doc, presets.RAG_FIXED_CHARS, presets.RAG_FIXED_OVERLAP)
    ]
    structural = [
        chunk for clean_doc in cleans
        for chunk in rag_chunks.structural_chunks(clean_doc, presets.RAG_MAX_CHARS)
    ]
    logger.info(
        "нарезка fixed (%d, перекрытие %d): %s · structural (до %d): %s · %.2f с",
        presets.RAG_FIXED_CHARS, presets.RAG_FIXED_OVERLAP, chunks_word(len(fixed)),
        presets.RAG_MAX_CHARS, chunks_word(len(structural)), time.perf_counter() - step,
    )

    embedder = Embedder(presets.RAG_EMBED_MODEL)
    step = time.perf_counter()

    def on_download() -> None:
        logger.info(
            "модели %s нет в кэше — скачиваю ≈1,1 ГБ в кэш Hugging Face (~/.cache/huggingface): "
            "один раз, несколько минут", presets.RAG_EMBED_MODEL,
        )

    embedder.load(on_download)
    loaded = time.perf_counter() - step
    if embedder.downloaded:
        logger.info("модель скачана и загружена за %.1f с", loaded)
    limit = embedder.max_seq_length
    logger.info(
        "модель %s · устройство %s · %d измерений · до %d токенов · загрузка %.1f с",
        presets.RAG_EMBED_MODEL, embedder.device, embedder.dim, limit, loaded,
    )

    tokens: dict[str, int] = {}
    vectors: dict[str, np.ndarray] = {}
    timing: dict[str, tuple[float, int]] = {}
    for strategy, chunks in (("fixed", fixed), ("structural", structural)):
        step = time.perf_counter()
        for chunk in chunks:
            tokens[chunk.chunk_id] = embedder.count_tokens(chunk.text)
        matrix = embedder.embed_passages([chunk.text for chunk in chunks])
        elapsed = time.perf_counter() - step
        for chunk, vector in zip(chunks, matrix, strict=True):
            vectors[chunk.chunk_id] = vector
        timing[strategy] = (elapsed, len(chunks))
        counts = [tokens[chunk.chunk_id] for chunk in chunks]
        over = sum(1 for count in counts if count > limit)
        logger.info(
            "эмбеддинги %s: %s · %.1f с · %.1f куск./с · токенов до %d · сверх %d токенов: %d",
            strategy, chunks_word(len(chunks)), elapsed, len(chunks) / elapsed if elapsed else 0.0,
            max(counts, default=0), limit, over,
        )
        if over:
            logger.warning(
                "%s: %d кусков длиннее %d токенов — модель молча отбросит их хвосты; "
                "уменьшите RAG_FIXED_CHARS/RAG_MAX_CHARS", strategy, over, limit,
            )

    meta = {
        "embed_model": presets.RAG_EMBED_MODEL,
        "embed_dim": str(embedder.dim),
        "max_seq_length": str(limit),
        "built_at": datetime.now().astimezone().isoformat(timespec="seconds"),
        "fixed_chars": str(presets.RAG_FIXED_CHARS),
        "fixed_overlap": str(presets.RAG_FIXED_OVERLAP),
        "max_chars": str(presets.RAG_MAX_CHARS),
        "device": embedder.device,
        "pymupdf": package_version("pymupdf"),
        **{name: value for name, value in embedder.versions().items()},
    }
    step = time.perf_counter()
    write_index(db, meta, cleans, shas, fixed + structural, tokens, vectors)
    logger.info(
        "запись %s · %.1f МБ · %s · %.2f с", shown(db), db.stat().st_size / 1_048_576,
        chunks_word(len(fixed) + len(structural)), time.perf_counter() - step,
    )

    print_report(rag_chunks.compare(fixed + structural, cleans, tokens, limit), timing)
    logger.info("готово за %.1f с", time.perf_counter() - started)
    return 0


# --- Пробные вопросы -------------------------------------------------------------

def probe(db: Path) -> int:
    """Пробные вопросы: три ближайших куска каждой стратегии. Ищет той же
    `RulesIndex.nearest()`, что и ассистент (день 22): проба и ответ не
    расходятся. Скачивание модели разрешено — как при сборке."""
    index = RulesIndex(
        db, presets.RAG_SEARCH_STRATEGY, presets.RAG_SEARCH_LANG, PROBE_TOP,
        allow_download=True, expected_model=presets.RAG_EMBED_MODEL,
    )
    info = index.info()
    if not info["ok"]:
        logger.error("%s", info["error"])
        return 2
    strategies = sorted(info["by_strategy"])
    try:
        step = time.perf_counter()
        index.load_model()
        logger.info(
            "индекс %s · собран %s · %s · модель %s · устройство %s · загрузка %.1f с",
            shown(db), info["built_at"],
            ", ".join(f"{name}: {chunks_word(info['by_strategy'][name])}" for name in strategies),
            info["model"], index.device, time.perf_counter() - step,
        )
        for number, (lang, question) in enumerate(presets.RAG_PROBE_QUESTIONS, 1):
            step = time.perf_counter()
            vector = index.query_vector(question)
            query_s = time.perf_counter() - step
            print()
            print(f"{number}. [{lang}] {question}   (вектор вопроса {query_s:.2f} с)")
            for strategy in strategies:
                print(f"   {strategy}")
                try:
                    hits = index.nearest(question, strategy, lang, PROBE_TOP, vector=vector)
                except RagError:
                    hits = []
                if not hits:
                    print("     кусков на этом языке нет")
                for hit in hits:
                    part = f" · часть {hit.part}" if hit.part else ""
                    print(
                        f"     {hit.score:.3f}  {hit.chunk_id} · {hit.section}{part} · "
                        f"{pages_text(hit.page_from, hit.page_to)}"
                    )
                    preview = " ".join(hit.text.split())
                    if len(preview) > PREVIEW_CHARS:
                        preview = preview[:PREVIEW_CHARS].rstrip() + "…"
                    print(f"            «{preview}»")
    except RagError as exc:
        logger.error("%s", exc)
        return 2
    print()
    return 0


# --- Точка входа -----------------------------------------------------------------

def _args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        prog="./run.sh index",
        description=(
            "Индекс правил: PDF → очистка → куски двумя стратегиями → эмбеддинги локальной "
            "моделью → SQLite. Документы, правила очистки и модель — в presets.py."
        ),
    )
    mode = parser.add_mutually_exclusive_group()
    mode.add_argument(
        "--probe", action="store_true",
        help="пробные вопросы по готовому индексу — три ближайших куска каждой стратегии, без пересборки",
    )
    mode.add_argument(
        "--dump", type=Path, metavar="КАТАЛОГ",
        help="сборка + выгрузка очищенного текста и выброшенного по правилам в каталог",
    )
    parser.add_argument(
        "--db", type=Path, default=presets.RAG_INDEX_DB, metavar="ПУТЬ",
        help=f"файл индекса (по умолчанию {shown(presets.RAG_INDEX_DB)})",
    )
    parser.add_argument(
        "--sources", type=Path, default=presets.RAG_SOURCES_DIR, metavar="КАТАЛОГ",
        help=f"каталог PDF (по умолчанию {shown(presets.RAG_SOURCES_DIR)}/)",
    )
    return parser.parse_args(argv)


def main() -> None:
    args = _args()
    handler = logging.StreamHandler(sys.stdout)
    handler.setFormatter(logging.Formatter("[Индекс] %(message)s"))
    logger.addHandler(handler)
    logger.setLevel(logging.INFO)
    logger.propagate = False
    db = args.db.expanduser()
    try:
        if args.probe:
            code = probe(db)
        else:
            dump = args.dump.expanduser() if args.dump is not None else None
            code = build(db, args.sources.expanduser(), dump)
    except KeyboardInterrupt:
        logger.error("прервано — прежний индекс %s не тронут", shown(db))
        code = 130
    sys.exit(code)


if __name__ == "__main__":
    main()
