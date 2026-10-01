# TooManyRules — локальные модели недели 5: эмбеддинги (день 21) и, с дня 23,
# переранжирование (`Reranker`).
#
# **Единственное место, где импортируется `sentence_transformers`** (в том числе
# `CrossEncoder`) — как `agent.py` для LLM API и `mcp_client.py` для SDK `mcp`.
# Модуль берут программа индексации (`rag_index.py`) и, с дня 22, поиск по
# индексу (`rag_search.py`). Про документы, куски и индекс модуль не знает: на
# входе строки, на выходе векторы `float32`, нормированные (косинусная близость
# = скалярное произведение), или оценки релевантности 0..1.
#
# Модель — многоязычная (спецификация дня 21, §3): русские и английские куски
# лягут в одно пространство векторов. У семейства E5 префиксы обязательны:
# куски — `passage: `, вопросы — `query: `; без них качество заметно падает, а
# ошибка молчаливая. Поэтому префиксы ставит только этот модуль, вызывающий их
# не пишет. Всё, что длиннее `max_seq_length` токенов, модель молча
# отбрасывает — поэтому `count_tokens()` считает кусок так, как он уйдёт в
# модель: с префиксом и служебными токенами.
#
# Реранкер (день 23, спецификация §3) — cross-encoder: читает пару «запрос, кусок»
# целиком и отдаёт одну оценку 0..1 (сигмоида логита). Медленнее эмбеддингов (пара
# на каждый кусок), поэтому идёт вторым этапом по кандидатам первого. На
# ускорителе (`mps`, `cuda`) — `float16`, на процессоре — `float32`; пары
# оцениваются пакетами малого размера: временная память оценки растёт с пакетом и
# процессу не возвращается (замер — `presets.RAG_RERANK_THRESHOLD`). Префиксов
# `query:`/`passage:` у реранкера нет.
#
# Модель загружается лениво, при первом вызове. Первый раз — скачивание в кэш
# Hugging Face (`~/.cache/huggingface`), дальше — только из кэша
# (`local_files_only=True`): без сети и без проверок обновлений. Логов у модуля
# нет: что скачивается и сколько заняло, пишет вызывающий (`load()` зовёт
# `on_download` перед скачиванием).

import os
from collections.abc import Callable
from importlib.metadata import PackageNotFoundError, version

# До импорта: иначе `tokenizers` предупреждает о fork на каждом запуске.
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

import numpy as np  # noqa: E402

PASSAGE_PREFIX = "passage: "
QUERY_PREFIX = "query: "
BATCH_SIZE = 16
# Файлы весов, по которым `in_cache()` судит, что модель скачана целиком.
WEIGHT_FILES = ("model.safetensors", "model.safetensors.index.json", "pytorch_model.bin")


def in_cache(model_name: str) -> bool:
    """Есть ли модель в кэше Hugging Face — без загрузки и без сети: и
    `config.json`, и файл весов (`config.json` скачивается первым, и по нему
    одному прерванное скачивание выглядело бы готовой моделью — правка по ревью
    дня 23; недокачанный файл в снимок кэша не попадает). Для строки лога при
    старте приложения и отказа программы сравнения."""
    from huggingface_hub import try_to_load_from_cache

    def cached(filename: str) -> bool:
        return isinstance(try_to_load_from_cache(model_name, filename), str)

    return cached("config.json") and any(cached(name) for name in WEIGHT_FILES)


def package_version(name: str) -> str:
    try:
        return version(name)
    except PackageNotFoundError:
        return "н/д"


class Embedder:
    def __init__(self, model_name: str, batch_size: int = BATCH_SIZE) -> None:
        self.model_name = model_name
        self.batch_size = batch_size
        self._model = None
        self.downloaded = False

    def load(self, on_download: Callable[[], None] | None = None, allow_download: bool = True) -> None:
        """Загрузка модели; повторный вызов ничего не делает. Модели нет в
        кэше — `on_download()`, затем скачивание. `allow_download=False`
        (приложение и программа сравнения, день 22) — вместо скачивания
        гигабайта посреди хода игрока исключение с понятным текстом."""
        if self._model is not None:
            return
        # Импорт здесь, а не наверху: `sentence_transformers` тянет `torch` —
        # секунды на старте, которые не нужны, пока модель не понадобилась.
        from sentence_transformers import SentenceTransformer
        from transformers.utils import logging as transformers_logging

        # Из кэша — без полосы «Loading weights» в терминале: загрузка —
        # секунды, её время пишет вызывающий. Полосы скачивания остаются.
        transformers_logging.disable_progress_bar()
        try:
            self._model = SentenceTransformer(self.model_name, local_files_only=True)
        except OSError as exc:
            transformers_logging.enable_progress_bar()
            if not allow_download:
                raise OSError(
                    f"модели {self.model_name} нет в кэше Hugging Face — сначала ./run.sh index "
                    "(скачает ≈1,1 ГБ)"
                ) from exc
            if on_download is not None:
                on_download()
            self._model = SentenceTransformer(self.model_name)
            self.downloaded = True
        finally:
            transformers_logging.enable_progress_bar()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def model(self):
        self.load()
        return self._model

    @property
    def device(self) -> str:
        return str(self.model.device)

    @property
    def dim(self) -> int:
        return int(self.model.get_embedding_dimension())

    @property
    def max_seq_length(self) -> int:
        return int(self.model.max_seq_length)

    @staticmethod
    def versions() -> dict[str, str]:
        return {
            "sentence-transformers": package_version("sentence-transformers"),
            "torch": package_version("torch"),
            "transformers": package_version("transformers"),
        }

    def count_tokens(self, text: str) -> int:
        """Токенов в куске так, как он уйдёт в модель: префикс `passage: ` и
        служебные токены входят в счёт."""
        ids = self.model.tokenizer(PASSAGE_PREFIX + text, add_special_tokens=True, verbose=False)["input_ids"]
        return len(ids)

    def _encode(self, texts: list[str]) -> np.ndarray:
        vectors = self.model.encode(
            texts,
            # Префикс уже в тексте; пустой `prompt` — чтобы префикс по
            # умолчанию из конфига модели, если он там есть, не встал вторым.
            prompt="",
            batch_size=self.batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,
            convert_to_numpy=True,
        )
        return np.asarray(vectors, dtype=np.float32)

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        """Векторы кусков, `(len(texts), dim)`, `float32`, нормированные."""
        if not texts:
            return np.zeros((0, self.dim), dtype=np.float32)
        return self._encode([PASSAGE_PREFIX + text for text in texts])

    def embed_query(self, text: str) -> np.ndarray:
        """Вектор вопроса, `(dim,)`, `float32`, нормированный."""
        return self._encode([QUERY_PREFIX + text])[0]


class Reranker:
    """Переранжирование кусков моделью-cross-encoder (день 23, §3)."""

    def __init__(self, model_name: str, max_length: int = 1024, batch_size: int = 4) -> None:
        self.model_name = model_name
        self.max_length = max_length
        self.batch_size = batch_size
        self._model = None
        self._dtype = ""
        self.downloaded = False

    def load(self, on_download: Callable[[], None] | None = None, allow_download: bool = True) -> None:
        """Загрузка модели; повторный вызов ничего не делает. Модели нет в
        кэше и `allow_download=False` — `OSError` с понятным текстом; иначе
        `on_download()` и скачивание. Импорт `torch` и `CrossEncoder` — здесь:
        старт приложения без `torch`."""
        if self._model is not None:
            return
        import torch
        from sentence_transformers import CrossEncoder
        from transformers.utils import logging as transformers_logging

        accelerated = torch.backends.mps.is_available() or torch.cuda.is_available()
        kwargs = {"model_kwargs": {"torch_dtype": torch.float16}} if accelerated else {}
        self._dtype = "float16" if accelerated else "float32"
        transformers_logging.disable_progress_bar()
        try:
            self._model = CrossEncoder(
                self.model_name, local_files_only=True, max_length=self.max_length, **kwargs,
            )
        except OSError as exc:
            transformers_logging.enable_progress_bar()
            if not allow_download:
                raise OSError(
                    f"модели реранкера {self.model_name} нет в кэше Hugging Face — сначала "
                    "./run.sh index --probe (скачает ≈1,5 ГБ)"
                ) from exc
            if on_download is not None:
                on_download()
            self._model = CrossEncoder(self.model_name, max_length=self.max_length, **kwargs)
            self.downloaded = True
        finally:
            transformers_logging.enable_progress_bar()

    @property
    def loaded(self) -> bool:
        return self._model is not None

    @property
    def device(self) -> str:
        return str(self._model.device) if self._model is not None else "н/д"

    @property
    def dtype(self) -> str:
        return self._dtype if self._model is not None else "н/д"

    def score(self, query: str, passages: list[str]) -> np.ndarray:
        """Оценки релевантности пар «запрос, кусок», `(len(passages),)`,
        `float32`, 0..1, в порядке `passages`. Сигмоида ровно одна: активация
        передаётся явно, а не берётся из конфига модели (у `-en-ru` там та же
        `Sigmoid`; у модели с другой не встанет дважды)."""
        if not passages:
            return np.zeros(0, dtype=np.float32)
        from torch.nn import Sigmoid

        scores = self._model_or_load().predict(
            [(query, passage) for passage in passages],
            batch_size=self.batch_size,
            activation_fn=Sigmoid(),
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return np.asarray(scores, dtype=np.float32)

    def _model_or_load(self):
        self.load()
        return self._model
