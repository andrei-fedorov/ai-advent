# TooManyRules — эмбеддинги локальной моделью (день 21, неделя 5).
#
# **Единственное место, где импортируется `sentence_transformers`** — как
# `agent.py` для LLM API и `mcp_client.py` для SDK `mcp`. Сегодня модуль берёт
# программа индексации (`rag_index.py`), дальше — поиск по индексу. Про
# документы, куски и индекс модуль не знает: на входе строки, на выходе
# векторы `float32`, нормированные (косинусная близость = скалярное
# произведение).
#
# Модель — многоязычная (спецификация дня 21, §3): русские и английские куски
# лягут в одно пространство векторов. У семейства E5 префиксы обязательны:
# куски — `passage: `, вопросы — `query: `; без них качество заметно падает, а
# ошибка молчаливая. Поэтому префиксы ставит только этот модуль, вызывающий их
# не пишет. Всё, что длиннее `max_seq_length` токенов, модель молча
# отбрасывает — поэтому `count_tokens()` считает кусок так, как он уйдёт в
# модель: с префиксом и служебными токенами.
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

    def load(self, on_download: Callable[[], None] | None = None) -> None:
        """Загрузка модели; повторный вызов ничего не делает. Модели нет в
        кэше — `on_download()`, затем скачивание."""
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
        except OSError:
            transformers_logging.enable_progress_bar()
            if on_download is not None:
                on_download()
            self._model = SentenceTransformer(self.model_name)
            self.downloaded = True
        finally:
            transformers_logging.enable_progress_bar()

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
