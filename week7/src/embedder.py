"""Граница fastembed: локальные модели ONNX, только CPU."""

from pathlib import Path

import numpy as np
from fastembed import TextEmbedding
from fastembed.common.model_description import ModelSource, PoolingType
from fastembed.rerank.cross_encoder import TextCrossEncoder


class ModelMissing(RuntimeError):
    pass


class Embedder:
    def __init__(self, model_name: str, cache_dir: Path):
        self.model_name, self.cache_dir = model_name, cache_dir
        self.model = None
        self.tokenizer = None

    def load(self, allow_download: bool = False):
        if self.model is not None:
            return self
        if self.model_name == "intfloat/multilingual-e5-base" and not any(
            model["model"] == self.model_name for model in TextEmbedding.list_supported_models()
        ):
            TextEmbedding.add_custom_model(
                model=self.model_name, pooling=PoolingType.MEAN, normalization=True,
                sources=ModelSource(hf=self.model_name), dim=768,
                model_file="onnx/model.onnx", license="MIT", size_in_gb=1.1,
                description="Многоязычный e5-base, среднее по токенам",
            )
        try:
            self.model = TextEmbedding(
                model_name=self.model_name, cache_dir=str(self.cache_dir),
                providers=["CPUExecutionProvider"], local_files_only=not allow_download,
            )
        except ValueError as exc:
            if not allow_download:
                raise ModelMissing("модели нет в кэше — запустите ./week7/run.sh index") from exc
            raise
        # Копия токенизатора без обрезки: иначе отчёт всегда видел бы максимум 512.
        original = self.model.model.tokenizer
        self.tokenizer = type(original).from_str(original.to_str())
        self.tokenizer.no_truncation()
        self.tokenizer.no_padding()
        return self

    def embed_passages(self, texts: list[str]) -> np.ndarray:
        self.load()
        vectors = np.asarray(list(self.model.embed(["passage: " + text for text in texts])), dtype=np.float32)
        return vectors / np.maximum(np.linalg.norm(vectors, axis=1, keepdims=True), 1e-12)

    def embed_query(self, text: str) -> np.ndarray:
        self.load()
        vector = np.asarray(next(iter(self.model.embed(["query: " + text]))), dtype=np.float32)
        return vector / max(float(np.linalg.norm(vector)), 1e-12)

    def count_tokens(self, text: str) -> int:
        self.load()
        return len(self.tokenizer.encode("passage: " + text).ids)


class Reranker:
    def __init__(self, model_name: str, cache_dir: Path):
        self.model_name, self.cache_dir = model_name, cache_dir
        self.model = None

    def load(self, allow_download: bool = False):
        if self.model is None:
            try:
                self.model = TextCrossEncoder(
                    model_name=self.model_name, cache_dir=str(self.cache_dir),
                    providers=["CPUExecutionProvider"], local_files_only=not allow_download,
                )
            except ValueError as exc:
                if not allow_download:
                    raise ModelMissing("модели нет в кэше — запустите ./week7/run.sh index") from exc
                raise
        return self

    def score(self, query: str, texts: list[str]) -> list[float]:
        self.load()
        return [float(score) for score in self.model.rerank(query, texts)]
