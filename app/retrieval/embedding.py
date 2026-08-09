"""
Embedding for the NTSB retrieval index.

The encoder is injectable. Tests run against a deterministic fake so the suite
never needs a model download or a network call, and the real model is loaded
lazily so importing this module stays cheap.

Vectors are L2-normalised at write time. sqlite-vec's default distance is L2,
and on unit vectors L2 ordering and cosine ordering agree, so normalising once
on ingest means every query is a cosine ranking without the extra term.
"""

from __future__ import annotations

import os
from dataclasses import dataclass
from typing import Iterable, Protocol, Sequence

#: bge-small-en-v1.5: 384 dims, ~130MB, strong retrieval quality on CPU.
#: Its v1.5 training expects an instruction prefix on queries but not on
#: documents - see QUERY_PREFIX. Override with TURBULENCE_EMBED_MODEL.
DEFAULT_MODEL = os.environ.get(
    "TURBULENCE_EMBED_MODEL", "BAAI/bge-small-en-v1.5"
)
EMBEDDING_DIM = int(os.environ.get("TURBULENCE_EMBED_DIM", "384"))

#: Applied to search queries only. Documents are embedded bare. Getting this
#: backwards degrades recall quietly, so both sides live here together.
QUERY_PREFIX = "Represent this sentence for searching relevant passages: "

DEFAULT_BATCH_SIZE = 16


class Encoder(Protocol):
    """Anything that turns text into vectors."""

    def encode(self, texts: Sequence[str]) -> list[list[float]]: ...

    @property
    def name(self) -> str: ...

    @property
    def dim(self) -> int: ...


@dataclass
class SentenceTransformerEncoder:
    """Real encoder. Model loads on first use, not on import."""

    model_name: str = DEFAULT_MODEL
    batch_size: int = DEFAULT_BATCH_SIZE
    _model: object | None = None

    @property
    def name(self) -> str:
        return self.model_name

    @property
    def dim(self) -> int:
        # sentence-transformers renamed this; support both spellings so the
        # pipeline is not pinned to one version of the library.
        model = self._load()
        getter = getattr(model, "get_embedding_dimension", None)
        if getter is None:
            getter = model.get_sentence_embedding_dimension
        return getter()

    def _load(self):
        if self._model is None:
            from sentence_transformers import SentenceTransformer
            self._model = SentenceTransformer(self.model_name, device="cpu")
        return self._model

    def encode(self, texts: Sequence[str]) -> list[list[float]]:
        model = self._load()
        vectors = model.encode(
            list(texts),
            batch_size=self.batch_size,
            normalize_embeddings=True,
            show_progress_bar=False,
            convert_to_numpy=True,
        )
        return [v.tolist() for v in vectors]


def normalize(vec: Sequence[float]) -> list[float]:
    total = sum(x * x for x in vec) ** 0.5
    if total == 0:
        return list(vec)
    return [x / total for x in vec]


def embed_query(encoder: Encoder, query: str) -> list[float]:
    """Encode a search query, with the instruction prefix the model expects."""
    prefix = QUERY_PREFIX if "bge" in encoder.name.lower() else ""
    return encoder.encode([f"{prefix}{query}"])[0]


def batched(items: Iterable, size: int):
    batch = []
    for item in items:
        batch.append(item)
        if len(batch) >= size:
            yield batch
            batch = []
    if batch:
        yield batch


def serialize(vec: Sequence[float]) -> bytes:
    """Pack a vector for sqlite-vec storage."""
    import struct
    return struct.pack(f"{len(vec)}f", *vec)
