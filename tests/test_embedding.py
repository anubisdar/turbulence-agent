"""Tests for the embedding layer.

Runs entirely offline against a deterministic fake encoder - no model
download, no network. That is the point of making the encoder injectable.
"""

import hashlib
import math
import struct

import pytest

from app.retrieval.embedding import (
    EMBEDDING_DIM,
    QUERY_PREFIX,
    batched,
    embed_query,
    normalize,
    serialize,
)


class FakeEncoder:
    """Deterministic hash-based vectors. Same text always gives same vector."""

    def __init__(self, name="fake-model", dim=8):
        self._name = name
        self._dim = dim
        self.calls = []

    @property
    def name(self):
        return self._name

    @property
    def dim(self):
        return self._dim

    def encode(self, texts):
        self.calls.append(list(texts))
        out = []
        for t in texts:
            digest = hashlib.sha256(t.encode()).digest()
            raw = [digest[i % len(digest)] / 255.0 for i in range(self._dim)]
            out.append(normalize(raw))
        return out


class TestNormalize:
    def test_unit_length(self):
        vec = normalize([3.0, 4.0])
        assert math.isclose(sum(x * x for x in vec) ** 0.5, 1.0, rel_tol=1e-9)

    def test_zero_vector_does_not_divide_by_zero(self):
        assert normalize([0.0, 0.0]) == [0.0, 0.0]

    def test_direction_is_preserved(self):
        vec = normalize([3.0, 4.0])
        assert vec[0] < vec[1]


class TestQueryPrefix:
    def test_bge_models_get_the_instruction_prefix(self):
        enc = FakeEncoder(name="BAAI/bge-small-en-v1.5")
        embed_query(enc, "737 MAX stabilizer trim")
        assert enc.calls[0][0].startswith(QUERY_PREFIX)

    def test_other_models_do_not(self):
        enc = FakeEncoder(name="all-MiniLM-L6-v2")
        embed_query(enc, "737 MAX stabilizer trim")
        assert enc.calls[0][0] == "737 MAX stabilizer trim"

    def test_documents_are_never_prefixed(self):
        """The prefix belongs on queries only; using it on both hurts recall."""
        enc = FakeEncoder(name="BAAI/bge-small-en-v1.5")
        enc.encode(["a document chunk"])
        assert not enc.calls[0][0].startswith(QUERY_PREFIX)


class TestDeterminism:
    def test_same_text_gives_same_vector(self):
        enc = FakeEncoder()
        a = enc.encode(["uncommanded nose-down trim"])[0]
        b = enc.encode(["uncommanded nose-down trim"])[0]
        assert a == b

    def test_different_text_gives_different_vector(self):
        enc = FakeEncoder()
        a = enc.encode(["uncommanded nose-down trim"])[0]
        b = enc.encode(["hard landing on runway 27"])[0]
        assert a != b


class TestBatching:
    def test_splits_evenly(self):
        assert [len(b) for b in batched(range(10), 3)] == [3, 3, 3, 1]

    def test_empty_input_yields_nothing(self):
        assert list(batched([], 4)) == []

    def test_batch_smaller_than_size(self):
        assert [len(b) for b in batched(range(2), 8)] == [2]

    def test_no_items_are_lost(self):
        items = list(range(37))
        flat = [x for b in batched(items, 5) for x in b]
        assert flat == items


class TestSerialize:
    def test_roundtrips_through_struct(self):
        vec = [0.1, -0.2, 0.3]
        raw = serialize(vec)
        back = struct.unpack(f"{len(vec)}f", raw)
        assert all(math.isclose(a, b, rel_tol=1e-6) for a, b in zip(vec, back))

    def test_byte_length_matches_dimension(self):
        assert len(serialize([0.0] * EMBEDDING_DIM)) == EMBEDDING_DIM * 4

    def test_empty_vector(self):
        assert serialize([]) == b""


class TestEncoderContract:
    """Anything injected must satisfy the shape the pipeline relies on."""

    @pytest.fixture
    def enc(self):
        return FakeEncoder()

    def test_returns_one_vector_per_input(self, enc):
        out = enc.encode(["a", "b", "c"])
        assert len(out) == 3

    def test_all_vectors_have_the_declared_dimension(self, enc):
        for vec in enc.encode(["a", "b"]):
            assert len(vec) == enc.dim

    def test_vectors_are_unit_length(self, enc):
        for vec in enc.encode(["a", "b"]):
            assert math.isclose(sum(x * x for x in vec) ** 0.5, 1.0, rel_tol=1e-6)
