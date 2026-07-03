"""
Basic tests for the retriever module.

These tests use mocked embeddings to avoid hitting the OpenAI API during CI.
They verify that the cosine similarity logic and chunk ranking work correctly.
"""

import pickle
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

# Добавляем src/ в путь импорта
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))


# ── Fixtures ──────────────────────────────────────────────────────────────────

def make_fake_index(tmp_path: Path) -> Path:
    """Creates a minimal pickle index with 3 fake chunks."""
    from ingest import Chunk

    chunks = [
        Chunk(text="Тариф Pro стоит 590 рублей в месяц за участника.", source="pricing.md"),
        Chunk(text="Для сброса пароля нажмите 'Забыли пароль?' на странице входа.", source="troubleshooting.md"),
        Chunk(text="Интеграция с Telegram доступна на тарифах Pro и Business.", source="integrations.md"),
    ]
    # Три фиктивных вектора — будем управлять сходством через их направление
    embeddings = [
        [1.0, 0.0, 0.0],   # pricing chunk  → сходен с [1,0,0]
        [0.0, 1.0, 0.0],   # troubleshooting
        [0.0, 0.0, 1.0],   # integrations
    ]
    index_path = tmp_path / "index.pkl"
    with open(index_path, "wb") as f:
        pickle.dump({"chunks": chunks, "embeddings": embeddings}, f)
    return index_path


# ── Tests ─────────────────────────────────────────────────────────────────────

def test_search_returns_most_relevant_chunk(tmp_path):
    """The chunk whose embedding is closest to the query should rank first."""
    index_path = make_fake_index(tmp_path)

    # query embedding направлен вдоль первой оси → должен найти pricing chunk
    query_embedding = [1.0, 0.0, 0.0]

    with (
        patch("retriever.INDEX_PATH", index_path),
        patch("retriever.client") as mock_client,
    ):
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=query_embedding)]
        mock_client.embeddings.create.return_value = mock_resp

        # Принудительно перезагружаем индекс в модуле retriever
        import importlib
        import retriever
        importlib.reload(retriever)

        results = retriever.search("сколько стоит про тариф", top_k=3)

    assert len(results) == 3
    assert results[0].chunk.source == "pricing.md"
    assert results[0].score == pytest.approx(1.0, abs=1e-5)


def test_search_top_k_limits_results(tmp_path):
    """search() should return at most top_k results."""
    index_path = make_fake_index(tmp_path)
    query_embedding = [0.0, 1.0, 0.0]

    with (
        patch("retriever.INDEX_PATH", index_path),
        patch("retriever.client") as mock_client,
    ):
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=query_embedding)]
        mock_client.embeddings.create.return_value = mock_resp

        import importlib
        import retriever
        importlib.reload(retriever)

        results = retriever.search("как сбросить пароль", top_k=2)

    assert len(results) == 2


def test_search_scores_are_descending(tmp_path):
    """Results must be sorted by score from highest to lowest."""
    index_path = make_fake_index(tmp_path)
    query_embedding = [0.6, 0.8, 0.0]   # не выровнен с одной осью

    with (
        patch("retriever.INDEX_PATH", index_path),
        patch("retriever.client") as mock_client,
    ):
        mock_resp = MagicMock()
        mock_resp.data = [MagicMock(embedding=query_embedding)]
        mock_client.embeddings.create.return_value = mock_resp

        import importlib
        import retriever
        importlib.reload(retriever)

        results = retriever.search("вопрос", top_k=3)

    scores = [r.score for r in results]
    assert scores == sorted(scores, reverse=True)


def test_missing_index_raises_error(tmp_path):
    """If the index file doesn't exist, loading should raise FileNotFoundError."""
    missing_path = tmp_path / "nonexistent.pkl"

    with patch("retriever.INDEX_PATH", missing_path):
        import importlib
        import retriever
        with pytest.raises(FileNotFoundError, match="Run `python src/ingest.py`"):
            importlib.reload(retriever)
