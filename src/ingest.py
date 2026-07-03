"""
Offline script: reads docs/, splits into chunks, computes embeddings, saves index.
Run once before starting the bot: python src/ingest.py
"""

import pickle
import sys
from pathlib import Path
from typing import NamedTuple

import tiktoken
from openai import OpenAI

sys.path.insert(0, str(Path(__file__).parent))
from config import (
    DOCS_DIR,
    DATA_DIR,
    INDEX_PATH,
    EMBEDDING_MODEL,
    CHUNK_SIZE,
    CHUNK_OVERLAP,
    OPENAI_API_KEY,
)

client = OpenAI(api_key=OPENAI_API_KEY)
tokenizer = tiktoken.encoding_for_model("text-embedding-3-small")


class Chunk(NamedTuple):
    text: str
    source: str   # имя файла-источника


def load_documents() -> list[tuple[str, str]]:
    """Returns list of (text, filename) for each .md file in DOCS_DIR."""
    docs = []
    for path in sorted(DOCS_DIR.glob("*.md")):
        docs.append((path.read_text(encoding="utf-8"), path.name))
        print(f"  Loaded: {path.name}")
    return docs


def split_into_chunks(text: str, source: str) -> list[Chunk]:
    """Splits text into overlapping token-based chunks."""
    tokens = tokenizer.encode(text)
    chunks = []
    start = 0
    while start < len(tokens):
        end = min(start + CHUNK_SIZE, len(tokens))
        chunk_tokens = tokens[start:end]
        chunk_text = tokenizer.decode(chunk_tokens)
        chunks.append(Chunk(text=chunk_text, source=source))
        if end == len(tokens):
            break
        start += CHUNK_SIZE - CHUNK_OVERLAP
    return chunks


def embed_chunks(chunks: list[Chunk]) -> list[list[float]]:
    """Calls OpenAI Embeddings API for all chunks in one batched request."""
    texts = [c.text for c in chunks]
    # OpenAI API принимает до 2048 строк за один запрос
    response = client.embeddings.create(model=EMBEDDING_MODEL, input=texts)
    return [item.embedding for item in response.data]


def build_index(chunks: list[Chunk], embeddings: list[list[float]]) -> dict:
    return {"chunks": chunks, "embeddings": embeddings}


def main() -> None:
    print("=== Flowly RAG — Ingestion Pipeline ===\n")

    print("1. Loading documents...")
    docs = load_documents()
    if not docs:
        print("ERROR: No .md files found in", DOCS_DIR)
        sys.exit(1)

    print(f"\n2. Splitting into chunks (size={CHUNK_SIZE}, overlap={CHUNK_OVERLAP})...")
    all_chunks: list[Chunk] = []
    for text, source in docs:
        doc_chunks = split_into_chunks(text, source)
        all_chunks.extend(doc_chunks)
        print(f"  {source}: {len(doc_chunks)} chunks")
    print(f"  Total: {len(all_chunks)} chunks")

    print(f"\n3. Computing embeddings via OpenAI ({EMBEDDING_MODEL})...")
    embeddings = embed_chunks(all_chunks)
    print(f"  Done. Embedding dimension: {len(embeddings[0])}")

    print(f"\n4. Saving index to {INDEX_PATH}...")
    DATA_DIR.mkdir(exist_ok=True)
    index = build_index(all_chunks, embeddings)
    with open(INDEX_PATH, "wb") as f:
        pickle.dump(index, f)

    print("\n=== Ingestion complete ===")
    print(f"Index saved: {len(all_chunks)} chunks from {len(docs)} documents.")


if __name__ == "__main__":
    main()
