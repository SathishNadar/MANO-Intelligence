#!/usr/bin/env python3
# backend/index_knowledge.py
# Step 2: Read chunks.json -> embed each chunk -> store in ChromaDB
#
# Pre-requisites:
#   pip install chromadb sentence-transformers
#
# Usage (run from project root):
#   python backend/index_knowledge.py                          (uses default paths)
#   python backend/index_knowledge.py path/to/chunks.json     (custom chunks file)

import sys
import os
import json

import chromadb
from sentence_transformers import SentenceTransformer

# ─── Config ──────────────────────────────────────────────────────────────────
# Paths are relative to the project root (parent of this file's directory)
_HERE        = os.path.dirname(os.path.abspath(__file__))
_ROOT        = os.path.dirname(_HERE)

CHUNKS_FILE  = os.path.join(_ROOT, "knowledge_base", "chunks.json")
CHROMA_PATH  = os.path.join(_HERE, "chroma_data")
COLLECTION   = "mano_kb"
EMBED_MODEL  = "all-MiniLM-L6-v2"   # free local model, no API key needed


def index_chunks(chunks_file: str = CHUNKS_FILE, chroma_path: str = CHROMA_PATH):

    # ── Sub-step 2c: Load chunks.json ────────────────────────────────────────
    if not os.path.exists(chunks_file):
        print(f"[ERROR] chunks.json not found: {chunks_file}")
        print("        Run scripts/chunk_data.ipynb first.")
        return 0

    with open(chunks_file, "r", encoding="utf-8") as f:
        chunks = json.load(f)

    print(f"[INDEX] Loaded {len(chunks)} chunks from: {chunks_file}")

    # ── Sub-step 2b: Connect to ChromaDB ─────────────────────────────────────
    os.makedirs(chroma_path, exist_ok=True)
    client     = chromadb.PersistentClient(path=chroma_path)
    collection = client.get_or_create_collection(name=COLLECTION)
    print(f"[INDEX] ChromaDB collection '{COLLECTION}' ready at: {chroma_path}")
    print(f"[INDEX] Existing vectors in collection: {collection.count()}")

    # ── Sub-step 2d: Load embedding model ────────────────────────────────────
    print(f"[INDEX] Loading embedding model ({EMBED_MODEL})...")
    embedder = SentenceTransformer(EMBED_MODEL)

    # ── Sub-step 2e: Embed all chunk content ─────────────────────────────────
    texts = [c["content"] for c in chunks]
    print(f"[INDEX] Embedding {len(texts)} chunks (may take ~30-60s first time)...")
    embeddings = embedder.encode(texts, show_progress_bar=True).tolist()

    # ── Sub-step 2f: Upsert into ChromaDB ────────────────────────────────────
    ids = [c["id"] for c in chunks]
    metadatas = [
        {
            "page":        c["page_name"],
            "url":         c["url"],
            "section":     c["section_num"],
            "source_file": c["source_file"],
        }
        for c in chunks
    ]

    collection.upsert(
        ids        = ids,
        embeddings = embeddings,
        documents  = texts,
        metadatas  = metadatas,
    )

    # ── Sub-step 2g: Verify ───────────────────────────────────────────────────
    count = collection.count()
    print(f"\n[SUCCESS] {count} vectors are now stored in ChromaDB.")
    print(f"          Collection: '{COLLECTION}'")
    print(f"          Data path:  {os.path.abspath(chroma_path)}")
    print("\n   Next step -> start the backend:")
    print("   cd backend && npm run dev")
    return count


if __name__ == "__main__":
    # Allow passing a custom chunks.json path as argument
    chunks_file = sys.argv[1] if len(sys.argv) > 1 else CHUNKS_FILE
    index_chunks(chunks_file)
