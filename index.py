"""Embed processed chunks with Ollama and save a local FAISS index."""
import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_community.vectorstores import FAISS
from langchain_ollama import OllamaEmbeddings

BASE_DIR = Path(__file__).resolve().parent
CHUNKS_FILE = BASE_DIR / "data" / "processed" / "chunks.jsonl"
INDEX_DIR = BASE_DIR / "data" / "index"
EMBEDDING_MODEL = "nomic-embed-text"


def main() -> None:
    if not CHUNKS_FILE.exists():
        raise SystemExit("Chunks not found. Run: python ingest.py")

    documents = []
    with CHUNKS_FILE.open("r", encoding="utf-8") as file:
        for line in file:
            item = json.loads(line)
            metadata = dict(item["metadata"])
            metadata["chunk_id"] = item["chunk_id"]
            documents.append(Document(page_content=item["text"], metadata=metadata))

    if not documents:
        raise SystemExit("The chunks file is empty. Add documents and rerun ingestion.")

    embeddings = OllamaEmbeddings(model=EMBEDDING_MODEL)
    vector_store = FAISS.from_documents(documents, embeddings)
    INDEX_DIR.mkdir(parents=True, exist_ok=True)
    vector_store.save_local(str(INDEX_DIR))

    print(f"Embedded chunks: {len(documents)}")
    print(f"Embedding model: {EMBEDDING_MODEL}")
    print(f"FAISS index saved to: {INDEX_DIR}")

    # Quick sanity check: retrieve against the first chunk's text.
    results = vector_store.similarity_search(documents[0].page_content[:300], k=1)
    print("Sanity-check retrieval:")
    print(results[0].metadata)


if __name__ == "__main__":
    main()
