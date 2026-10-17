"""Load documents from data/knowledge_base, split them, and save chunks as JSONL."""
import json
from pathlib import Path
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader

BASE_DIR = Path(__file__).resolve().parent
KB_DIR = BASE_DIR / "data" / "knowledge_base"
OUTPUT_FILE = BASE_DIR / "data" / "processed" / "chunks.jsonl"

SUPPORTED_TEXT_EXTENSIONS = {".txt", ".md"}


def load_documents() -> list[Document]:
    documents = []

    for path in sorted(KB_DIR.rglob("*")):
        if not path.is_file():
            continue

        suffix = path.suffix.lower()
        if suffix == ".pdf":
            loaded = PyPDFLoader(str(path)).load()
            for doc in loaded:
                doc.metadata["source"] = str(path.relative_to(BASE_DIR))
                doc.metadata["title"] = path.stem
            documents.extend(loaded)
        elif suffix in SUPPORTED_TEXT_EXTENSIONS:
            text = path.read_text(encoding="utf-8", errors="replace").strip()
            if text:
                documents.append(
                    Document(
                        page_content=text,
                        metadata={
                            "source": str(path.relative_to(BASE_DIR)),
                            "title": path.stem,
                            "file_type": suffix.lstrip("."),
                        },
                    )
                )

    return documents


def main() -> None:
    KB_DIR.mkdir(parents=True, exist_ok=True)
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)

    documents = load_documents()
    if not documents:
        raise SystemExit(
            f"No supported documents found in {KB_DIR}. Add PDF, TXT, or Markdown files."
        )

    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=150,
        add_start_index=True,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    chunks = splitter.split_documents(documents)

    with OUTPUT_FILE.open("w", encoding="utf-8") as file:
        for chunk_id, chunk in enumerate(chunks, start=1):
            record = {
                "chunk_id": f"CH{chunk_id:04d}",
                "text": chunk.page_content,
                "metadata": chunk.metadata,
            }
            file.write(json.dumps(record, ensure_ascii=False) + "\n")

    print(f"Loaded documents: {len(documents)}")
    print(f"Created chunks: {len(chunks)}")
    print(f"Saved chunks to: {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
