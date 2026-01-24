import os
import pylibmagic  # needed for libmagic library
import magic
from dotenv import load_dotenv
from pathlib import Path

import ollama
import chromadb
from docling.document_converter import DocumentConverter
from docling.chunking import HybridChunker

load_dotenv()

client = chromadb.PersistentClient(path="./chroma_db")
collection = client.create_collection(name="docs")


def is_supported_file_type(path: Path) -> Path:
    # Parse Markdown and code files, PDFs, and images
    # Ignore other formats, e.g. docx, csv, videos, for now...
    if path.is_file():
        file_type = magic.from_file(path, mime=True)

        if (
            file_type == "application/pdf"
            or file_type.startswith("image/")
            or file_type.startswith("text/")
        ):
            return True

    return False


def embed_and_store_document_chunks(path: Path):
    converter = DocumentConverter()
    doc = converter.convert(path).document

    chunker = HybridChunker()
    chunk_iter = chunker.chunk(dl_doc=doc)

    for i, chunk in enumerate(chunk_iter):
        enriched_text = chunker.contextualize(chunk=chunk)

        response = ollama.embed(
            model="mxbai-embed-large",
            input=enriched_text,
        )
        embeddings = response["embeddings"]

        collection.add(
            ids=[f"{path}_{str(i)}"], embeddings=embeddings, documents=[enriched_text]
        )


def load_external_knowledge_dir() -> None:
    source_dir = os.environ.get("EXTERNAL_KNOWLEDGE_DIRECTORY")

    for path in Path(source_dir).rglob("*"):
        abs_path = path.absolute()
        if is_supported_file_type(abs_path):
            embed_and_store_document_chunks(abs_path)
