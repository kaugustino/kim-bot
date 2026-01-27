import ollama
import chromadb
from chromadb import Collection
from docling.chunking import HybridChunker
from docling.document_converter import DocumentConverter
from transformers import AutoTokenizer, logging

from pathlib import Path
from alive_progress import alive_bar

from config import SOURCE_DIR, EMBEDDINGS_MODEL, TOKENIZER
from util import is_supported_file_type, get_file_count

logging.set_verbosity_error()


client = chromadb.PersistentClient(path="./chroma_db")


def embed_and_store_document_chunks(path: Path, collection: Collection) -> None:
    converter = DocumentConverter()
    doc = converter.convert(path).document

    tokenizer = AutoTokenizer.from_pretrained(TOKENIZER)

    # https://docling-project.github.io/docling/faq/#hybridchunker-triggers-warning-token-indices-sequence-length-is-longer-than-the-specified-maximum-sequence-length-for-this-model
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
    chunk_iter = chunker.chunk(dl_doc=doc)

    for i, chunk in enumerate(chunk_iter):
        enriched_text = chunker.contextualize(chunk=chunk)

        response = ollama.embed(
            model=EMBEDDINGS_MODEL,
            input=enriched_text,
        )
        embeddings = response["embeddings"]

        collection.add(
            ids=[f"{path}_{str(i)}"], embeddings=embeddings, documents=[enriched_text]
        )


def load_external_knowledge_dir(collection: Collection) -> None:
    with alive_bar(get_file_count(Path(SOURCE_DIR))) as bar:
        for path in Path(SOURCE_DIR).rglob("*"):
            abs_path = path.absolute()
            if is_supported_file_type(abs_path):
                embed_and_store_document_chunks(path=abs_path, collection=collection)
            bar()


def init_collection() -> None:
    client.delete_collection(name="docs")
    collection = client.create_collection(name="docs")
    load_external_knowledge_dir(collection=collection)


def get_collection() -> Collection:
    return client.get_collection(name="docs")
