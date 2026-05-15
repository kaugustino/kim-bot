import os
import logging
from pathlib import Path
from chromadb import Collection, PersistentClient

from kim_bot.config import config, db_path
from kim_bot.util import is_supported_file_type


logger = logging.getLogger(__name__)


db_path.mkdir(parents=True, exist_ok=True)
client = PersistentClient(path=str(db_path))


def embed_and_store_document_chunks(path: Path, collection: Collection) -> None:
    import ollama
    from docling.chunking import HybridChunker
    from docling.document_converter import DocumentConverter
    from transformers import AutoTokenizer, logging

    logging.set_verbosity_error()

    converter = DocumentConverter()
    doc = converter.convert(path).document

    tokenizer = AutoTokenizer.from_pretrained(config["TOKENIZER"])

    # https://docling-project.github.io/docling/faq/#hybridchunker-triggers-warning-token-indices-sequence-length-is-longer-than-the-specified-maximum-sequence-length-for-this-model
    # "Token indices sequence length is longer than the specified maximum sequence length for this model (520 > 512). Running this sequence through the model will result in indexing errors"
    # False alarm
    chunker = HybridChunker(tokenizer=tokenizer, merge_peers=True)
    chunk_iter = chunker.chunk(dl_doc=doc)

    for i, chunk in enumerate(chunk_iter):
        enriched_text = chunker.contextualize(chunk=chunk)

        response = ollama.embed(
            model=config["EMBEDDINGS_MODEL"],
            input=enriched_text,
        )
        embeddings = response["embeddings"]

        collection.add(
            ids=[f"{path}_{str(i)}"], embeddings=embeddings, documents=[enriched_text]
        )


def load_external_knowledge_dir(collection: Collection, seed_directory: Path) -> None:
    from alive_progress import alive_bar

    with alive_bar(dual_line=True) as bar:
        for root, dirs, files in os.walk(seed_directory):
            for file in files:
                abs_path = Path(os.path.abspath(os.path.join(root, file)))
                if is_supported_file_type(abs_path):
                    bar.text = f"Processing: {file}"
                    try:
                        embed_and_store_document_chunks(
                            path=abs_path, collection=collection
                        )
                        bar()
                    except Exception as e:
                        logger.error(
                            f"Unable to process {abs_path} due to the following:"
                        )
                        logger.error(e)
                        logger.error("Moving on...")

            dirs[:] = [
                d
                for d in dirs
                if not (
                    d.startswith((".", "~", "__", "_"))
                    or "db" in d.lower()
                    or "ghidra" in d.lower()
                    or "log" in d.lower()
                )
            ]


def init_collection(seed: str) -> None:
    for collection in client.list_collections():
        if collection == "docs":
            client.delete_collection(name="docs")
            break

    collection = client.get_or_create_collection(name="docs")
    load_external_knowledge_dir(collection=collection, seed_directory=Path(seed))


def get_collection() -> Collection:
    return client.get_or_create_collection(name="docs")
