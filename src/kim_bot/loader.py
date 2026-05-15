import os
import logging
from pathlib import Path
from chromadb import Collection, PersistentClient

from kim_bot.config import config, db_path
from kim_bot.util import is_supported_file_type


logger = logging.getLogger(__name__)


db_path.mkdir(parents=True, exist_ok=True)
client = PersistentClient(path=str(db_path))


def get_embed_context_length(model: str) -> int:
    import ollama

    info = ollama.show(model)
    for key, value in info.get("model_info", {}).items():
        if key.endswith(".context_length"):
            return int(value)
    return 512


def embed_and_store_document_chunks(
    path: Path, collection: Collection, converter: any, chunker: any
) -> None:
    import ollama

    doc = converter.convert(path).document

    # Possible that Docling does not extract any chunks from a given document
    chunk_iter = chunker.chunk(dl_doc=doc)

    batch_ids = []
    batch_text = []

    for i, chunk in enumerate(chunk_iter):
        # contextualize() prepends heading/structural context not counted in max_tokens,
        # so hard-truncate to the model's actual context window before embedding.
        enriched_text = chunker.contextualize(chunk=chunk)

        id = f"{path}_{str(i)}"
        batch_ids.append(id)
        batch_text.append(enriched_text)

    if batch_text:
        response = ollama.embed(
            model=config["EMBEDDINGS_MODEL"],
            input=batch_text,
        )
        embeddings = response["embeddings"]

        collection.add(ids=batch_ids, embeddings=embeddings, documents=batch_text)


def load_external_knowledge_dir(collection: Collection, seed_directory: Path) -> None:
    from time import sleep
    from alive_progress import alive_bar
    from docling.document_converter import DocumentConverter
    from docling.chunking import HybridChunker
    from transformers import AutoTokenizer, logging

    logging.set_verbosity_error()

    # Avoid start up costs
    converter = DocumentConverter()
    tokenizer = AutoTokenizer.from_pretrained(config["TOKENIZER"])

    # https://docling-project.github.io/docling/faq/#hybridchunker-triggers-warning-token-indices-sequence-length-is-longer-than-the-specified-maximum-sequence-length-for-this-model
    # "Token indices sequence length is longer than the specified maximum sequence length for this model (520 > 512). Running this sequence through the model will result in indexing errors" -> false alarm
    # Reserve ~25% headroom below the model's actual context window.
    context_length = get_embed_context_length(config["EMBEDDINGS_MODEL"])
    chunker = HybridChunker(
        tokenizer=tokenizer, merge_peers=True, max_tokens=int(context_length * 0.75)
    )

    with alive_bar(dual_line=True) as bar:
        for root, dirs, files in os.walk(seed_directory):
            for i, file in enumerate(files):
                abs_path = Path(os.path.abspath(os.path.join(root, file)))
                if is_supported_file_type(abs_path):
                    bar.text = f"Processing: {file}"
                    try:
                        embed_and_store_document_chunks(
                            path=abs_path,
                            collection=collection,
                            converter=converter,
                            chunker=chunker,
                        )
                        bar()
                    except Exception as e:
                        logger.error(
                            f"Unable to process {abs_path} due to {type(e)}: {e}."
                        )

                # Help pace Ollama server
                if i % 30 == 0:
                    sleep(0.5)

            dirs[:] = [
                d
                for d in dirs
                if not (
                    d.startswith((".", "~", "__", "_"))
                    or "db" in d.lower()
                    or "ghidra" in d.lower()
                    or "log" in d.lower()
                    or "test" in d.lower()
                    or "env" in d.lower()
                )
            ]


def init_collection(seed: str) -> None:
    for collection in client.list_collections():
        if collection.name == "docs":
            client.delete_collection(name="docs")
            break

    collection = client.get_or_create_collection(name="docs")
    load_external_knowledge_dir(collection=collection, seed_directory=Path(seed))


def get_collection() -> Collection:
    return client.get_or_create_collection(name="docs")
