import pylibmagic  # ships magic libraries, e.g. libmagic - the underlying UNIX `file` library
import magic
from pathlib import Path

from kim_bot.config import config, config_file

EXCLUDE_PREFIXES = (".", "~", "__", "_")

EXCLUDE_SUBSTRINGS = [
    "db",
    "ghidra",
    "log",
    "test",
    "env",
    "$",
    "build",
    "antlr",
    "java",
    "org",
    "out",
    "bin",
    "src",
]


def is_a_bad_dir(dir: str) -> bool:
    if dir.startswith(EXCLUDE_PREFIXES):
        return True

    d = dir.lower()

    for substring in EXCLUDE_SUBSTRINGS:
        if substring in d:
            return True

    return False


def is_supported_file_type(path: Path) -> Path:
    # Parse Markdown, PDFs, and images
    # Ignore other formats, e.g. code, docx, csv, videos, for now...
    if path.is_file():
        if path.name.startswith("."):
            return False

        file_type = magic.from_file(path, mime=True)

        if file_type in ["application/pdf", "image/png", "image/jpeg", "text/html"]:
            return True

        if file_type == "text/plain" and path.suffix == ".md":
            return True

    return False


def create_config_file(embeddings_model: str, tokenizer: str, model: str) -> None:
    config_file.parent.mkdir(parents=True, exist_ok=True)

    if not embeddings_model:
        embeddings_model = config["EMBEDDINGS_MODEL"]

    if not tokenizer:
        tokenizer = config["TOKENIZER"]

    if not model:
        model = config["MODEL"]

    with open(config_file, "w") as file:
        file.write("[specs]\n")
        file.write(f"EMBEDDINGS_MODEL={embeddings_model}\n")
        file.write(f"TOKENIZER={tokenizer}\n")
        file.write(f"MODEL={model}\n")

    print(f"Created file at {config_file}.")
