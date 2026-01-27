import pylibmagic  # needed for libmagic library
import magic
from pathlib import Path


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


def get_file_count(path: Path) -> int:
    count = 0
    for path in path.rglob("*"):
        abs_path = path.absolute()
        if is_supported_file_type(abs_path):
            count += 1
    return count
