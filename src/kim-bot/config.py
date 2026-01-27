import os
from dotenv import load_dotenv

load_dotenv()

SOURCE_DIR = os.environ.get("EXTERNAL_KNOWLEDGE_DIRECTORY")
EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "mxbai-embed-large")
TOKENIZER = os.environ.get("TOKENIZER", "mixedbread-ai/mxbai-embed-large-v1")
