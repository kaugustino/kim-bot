import os
from dotenv import load_dotenv

load_dotenv()

EMBEDDINGS_MODEL = os.environ.get("EMBEDDINGS_MODEL", "mxbai-embed-large")
TOKENIZER = os.environ.get("TOKENIZER", "mixedbread-ai/mxbai-embed-large-v1")
MODEL = os.environ.get("MODEL", "mistral")
