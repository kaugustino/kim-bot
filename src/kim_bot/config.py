from pathlib import Path
from configparser import ConfigParser
from platformdirs import PlatformDirs


dirs = PlatformDirs("kim-bot")

config_file = Path(dirs.user_config_dir) / "config.ini"
db_path = Path(dirs.user_data_dir) / "chroma_db"


DEFAULT = {
    "specs": {
        "EMBEDDINGS_MODEL": "mxbai-embed-large",
        "TOKENIZER": "mixedbread-ai/mxbai-embed-large-v1",
        "MODEL": "mistral",
    }
}


def get_config() -> dict:
    if config_file.exists():
        try:
            config = ConfigParser()
            # Read config keys how they are
            config.optionxform = str
            config.read(config_file)

            return config["specs"]
        except Exception:
            print(f"Something is wrong with {config_file}...")
            print("Using default values...")

    return DEFAULT["specs"]


config = get_config()
