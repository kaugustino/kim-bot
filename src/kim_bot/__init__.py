import logging


logging.basicConfig(
    level=logging.ERROR,
    format="[%(levelname)s] %(asctime)s %(name)s %(message)s",
    datefmt="%Y-%m-%d %H:%M:%S",
    force=True,
)
