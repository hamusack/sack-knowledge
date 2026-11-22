import tomllib
from pathlib import Path


def load_config(path):
    with Path(path).open("rb") as file:
        return tomllib.load(file)
