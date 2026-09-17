import os
import tomllib
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
PYPROJECT_PATH = BASE_DIR / "pyproject.toml"


def get_app_version() -> str:
    try:
        with open(PYPROJECT_PATH, "rb") as f:
            data = tomllib.load(f)
            return data.get("project", {}).get("version", "0.1.0")
    except (OSError, tomllib.TOMLDecodeError):
        return "0.1.0"


DATABASE_URL = os.getenv(
    "DATABASE_URL", "postgresql://postgres:postgres@localhost:5432/mlpops"
)
