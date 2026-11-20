"""Settings, dataset loading, and saving/loading the n-gram model."""

from __future__ import annotations

import csv
import json
import os
from dataclasses import dataclass
from pathlib import Path

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:  # python-dotenv is optional for the tests
    pass

from app.core import NGramModel


@dataclass
class Settings:
    base_model: str = "gpt2"
    ngram_order: int = 3
    model_dir: str = "artifacts"
    data_path: str = "data/clauses.csv"
    epochs: int = 4
    batch_size: int = 8
    learning_rate: float = 5e-5
    seed: int = 42

    @classmethod
    def from_env(cls) -> "Settings":
        env = os.environ.get
        return cls(
            base_model=env("BASE_MODEL", cls.base_model),
            ngram_order=int(env("NGRAM_ORDER", cls.ngram_order)),
            model_dir=env("MODEL_DIR", cls.model_dir),
            data_path=env("DATA_PATH", cls.data_path),
            epochs=int(env("EPOCHS", cls.epochs)),
            batch_size=int(env("BATCH_SIZE", cls.batch_size)),
            learning_rate=float(env("LEARNING_RATE", cls.learning_rate)),
            seed=int(env("SEED", cls.seed)),
        )


def load_rows(path: str | Path) -> list[dict]:
    """Read the clause dataset: a CSV with `text` and `label` columns."""
    with open(path, newline="", encoding="utf-8") as f:
        return [
            {"text": r["text"].strip(), "label": r["label"].strip()}
            for r in csv.DictReader(f)
            if r.get("text") and r.get("label")
        ]


def ngram_path(settings: Settings) -> Path:
    return Path(settings.model_dir) / "ngram.json"


def save_ngram(model: NGramModel, settings: Settings) -> Path:
    path = ngram_path(settings)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(model.to_dict()), encoding="utf-8")
    return path


def load_ngram(settings: Settings) -> NGramModel:
    path = ngram_path(settings)
    if not path.exists():
        raise FileNotFoundError(
            f"No n-gram model at {path}. Run: python -m app.cli train-ngram"
        )
    return NGramModel.from_dict(json.loads(path.read_text(encoding="utf-8")))
