from pathlib import Path

from app.cli import main
from app.core import LABELS, NGramModel
from app.storage import Settings, load_ngram, load_rows, save_ngram

DATA = Path(__file__).resolve().parent.parent / "data" / "clauses.csv"


def test_hello_runs(capsys):
    assert main(["hello"]) == 0
    assert "ready" in capsys.readouterr().out


def test_dataset_has_every_label():
    rows = load_rows(DATA)
    assert {r["label"] for r in rows} == set(LABELS)
    assert len(rows) >= 80


def test_ngram_save_and_load(tmp_path):
    settings = Settings(model_dir=str(tmp_path))
    model = NGramModel(3).fit(["The Borrower shall pay interest."])
    save_ngram(model, settings)
    loaded = load_ngram(settings)
    assert loaded.next_words("The Borrower", k=1)[0][0] == "shall"


def test_draft_without_training_fails_cleanly(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv("MODEL_DIR", str(tmp_path / "nothing"))
    assert main(["draft", "The Borrower"]) == 1
    assert "train-ngram" in capsys.readouterr().err
