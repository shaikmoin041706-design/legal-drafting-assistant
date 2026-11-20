"""Fine-tuned GPT-2 clause classifier.

torch and transformers are imported inside functions so the rest of the
project (and the tests) work without them.
"""

from __future__ import annotations

import random
from pathlib import Path

from app.core import LABELS, accuracy, train_test_split
from app.storage import Settings


def classifier_dir(settings: Settings) -> Path:
    return Path(settings.model_dir) / "classifier"


def train_classifier(rows: list[dict], settings: Settings) -> dict:
    """Fine-tune GPT-2 for sequence classification and save it. Returns metrics."""
    import torch
    from transformers import AutoTokenizer, GPT2ForSequenceClassification

    random.seed(settings.seed)
    torch.manual_seed(settings.seed)

    label2id = {label: i for i, label in enumerate(LABELS)}
    id2label = {i: label for label, i in label2id.items()}

    tokenizer = AutoTokenizer.from_pretrained(settings.base_model)
    tokenizer.pad_token = tokenizer.eos_token
    model = GPT2ForSequenceClassification.from_pretrained(
        settings.base_model,
        num_labels=len(LABELS),
        id2label=id2label,
        label2id=label2id,
    )
    model.config.pad_token_id = tokenizer.pad_token_id

    train_rows, test_rows = train_test_split(rows, seed=settings.seed)
    optimizer = torch.optim.AdamW(model.parameters(), lr=settings.learning_rate)

    def batches(data, shuffle):
        data = data[:]
        if shuffle:
            random.shuffle(data)
        for i in range(0, len(data), settings.batch_size):
            chunk = data[i : i + settings.batch_size]
            enc = tokenizer(
                [r["text"] for r in chunk],
                padding=True,
                truncation=True,
                max_length=64,
                return_tensors="pt",
            )
            labels = torch.tensor([label2id[r["label"]] for r in chunk])
            yield enc, labels

    model.train()
    for epoch in range(settings.epochs):
        total = 0.0
        for enc, labels in batches(train_rows, shuffle=True):
            out = model(**enc, labels=labels)
            out.loss.backward()
            optimizer.step()
            optimizer.zero_grad()
            total += out.loss.item()
        print(f"epoch {epoch + 1}/{settings.epochs}  loss {total:.3f}")

    model.eval()
    y_true, y_pred = [], []
    with torch.no_grad():
        for enc, labels in batches(test_rows, shuffle=False):
            preds = model(**enc).logits.argmax(dim=-1)
            y_true += [id2label[int(i)] for i in labels]
            y_pred += [id2label[int(i)] for i in preds]
    acc = accuracy(y_true, y_pred)

    out_dir = classifier_dir(settings)
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    tokenizer.save_pretrained(out_dir)
    return {"accuracy": acc, "test_size": len(test_rows), "train_size": len(train_rows)}


class ClauseClassifier:
    """Loads the saved fine-tuned model and labels sentences."""

    def __init__(self, settings: Settings):
        import torch
        from transformers import AutoTokenizer, GPT2ForSequenceClassification

        path = classifier_dir(settings)
        if not path.exists():
            raise FileNotFoundError(
                f"No classifier at {path}. Run: python -m app.cli train-classifier"
            )
        self._torch = torch
        self.tokenizer = AutoTokenizer.from_pretrained(path)
        self.model = GPT2ForSequenceClassification.from_pretrained(path)
        self.model.eval()

    def classify(self, text: str) -> tuple[str, float]:
        enc = self.tokenizer(text, return_tensors="pt", truncation=True, max_length=64)
        with self._torch.no_grad():
            probs = self.model(**enc).logits.softmax(dim=-1)[0]
        idx = int(probs.argmax())
        return self.model.config.id2label[idx], float(probs[idx])
