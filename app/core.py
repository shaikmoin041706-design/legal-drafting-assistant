"""Core logic: tokenizing, the n-gram next-word model, data splitting, metrics.

Nothing here imports torch, so the tests run fast and CI stays light.
"""

from __future__ import annotations

import random
import re
from collections import Counter, defaultdict

LABELS = [
    "payment",
    "interest",
    "confidentiality",
    "termination",
    "governing_law",
    "indemnification",
    "liability",
    "default",
]

START = "<s>"
END = "</s>"
BACKOFF = 0.4  # score multiplier each time we fall back to a shorter context


def tokenize(text: str) -> list[str]:
    """Lowercase words plus , ; . as separate tokens."""
    return re.findall(r"[a-z0-9']+|[.,;]", text.lower())


def detokenize(tokens: list[str]) -> str:
    out = ""
    for tok in tokens:
        if tok in {".", ",", ";"} or not out:
            out += tok
        else:
            out += " " + tok
    return out


class NGramModel:
    """Next-word predictor with simple backoff to shorter contexts."""

    def __init__(self, n: int = 3):
        if n < 2:
            raise ValueError("n must be at least 2")
        self.n = n
        # counts[context_tuple][next_token] -> count, for context lengths 0..n-1
        self.counts: dict[tuple[str, ...], Counter] = defaultdict(Counter)

    # ---- training -------------------------------------------------------
    def fit(self, sentences: list[str]) -> "NGramModel":
        for sentence in sentences:
            tokens = [START] * (self.n - 1) + tokenize(sentence) + [END]
            for i in range(self.n - 1, len(tokens)):
                nxt = tokens[i]
                for length in range(0, self.n):
                    ctx = tuple(tokens[i - length : i])
                    self.counts[ctx][nxt] += 1
        return self

    # ---- prediction -----------------------------------------------------
    def _context(self, text: str) -> list[str]:
        toks = [START] * (self.n - 1) + tokenize(text)
        return toks

    def next_words(self, text: str, k: int = 5) -> list[tuple[str, float]]:
        """Top-k (word, score) candidates; scores sum to 1 across all candidates."""
        toks = self._context(text)
        scores: dict[str, float] = {}
        for length in range(self.n - 1, -1, -1):
            ctx = tuple(toks[len(toks) - length :]) if length else ()
            dist = self.counts.get(ctx)
            if not dist:
                continue
            total = sum(dist.values())
            weight = BACKOFF ** (self.n - 1 - length)
            for word, cnt in dist.items():
                scores[word] = max(scores.get(word, 0.0), weight * cnt / total)
        norm = sum(scores.values())
        if not norm:
            return []
        ranked = sorted(scores.items(), key=lambda kv: (-kv[1], kv[0]))
        return [(w, s / norm) for w, s in ranked[:k]]

    def generate(
        self,
        prompt: str,
        max_words: int = 25,
        greedy: bool = False,
        seed: int | None = None,
    ) -> str:
        """Continue `prompt` one word at a time until END or max_words."""
        rng = random.Random(seed)
        tokens = tokenize(prompt)
        for _ in range(max_words):
            cands = self.next_words(detokenize(tokens), k=5)
            if not cands:
                break
            if greedy:
                word = cands[0][0]
            else:
                words, probs = zip(*cands)
                word = rng.choices(words, weights=probs, k=1)[0]
            if word == END:
                break
            tokens.append(word)
        text = detokenize(tokens)
        if text and text[-1] not in ".;":
            text += "."
        return text[0].upper() + text[1:] if text else text

    # ---- (de)serialization ---------------------------------------------
    def to_dict(self) -> dict:
        return {
            "n": self.n,
            "counts": {
                "\t".join(ctx): dict(counter) for ctx, counter in self.counts.items()
            },
        }

    @classmethod
    def from_dict(cls, data: dict) -> "NGramModel":
        model = cls(data["n"])
        for key, counter in data["counts"].items():
            ctx = tuple(key.split("\t")) if key else ()
            model.counts[ctx] = Counter(counter)
        return model


# ---- data helpers and metrics --------------------------------------------
def train_test_split(
    rows: list[dict], test_frac: float = 0.2, seed: int = 42
) -> tuple[list[dict], list[dict]]:
    """Stratified split so every label appears in both parts."""
    rng = random.Random(seed)
    by_label: dict[str, list[dict]] = defaultdict(list)
    for row in rows:
        by_label[row["label"]].append(row)
    train, test = [], []
    for label in sorted(by_label):
        group = by_label[label][:]
        rng.shuffle(group)
        n_test = max(1, round(len(group) * test_frac))
        test.extend(group[:n_test])
        train.extend(group[n_test:])
    return train, test


def accuracy(y_true: list[str], y_pred: list[str]) -> float:
    if not y_true:
        return 0.0
    return sum(a == b for a, b in zip(y_true, y_pred)) / len(y_true)


def top_k_next_word_accuracy(model: NGramModel, sentences: list[str], k: int = 3) -> float:
    """How often the true next word is in the model's top-k guesses."""
    hits = total = 0
    for sentence in sentences:
        toks = tokenize(sentence)
        for i in range(1, len(toks)):
            guesses = [w for w, _ in model.next_words(detokenize(toks[:i]), k=k)]
            hits += toks[i] in guesses
            total += 1
    return hits / total if total else 0.0
