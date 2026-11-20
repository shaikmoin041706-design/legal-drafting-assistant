"""Command line interface.

    python -m app.cli hello
    python -m app.cli train-ngram
    python -m app.cli draft "This Agreement shall be"
    python -m app.cli train-classifier
    python -m app.cli classify "Either party may terminate on notice."
    python -m app.cli assist "The Borrower shall"
"""

from __future__ import annotations

import argparse
import sys

from app.core import NGramModel, top_k_next_word_accuracy, train_test_split
from app.storage import Settings, load_ngram, load_rows, save_ngram


def cmd_hello(args, settings: Settings) -> int:
    print("Legal Drafting Assistant is ready.")
    print(f"  n-gram order : {settings.ngram_order}")
    print(f"  base model   : {settings.base_model}")
    print(f"  data file    : {settings.data_path}")
    return 0


def cmd_train_ngram(args, settings: Settings) -> int:
    rows = load_rows(settings.data_path)
    train_rows, test_rows = train_test_split(rows, seed=settings.seed)
    model = NGramModel(settings.ngram_order).fit([r["text"] for r in train_rows])
    acc = top_k_next_word_accuracy(model, [r["text"] for r in test_rows], k=3)
    print(f"Held-out top-3 next-word accuracy: {acc:.1%}")
    # Save a model trained on everything for drafting.
    final = NGramModel(settings.ngram_order).fit([r["text"] for r in rows])
    print(f"Saved n-gram model to {save_ngram(final, settings)}")
    return 0


def cmd_draft(args, settings: Settings) -> int:
    model = load_ngram(settings)
    print("Next-word suggestions:")
    for word, score in model.next_words(args.prompt, k=5):
        print(f"  {word:<15} {score:.2f}")
    print()
    print(model.generate(args.prompt, max_words=args.words, seed=args.seed))
    return 0


def cmd_train_classifier(args, settings: Settings) -> int:
    from app.llm import train_classifier

    metrics = train_classifier(load_rows(settings.data_path), settings)
    print(
        f"Classification accuracy: {metrics['accuracy']:.1%} "
        f"on {metrics['test_size']} held-out sentences"
    )
    return 0


def cmd_classify(args, settings: Settings) -> int:
    from app.llm import ClauseClassifier

    label, conf = ClauseClassifier(settings).classify(args.text)
    print(f"{label} ({conf:.0%})")
    return 0


def cmd_assist(args, settings: Settings) -> int:
    from app.llm import ClauseClassifier

    sentence = load_ngram(settings).generate(args.prompt, max_words=args.words, seed=args.seed)
    label, conf = ClauseClassifier(settings).classify(sentence)
    print(sentence)
    print(f"-> clause type: {label} ({conf:.0%})")
    return 0


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="app.cli", description=__doc__.splitlines()[0])
    sub = p.add_subparsers(dest="command", required=True)

    sub.add_parser("hello", help="check the setup").set_defaults(func=cmd_hello)
    sub.add_parser("train-ngram", help="train and evaluate the n-gram model").set_defaults(
        func=cmd_train_ngram
    )
    sub.add_parser("train-classifier", help="fine-tune GPT-2 and evaluate it").set_defaults(
        func=cmd_train_classifier
    )

    d = sub.add_parser("draft", help="suggest next words and finish a sentence")
    d.add_argument("prompt")
    d.add_argument("--words", type=int, default=25)
    d.add_argument("--seed", type=int, default=None)
    d.set_defaults(func=cmd_draft)

    c = sub.add_parser("classify", help="label a sentence with its clause type")
    c.add_argument("text")
    c.set_defaults(func=cmd_classify)

    a = sub.add_parser("assist", help="draft a sentence, then classify it")
    a.add_argument("prompt")
    a.add_argument("--words", type=int, default=25)
    a.add_argument("--seed", type=int, default=None)
    a.set_defaults(func=cmd_assist)
    return p


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return args.func(args, Settings.from_env())
    except FileNotFoundError as err:
        print(err, file=sys.stderr)
        return 1


if __name__ == "__main__":
    sys.exit(main())
