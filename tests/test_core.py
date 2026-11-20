from app.core import (
    LABELS,
    NGramModel,
    accuracy,
    detokenize,
    tokenize,
    top_k_next_word_accuracy,
    train_test_split,
)

SENTENCES = [
    "This Agreement shall be governed by the laws of India.",
    "This Agreement shall be governed by the laws of New York.",
    "This Agreement shall terminate on notice.",
]


def test_tokenize_and_detokenize_roundtrip():
    toks = tokenize("The Borrower shall pay, in full.")
    assert toks == ["the", "borrower", "shall", "pay", ",", "in", "full", "."]
    assert detokenize(toks) == "the borrower shall pay, in full."


def test_next_word_prediction():
    model = NGramModel(3).fit(SENTENCES)
    top = [w for w, _ in model.next_words("This Agreement shall be governed by the laws", k=1)]
    assert top == ["of"]


def test_probabilities_sum_to_one():
    model = NGramModel(3).fit(SENTENCES)
    scores = [s for _, s in model.next_words("This Agreement", k=100)]
    assert abs(sum(scores) - 1.0) < 1e-9


def test_generate_is_deterministic_with_seed():
    model = NGramModel(3).fit(SENTENCES)
    a = model.generate("This Agreement", seed=1)
    b = model.generate("This Agreement", seed=1)
    assert a == b and a.endswith(".")


def test_serialization_roundtrip():
    model = NGramModel(3).fit(SENTENCES)
    clone = NGramModel.from_dict(model.to_dict())
    assert clone.next_words("This Agreement", k=3) == model.next_words("This Agreement", k=3)


def test_split_is_stratified():
    rows = [{"text": f"t{i}", "label": lab} for lab in LABELS for i in range(10)]
    train, test = train_test_split(rows, test_frac=0.2, seed=0)
    assert {r["label"] for r in test} == set(LABELS)
    assert len(train) + len(test) == len(rows)


def test_accuracy():
    assert accuracy(["a", "b"], ["a", "c"]) == 0.5


def test_top_k_accuracy_on_training_data_is_high():
    model = NGramModel(3).fit(SENTENCES)
    assert top_k_next_word_accuracy(model, SENTENCES, k=3) > 0.9
