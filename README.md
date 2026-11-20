# Legal Drafting Assistant

A command-line assistant for paralegals. It does two things:

1. **Drafts** simple legal sentences with an n-gram next-word model trained on a small banking and finance clause corpus.
2. **Classifies** any sentence into a clause type (payment, interest, confidentiality, termination, governing\_law, indemnification, liability, default) with a fine-tuned GPT-2 model.

Everything runs on your laptop. No API key and no hosting are needed.

## Setup

You need Python 3.10 and git.

```bash
python3.10 -m venv .venv \&\& source .venv/bin/activate   # Windows: .venv\\Scripts\\activate
pip install -r requirements.txt
cp .env.example .env
python -m app.cli hello
```

## Usage

```bash
# 1. Train the n-gram model (prints held-out top-3 next-word accuracy)
python -m app.cli train-ngram

# 2. Get next-word suggestions and a finished sentence
python -m app.cli draft "The Borrower shall"

# 3. Fine-tune GPT-2 (downloads gpt2 once, about 500 MB; a few minutes on CPU)
python -m app.cli train-classifier

# 4. Classify any sentence
python -m app.cli classify "Either party may terminate on thirty days notice."

# 5. Draft then classify in one step
python -m app.cli assist "Interest shall"
```

## How it works

* `app/core.py`: tokenizer, `NGramModel` (counts for context lengths 0 to n-1, backing off to shorter contexts), stratified split, metrics.
* `app/llm.py`: fine-tunes `GPT2ForSequenceClassification` on `data/clauses.csv` and loads it for prediction.
* `app/storage.py`: settings from `.env`, dataset loading, saving the n-gram model.
* `app/cli.py`: the command-line commands above.

## Evaluation

`train-ngram` reports top-3 next-word accuracy on a held-out 20% split. `train-classifier` reports classification accuracy on the same kind of stratified split. Paste your own numbers here after you run them:

|Model|Metric|Result|
|-|-|-|
|n-gram (n=3)|top-3 next-word accuracy|*32.4*|
|fine-tuned GPT-2|classification accuracy|*31.2*|

## Tests

```bash
pytest
```

The tests do not need torch, so they also run on GitHub Actions.

## Limits

The corpus is small (80 sentences) and written for teaching. Drafts are suggestions for a human to review, not legal advice.

