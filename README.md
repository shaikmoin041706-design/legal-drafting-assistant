# Legal Drafting Assistant

A command-line assistant for paralegals. It does two things:

1. **Drafts** simple legal sentences with an n-gram next-word model trained on a small banking and finance clause corpus.
2. **Classifies** any sentence into a clause type (payment, interest, confidentiality, termination, governing\_law, indemnification, liability, default) with a fine-tuned GPT-2 model.

Everything runs on your laptop. No API key and no hosting are needed.

## Setup

You need Python 3.10 and git.

```bash
python3.10 -m venv .venv \\\&\\\& source .venv/bin/activate   # Windows: .venv\\\\Scripts\\\\activate
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

`train-ngram` reports top-3 next-word accuracy on a held-out 20% split. `train-classifier` reports classification accuracy on the same kind of stratified split.

|Model|Metric|Result|
|-|-|-|
|n-gram (n=3)|top-3 next-word accuracy|32.4|
|fine-tuned GPT-2|classification accuracy|31.2|

\## Data and model



\*\*Data:\*\* `data/clauses.csv`, 80 short sentences from banking and finance

agreements, each labeled with one of 8 clause types: payment, interest,

confidentiality, termination, governing\_law, indemnification, liability,

default. <these were given by thilly>



\*\*Models:\*\*

\* The n-gram drafting model is implemented from scratch in `app/core.py`.

\* The classifier is the pretrained `gpt2` model from Hugging Face,

&#x20; fine-tuned with `GPT2ForSequenceClassification` on the 8 labels

&#x20; (`app/llm.py`).



\*\*Weights are not included in this repo.\*\* The fine-tuned classifier in

`artifacts/classifier/` is about 500 MB, over GitHub's 100 MB file limit.

Recreate it with `python -m app.cli train-classifier`, which takes a few

minutes on CPU.

example output:(.venv) C:\\Users\\Shaik\\Downloads\\legal-drafting-assistant\\legal-drafting-assistant>git commit -m "Document data and model, fix README formatting"

\[main 56fc3db] Document data and model, fix README formatting

&#x20;1 file changed, 1 insertion(+), 1 deletion(-)



(.venv) C:\\Users\\Shaik\\Downloads\\legal-drafting-assistant\\legal-drafting-assistant>git pull --rebase origin main

From https://github.com/shaikmoin041706-design/legal-drafting-assistant

&#x20;\* branch            main       -> FETCH\_HEAD

Current branch main is up to date.



(.venv) C:\\Users\\Shaik\\Downloads\\legal-drafting-assistant\\legal-drafting-assistant>git push

Enumerating objects: 9, done.

Counting objects: 100% (9/9), done.

Delta compression using up to 8 threads

Compressing objects: 100% (5/5), done.

Writing objects: 100% (6/6), 637 bytes | 637.00 KiB/s, done.

Total 6 (delta 3), reused 0 (delta 0), pack-reused 0 (from 0)

remote: Resolving deltas: 100% (3/3), completed with 2 local objects.

To https://github.com/shaikmoin041706-design/legal-drafting-assistant.git

&#x20;  4e75a88..56fc3db  main -> main



(.venv) C:\\Users\\Shaik\\Downloads\\legal-drafting-assistant\\legal-drafting-assistant>python -m app.cli train-ngram

Held-out top-3 next-word accuracy: 32.4%

Saved n-gram model to artifacts\\ngram.json



(.venv) C:\\Users\\Shaik\\Downloads\\legal-drafting-assistant\\legal-drafting-assistant>python -m app.cli train-classifier

Loading weights: 100%|████████████████████████████████████████████████████████████████████████| 148/148 \[00:00<00:00, 165.52it/s]

\[transformers] GPT2ForSequenceClassification LOAD REPORT from: gpt2

Key          | Status  |

\-------------+---------+-

score.weight | MISSING |



Notes:

\- MISSING:      those params were newly initialized because missing from the checkpoint. Consider training on your downstream task.

epoch 1/4  loss 25.348

epoch 2/4  loss 16.296

epoch 3/4  loss 14.431

epoch 4/4  loss 12.580

Writing model shards: 100%|████████████████████████████████████████████████████████████████████████| 1/1 \[00:00<00:00,  2.38it/s]

Classification accuracy: 31.2% on 16 held-out sentences



(.venv) C:\\Users\\Shaik\\Downloads\\legal-drafting-assistant\\legal-drafting-assistant>python -m app.cli draft "The Borrower shall"

Next-word suggestions:

&#x20; pay             0.30

&#x20; be              0.15

&#x20; indemnify       0.15

&#x20; repay           0.15

&#x20; not             0.02



The borrower shall indemnify the bank for any loss resulting from third party without prior written consent.



(.venv) C:\\Users\\Shaik\\Downloads\\legal-drafting-assistant\\legal-drafting-assistant>python -m app.cli classify "Either party may terminate on thirty days notice."

Loading weights: 100%|████████████████████████████████████████████████████████████████████████| 149/149 \[00:00<00:00, 327.24it/s]

termination (22%)



(.venv) C:\\Users\\Shaik\\Downloads\\legal-drafting-assistant\\legal-drafting-assistant>python -m app.cli assist "Interest shall"

Loading weights: 100%|████████████████████████████████████████████████████████████████████████| 149/149 \[00:00<00:00, 378.78it/s]

Interest shall accrue on the basis of a year of three hundred the bank by wire transfer.

\-> clause type: interest (92%)



(.venv) C:\\Users\\Shaik\\Downloads\\legal-drafting-assistant\\legal-drafting-assistant>

## Tests

```bash
pytest
```

The tests do not need torch, so they also run on GitHub Actions.

## Limits

The corpus is small (80 sentences) and written for teaching. Drafts are suggestions for a human to review, not legal advice.

