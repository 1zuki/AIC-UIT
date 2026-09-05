# IOAI Potato — Semantic Word Guessing

This folder contains a self-contained `solution.ipynb` submission and the
provided local evaluator.

The solution uses only:

- `dataset/vocabulary.json`
- `dataset/public_embeddings.npy`

It does not download models or data at runtime, and its standard output remains
JSON-only for the contest protocol.

## Local setup

From this directory:

```bash
uv venv --python 3.11 .venv
uv pip install --python .venv/bin/python -r requirements.txt
source .venv/bin/activate
```

The environment can be used to open the notebook:

```bash
jupyter lab solution.ipynb
```

## Test the submission

Run the supplied public evaluator:

```bash
python local_test.py solution.ipynb
```

The public score is only an approximation of the private judge score. The
submission deliberately does not read `test_public.json` as part of its
strategy.
