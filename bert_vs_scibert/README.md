# BERT vs SciBERT: Relation Classification & NER Comparison

This notebook compares **BERT** (`bert-base-uncased`) and **SciBERT** (`allenai/scibert_scivocab_uncased`) on two scientific/biomedical NLP tasks:

1. **Relation Classification** on the [SciERC](https://huggingface.co/datasets/nsusemiehl/SciERC) dataset
2. **Named Entity Recognition (NER)** on the **BC5CDR** dataset (chemical entity tagging)

For each task, both models are fine-tuned and evaluated so their performance can be directly compared.

## What's in the notebook

| Cells | What it does |
|---|---|
| 1–2 | Install dependencies, check environment versions |
| 3 | Load and sample the SciERC dataset (100 examples per split) |
| 4–6 | Mount Google Drive and load the BC5CDR dataset from JSON |
| 7 | Convert BC5CDR tags into BIO-style labels (`O` / `B-Chemical`) |
| 8–9 | Inspect label sets for both datasets |
| 10 | Map SciERC string labels to integer IDs |
| 11 | Tokenization helper functions |
| 12 | Tokenize both datasets with `bert-base-uncased` |
| 13 | Initialize baseline BERT models |
| 14 | Data collator and metric functions (accuracy / seqeval) |
| 15 | `TrainingArguments` for SciERC and BC5CDR |
| 16 | Initial `Trainer` setup (early version) |
| 17 | **BERT** — SciERC relation classification: train + evaluate |
| 18 | **SciBERT** — SciERC relation classification: train + evaluate |
| 19 | **BERT** — BC5CDR NER: train + evaluate |
| 20 | **SciBERT** — BC5CDR NER: train + evaluate |

Each of the four main experiment cells (17–20) reports **accuracy, precision, recall, and F1**, so you can compare BERT vs SciBERT on both tasks side by side.

## Requirements

- Google Colab (recommended) or a local Jupyter environment with a GPU
- A Google Drive folder containing the BC5CDR data:
  ```
  /content/drive/MyDrive/BC5CDR/train.json
  /content/drive/MyDrive/BC5CDR/test.json
  ```
  (each file expected in JSON-lines format, one example per line, with `tokens` and `tags` fields)
- Internet access to download `bert-base-uncased`, `allenai/scibert_scivocab_uncased`, and the SciERC dataset from the Hugging Face Hub

## Setup

1. Open the notebook in Google Colab.
2. Run the first cell to install/upgrade dependencies:
   ```
   pip install -U "transformers>=4.46" datasets seqeval evaluate scikit-learn accelerate -q
   ```
3. **Restart the runtime** (Runtime → Restart runtime) — required for the upgraded packages to load correctly.
4. Run the version-check cell to confirm your `torch` / `transformers` / `datasets` versions.
5. Run the remaining cells top to bottom. You'll be prompted to authorize Google Drive access when the BC5CDR loading cell runs.

## Notes on compatibility

This notebook has been updated to work with recent versions of `transformers`:
- `Trainer(..., tokenizer=...)` was replaced with `Trainer(..., processing_class=...)`, since `tokenizer` was removed from `Trainer.__init__()` in `transformers` 4.46+.
- Dependencies are upgraded/pinned to avoid a known `torchvision`/`VideoReader` import error that can occur with mismatched `transformers`/`torchvision` versions.

If you hit errors after further library updates, run the version-check cell and compare against:
- `transformers >= 4.46`
- a `torch`/`torchvision` pair from the same release cycle (check [pytorch.org](https://pytorch.org) for compatible pairs)

## Sample size

By default, this notebook trains on a **100-example subset** of each dataset (for fast iteration/demo purposes). For meaningful results, increase the `.select(range(100))` calls to use the full dataset splits, and increase `num_train_epochs` in `TrainingArguments` accordingly.

## Output

Each experiment prints a metrics dictionary, e.g.:
```
SciERC Relation Classification Results (BERT): {'eval_loss': ..., 'eval_accuracy': ..., 'eval_f1': ..., ...}
SciERC Relation Classification Results (SciBERT): {'eval_loss': ..., 'eval_accuracy': ..., 'eval_f1': ..., ...}
```
Compare the BERT vs SciBERT dictionaries for each task to see which model performs better on scientific/biomedical text.
