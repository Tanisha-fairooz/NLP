

## Project Overview

This project compares **BERT** (general-domain pretraining) and **SciBERT** (pretrained on scientific papers) to test whether domain-specific pretraining improves performance on scientific/biomedical NLP tasks.

## Method

**Datasets:**
- **SciERC** — sentence-level relation classification on CS research abstracts (e.g. `USED-FOR`, `PART-OF`)
- **BC5CDR** — token-level NER for chemical mentions (`O` / `B-Chemical`)

Both datasets are subsampled to 100 examples per split for fast iteration.

**Models:** `bert-base-uncased` vs `allenai/scibert_scivocab_uncased`, each fine-tuned with a fresh classification head (`AutoModelForSequenceClassification` for SciERC, `AutoModelForTokenClassification` for BC5CDR).

**Training:** Identical hyperparameters for both models per task — 2 epochs, batch size 8, learning rate 2e-5 — so pretraining corpus is the only variable that differs.

**Metrics:** Accuracy/Precision/Recall/F1 (weighted) for relation classification via `sklearn`; span-level Precision/Recall/F1/Accuracy for NER via `seqeval`.

##  Results

### 🔹 Relation Classification (SciERC)
- **BERT**: Accuracy = 58%, F1 = 0.42  
- **SciBERT**: Accuracy = 64%, F1 = 0.49  

### 🔹 Named Entity Recognition (BC5CDR)
- **BERT**: Accuracy = 93.4%, F1 = 0.69  
- **SciBERT**: Accuracy = 96.2%, F1 = 0.81  

| Task | Model   | Accuracy | Precision | Recall | F1-score |
|------|---------|----------|-----------|--------|----------|
| RC   | BERT    | 0.58     | 0.34      | 0.58   | 0.43     |
| RC   | SciBERT | 0.64     | 0.41      | 0.64   | 0.50     |
| NER  | BERT    | 0.934    | 0.60      | 0.81   | 0.69     |
| NER  | SciBERT | 0.963    | 0.76      | 0.87   | 0.81     |

SciBERT > BERT for both RC and NER tasks

## Tech Stack

**Language & Environment**
- Python 3
- Google Colab (GPU runtime)

**ML/NLP Libraries**
- `transformers` (Hugging Face) — model loading, tokenization, `Trainer` API
- `datasets` (Hugging Face) — dataset loading and processing
- `evaluate` + `seqeval` — NER evaluation metrics
- `scikit-learn` — classification metrics (accuracy, F1, precision, recall)
- `accelerate` — backend for `Trainer` on GPU
- PyTorch (`torch`) — underlying deep learning framework

**Models**
- `bert-base-uncased`
- `allenai/scibert_scivocab_uncased`

**Data Sources**
- SciERC dataset — via Hugging Face Hub (`nsusemiehl/SciERC`)
- BC5CDR dataset — loaded from Google Drive (JSON files)

