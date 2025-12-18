# NLP Text Classification

This project studies text classification using two different approaches:
a classical machine learning baseline and a transformer-based model.
The goal is to compare their performance, strengths, and limitations
on the same dataset.

## Problem Statement

Given a text document, the task is to predict its category.
This project focuses on multi-class text classification.

The main objective is to find a contiguous and reproducible pipeline
that goes from raw text data to trained models and evaluated results.

## Dataset

The project uses a publicly available text classification dataset
(e.g. AG News).

Each sample consists of:
- a text (news article)
- a label (category)

The dataset is split into training, validation, and test sets.

Raw data is not stored in the repository. Instead, it is loaded using
standard dataset utilities.

## Approach

Two different approaches are implemented and compared.

### 1. Baseline: TF-IDF + Logistic Regression

This approach represents text using TF-IDF features and applies a
Logistic Regression classifier.

Pipeline:
- text preprocessing
- TF-IDF vectorization
- Logistic Regression training
- evaluation using standard classification metrics

This model serves as a strong and interpretable baseline.

### 2. Transformer-based Model: DistilBERT

A pre-trained DistilBERT model is fine-tuned on the same dataset.

Pipeline:
- tokenization using a pre-trained tokenizer
- fine-tuning DistilBERT for classification
- evaluation on validation and test sets

This approach is expected to capture deeper semantic information
from the text.

## Evaluation

Models are evaluated using:
- Accuracy
- F1-score (macro)
- Confusion matrix

Results from both approaches are compared to highlight
the trade-offs between classical and transformer-based methods.

## Project Structure

```text
nlp-text-classification/
├── src/
│   ├── train_tfidf_lr.py     # baseline training
│   ├── train_bert.py         # transformer fine-tuning
│   ├── evaluate.py           # metrics and evaluation
│   └── utils.py              # shared utilities
├── notebooks/
│   └── EDA.ipynb             # exploratory data analysis
├── reports/
│   └── figures/              # plots and figures
├── requirements.txt
└── README.md
```

## How to Run

1. Install dependencies:
```bash
pip install -r requirements.txt
```
2.	Train the baseline model:
```bash
python src/train_tfidf_lr.py
```
3. Train the transformer model:
```bash
python src/train_bert.py
```
4. 	Evaluate results:
```bash
python src/evaluate.py
```
