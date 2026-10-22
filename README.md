# FinBERT for Financial Tweet Sentiment & Stock-Return Direction Prediction

> Can sentiment in tweets about large companies predict whether their stock goes up or down? This project compares FinBERT-based architectures (plus BERT, RoBERTa-Twitter and Bi-LSTM baselines) for predicting the direction of 1-, 2-, 3- and 7-day returns.

![Python](https://img.shields.io/badge/Python-3.11-3776AB?logo=python&logoColor=white)
![PyTorch](https://img.shields.io/badge/PyTorch-EE4C2C?logo=pytorch&logoColor=white)
![Hugging Face](https://img.shields.io/badge/🤗%20Transformers-FFD21E)
![scikit-learn](https://img.shields.io/badge/scikit--learn-F7931E?logo=scikitlearn&logoColor=white)

Final project for **COSE461 – Natural Language Processing** at Korea University, completed during an exchange semester (Group 17).

## Problem

Classical stock forecasting relies on prices and financial indicators. Social media adds an unstructured but timely signal: investor mood. The question this project asks is:

> *Can sentiment derived from tweets reliably predict stock returns, and which model architecture works best?*

## Data

[Tweet Sentiment's Impact on Stock Returns](https://www.kaggle.com/datasets/thedevastator/tweet-sentiment-s-impact-on-stock-returns) (Kaggle): about 862 k tweets mentioning companies from the *Financial Times Top 100 Global Brands*. Each tweet is linked to the stock's price, volume, 10/30-day volatility, 1/2/3/7-day returns, and LSTM/TextBlob polarity scores.

**Preprocessing**
- Repaired misaligned CSV rows and removed duplicates and nulls.
- Kept the only continuous date range (2018-07-11 → 2018-10-02), which leaves about **281 k tweets**.
- Grouped stocks into top, middle and bottom subsets by tweet volume.
- Created binary direction labels (return > 0) for each horizon. Classes are roughly balanced (≈ 52 / 48).

## Models

The notebook in this repo implements the FinBERT family on [`yiyanghkust/finbert-tone`](https://huggingface.co/yiyanghkust/finbert-tone):

| Model | Architecture |
|---|---|
| **FinBERT (baseline)** | `[CLS]` embedding (768-d) → linear head |
| **Hybrid FinBERT** | Frozen FinBERT `[CLS]` + numeric branch (polarity, volume, 30-day volatility → 64-d MLP) → linear head on 832-d |
| **FinBERT + LSTM** | Full token sequence → LSTM (128 hidden units) ‖ numeric branch (64-d) → linear head |

Training: AdamW, BCE-with-logits loss, mixed precision (fp16) on GPU, 10 epochs, best checkpoint chosen by validation F1, and a stratified 70 / 15 / 15 split.

The group's report also compares Bi-LSTM (with about 25 engineered sentiment, return, volume, volatility and text features), BERT, Hybrid BERT and RoBERTa-Twitter.

## Results

**FinBERT + LSTM** reached **74.2 % accuracy and 0.74 F1** on the held-out test set for 7-day return direction.

In the per-stock analysis, prediction quality varies a lot by company. Some stocks (e.g. P&G, BASF, easyJet) reach F1 > 0.70 on longer horizons, while performance averaged over all stocks stays close to chance. Longer horizons (3–7 days) are consistently more predictable than 1-day returns. The full comparison across all architectures and horizons is in the [final report](COSE461_Project_Final_Report_Template__2023_.pdf).

## Repository contents

| File | Description |
|---|---|
| `FinBERT.ipynb` | Data download & cleaning, FinBERT / Hybrid / FinBERT+LSTM training and evaluation, per-stock analysis |
| `COSE461_Project_Final_Report_Template__2023_.pdf` | Final report (all models, methodology, results) |

## How to run

1. Open `FinBERT.ipynb` in Google Colab with a GPU runtime.
2. Add your own Kaggle API token (`kaggle.json`) to download the dataset.
3. Run all cells.

## Team

Gwendelyn Wong Rhien Yhung · Muhammad Hafizuddin Asyhraf Bin Mazlan · Adina Meiremkhankyzy · **Laura Nicole Bermúdez Santa** (FinBERT models)
