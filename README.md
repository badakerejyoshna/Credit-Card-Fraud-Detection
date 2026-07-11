# Credit Card Fraud Detection

A machine learning project that predicts whether a credit card transaction is genuine
or fraudulent.

## Dataset

[Credit Card Fraud Detection dataset](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud),
made available on Kaggle by the Machine Learning Group at Université Libre de Bruxelles
(ULB). Contains 284,807 European cardholder transactions from September 2013, with 492
labeled as fraud (~0.17%).


## Approach

- **Preprocessing** — scaled `Time` and `Amount` (the only non-PCA features)
- **Class imbalance** — used `class_weight="balanced"` so the model doesn't ignore the
  rare fraud class
- **Models** — Logistic Regression and a Decision Tree, compared side by side
- **Evaluation** — precision, recall, F1, and a confusion matrix, since accuracy alone
  is misleading when fraud is under 1% of the data
- **App** — a small Streamlit interface to test transactions interactively

## Files

- `fraud_detection.ipynb` — full pipeline: EDA, preprocessing, model training,
  evaluation
- `app.py` — Streamlit app for testing individual transactions
- `requirements.txt`

Running the notebook also produces `fraud_model.pkl`, `scaler.pkl`, and
`test_sample.csv`, which the app then loads.

## How to run

```bash
pip install -r requirements.txt
```

1. Download `creditcard.csv` from Kaggle and place it in this folder
2. Run `fraud_detection.ipynb` (Jupyter, VS Code, or Google Colab all work)
3. `streamlit run app.py`

## Possible extensions

- Try Random Forest or XGBoost for potentially stronger performance
- Tune the classification threshold based on the cost of a missed fraud vs. a false
  alarm
- Explore SMOTE or other oversampling techniques for the class imbalance
