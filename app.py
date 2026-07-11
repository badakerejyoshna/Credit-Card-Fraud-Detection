"""
Credit Card Fraud Detection — Simple App
Run with: streamlit run app.py
Requires fraud_model.pkl, scaler.pkl, and test_sample.csv in the same folder
(produced by running fraud_detection.ipynb on the real dataset).
"""

import numpy as np
import pandas as pd
import streamlit as st
import joblib

st.title("💳 Credit Card Fraud Checker")
st.write("Enter a transaction's details below and the model will predict if it looks genuine or fraudulent.")

model = joblib.load("fraud_model.pkl")
scaler = joblib.load("scaler.pkl")
test_df = pd.read_csv("test_sample.csv")
feature_cols = [c for c in test_df.columns if c != "Class"]

st.subheader("Transaction Details")
amount = st.number_input("Amount ($)", min_value=0.0, value=50.0)
txn_time = st.number_input("Time (seconds since first transaction)", min_value=0.0, value=50000.0)

st.write("The V1-V28 fields below come from the original bank data and are hard to set by hand, "
         "so pick a real example from the test set to try:")
choice = st.selectbox("Load an example transaction", ["Genuine example", "Fraud example"])

if choice == "Genuine example":
    example = test_df[test_df["Class"] == 0].sample(1, random_state=1)
else:
    example = test_df[test_df["Class"] == 1].sample(1, random_state=1)

v_values = example[[f"V{i}" for i in range(1, 29)]].values[0]

if st.button("Check This Transaction"):
    scaled_time_amount = scaler.transform([[txn_time, amount]])[0]
    row = pd.DataFrame([np.concatenate([v_values, scaled_time_amount])],
                        columns=[f"V{i}" for i in range(1, 29)] + ["Time", "Amount"])
    row = row[feature_cols]

    prediction = model.predict(row)[0]
    probability = model.predict_proba(row)[0][1]

    st.subheader("Result")
    if prediction == 1:
        st.error(f"🚨 This looks like FRAUD (probability: {probability:.1%})")
    else:
        st.success(f"✅ This looks genuine (fraud probability: {probability:.1%})")

st.markdown("---")
st.subheader("Try a Few Random Transactions")
st.write("Click below to test the model on 5 random transactions from the test set.")

if st.button("Test 5 Random Transactions"):
    sample = test_df.sample(5)
    results = []
    for _, row in sample.iterrows():
        X_row = row[feature_cols].values.reshape(1, -1)
        pred = model.predict(X_row)[0]
        proba = model.predict_proba(X_row)[0][1]
        results.append({
            "Amount": f"${row['Amount']:.2f}",
            "Prediction": "FRAUD" if pred == 1 else "Genuine",
            "Fraud Probability": f"{proba:.1%}",
            "Actual Answer": "FRAUD" if row["Class"] == 1 else "Genuine",
        })
    st.table(pd.DataFrame(results))
