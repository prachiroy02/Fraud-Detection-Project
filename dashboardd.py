import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

# -----------------------
# Load dataset
# -----------------------
@st.cache_data
def load_data():
    df = pd.read_csv(r"C:\Users\drbdp\Downloads\archive (2)\creditcard.csv")
    return df

df = load_data()

# -----------------------
# Load trained model
# -----------------------
@st.cache_resource
def load_model():
    model = joblib.load(r"C:\Users\drbdp\Downloads\fraud detection\fraud_model.pkl")
    return model

model = load_model()

# -----------------------
# Dashboard Layout
# -----------------------
st.set_page_config(page_title="Fraud Detection Dashboard", layout="wide")
st.title("💳 Fraud Detection Dashboard")

# -----------------------
# KPIs
# -----------------------
total_transactions = len(df)
fraud_cases = df['Class'].sum()

col1, col2 = st.columns(2)
col1.metric("Total Transactions", f"{total_transactions:,}")
col2.metric("Fraud Cases", f"{fraud_cases:,}")

# -----------------------
# Fraud vs Legit Chart
# -----------------------
st.subheader("Fraud vs Legit Transactions")
fig, ax = plt.subplots()
sns.countplot(data=df, x="Class", palette="Blues", ax=ax)
ax.set_xticklabels(["Legit", "Fraud"])
st.pyplot(fig)

# -----------------------
# Transaction Amount Distribution
# -----------------------
st.subheader("Transaction Amount Distribution")
fig2, ax2 = plt.subplots()
sns.histplot(df['Amount'], bins=50, kde=False, ax=ax2, color="blue")
st.pyplot(fig2)

# -----------------------
# Prediction Demo
# -----------------------
st.sidebar.header("🔎 Try a Transaction Prediction")

# For demo, we use a few important features
amount = st.sidebar.number_input("Amount", min_value=0.0, step=0.01)
v1 = st.sidebar.slider("V1", -30.0, 30.0, 0.0)
v2 = st.sidebar.slider("V2", -30.0, 30.0, 0.0)
v3 = st.sidebar.slider("V3", -30.0, 30.0, 0.0)
v4 = st.sidebar.slider("V4", -30.0, 30.0, 0.0)
v5 = st.sidebar.slider("V5", -30.0, 30.0, 0.0)

# Prepare input (must match training features shape)
sample = np.zeros((1, df.shape[1] - 1))  # all features except 'Class'
# Fill only some features for demo, rest remain 0
sample[0, 0] = 0          # Time (kept 0 for demo)
sample[0, 1] = v1
sample[0, 2] = v2
sample[0, 3] = v3
sample[0, 4] = v4
sample[0, 5] = v5
sample[0, -1] = amount    # Amount is last feature

if st.sidebar.button("Predict Fraud"):
    prediction = model.predict(sample)
    probability = model.predict_proba(sample)[0][1] * 100

    st.subheader("🔮 Prediction Result")
    if prediction[0] == 1:
        st.error(f"⚠️ Fraudulent Transaction Detected! (Probability: {probability:.2f}%)")
    else:
        st.success(f"✅ Legitimate Transaction (Fraud Probability: {probability:.2f}%)")
