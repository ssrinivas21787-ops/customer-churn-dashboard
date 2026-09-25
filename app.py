import streamlit as st
import pandas as pd
import joblib

# -----------------------------
# Page setup
# -----------------------------
st.set_page_config(
    page_title="RetainAI",
    page_icon="📊",
    layout="wide"
)

st.title("📊 RetainAI")
st.subheader("AI-Powered Customer Retention Intelligence")

st.write(
    "RetainAI helps businesses identify customers at risk of churn "
    "and prioritize retention efforts using AI."
)

# -----------------------------
# Load model
# -----------------------------
try:
    model = joblib.load("models/churn_model.pkl")
    st.success("✅ Churn model loaded successfully!")

except Exception as e:
    st.error("❌ Could not load the churn model.")
    st.write(e)
    st.stop()

# -----------------------------
# Upload customer data
# -----------------------------
st.header("📁 Customer Data")

uploaded_file = st.file_uploader(
    "Upload a customer CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success(f"✅ {len(df)} customers loaded!")

    st.subheader("Customer Data Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    st.info(
        "Next, RetainAI will calculate churn risk and retention priority "
        "for the uploaded customers."
    )
