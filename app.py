import streamlit as st
import joblib

st.set_page_config(
    page_title="RetainAI",
    page_icon="📊",
    layout="wide"
)

st.title("📊 RetainAI")
st.subheader("AI-Powered Customer Retention Intelligence")

st.write(
    "A customer churn intelligence system that helps businesses "
    "identify customers who may leave and prioritize retention efforts."
)

try:
    model = joblib.load("models/churn_model.pkl")
    st.success("✅ Churn model loaded successfully!")

except Exception as e:
    st.error("❌ Could not load the churn model.")
    st.write(e)
