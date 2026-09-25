import streamlit as st
import pickle

st.set_page_config(
    page_title="RetainAI - Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)

st.title("📊 RetainAI")
st.subheader("AI-Powered Customer Retention Intelligence")

st.write(
    "Predict customer churn, understand customer risk, "
    "and identify customers who need attention first."
)

# Load trained model
try:
    with open("models/churn_model.pkl", "rb") as file:
        model = pickle.load(file)

    st.success("✅ Churn model loaded successfully!")

except Exception as e:
    st.error("❌ Model could not be loaded.")
    st.code(str(e))
