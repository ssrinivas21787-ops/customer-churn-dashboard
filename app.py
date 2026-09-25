import streamlit as st
import pandas as pd
import joblib

# -----------------------------------
# Page Configuration
# -----------------------------------
st.set_page_config(
    page_title="RetainAI",
    page_icon="📊",
    layout="wide"
)

st.title("📊 RetainAI")
st.subheader("AI-Powered Customer Retention Intelligence")

st.write(
    "RetainAI helps businesses identify customers at risk of churn "
    "and prioritize retention efforts using machine learning."
)

# -----------------------------------
# Load Model
# -----------------------------------
try:
    model = joblib.load("./models/churn_model.pkl")
    st.success("✅ Churn model loaded successfully!")

except Exception as e:
    st.error("❌ Could not load the churn model.")
    st.write(e)
    st.stop()


# -----------------------------------
# Upload Customer Data
# -----------------------------------
st.header("📁 Customer Data")

uploaded_file = st.file_uploader(
    "Upload a customer CSV file",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.success(f"✅ {len(df)} customers loaded!")

    # -----------------------------------
    # Customer Preview
    # -----------------------------------
    st.subheader("Customer Data Preview")

    st.dataframe(
        df.head(10),
        use_container_width=True
    )

    # -----------------------------------
    # Model Feature Check
    # -----------------------------------
    st.subheader("🤖 RetainAI Analysis")

    try:
        # Check the number of model features
        expected_features = model.n_features_in_

        st.write(
            f"Model expects **{expected_features} features**."
        )

        # Check whether uploaded data already matches model input
        if df.shape[1] == expected_features:

            predictions = model.predict(df)

            probabilities = model.predict_proba(df)

            # Probability of positive/churn class
            churn_probability = probabilities[:, 1] * 100

            result = df.copy()

            result["Churn Probability (%)"] = churn_probability

            # -----------------------------------
            # Risk Level
            # -----------------------------------
            result["Risk Level"] = pd.cut(
                result["Churn Probability (%)"],
                bins=[-1, 30, 70, 100],
                labels=["Low", "Medium", "High"]
            )

            # -----------------------------------
            # Retention Priority
            # -----------------------------------
            result["Retention Priority"] = pd.cut(
                result["Churn Probability (%)"],
                bins=[-1, 30, 70, 85, 100],
                labels=[
                    "Low",
                    "Medium",
                    "High",
                    "Critical"
                ]
            )

            st.success("✅ Churn analysis completed!")

            # -----------------------------------
            # Summary Metrics
            # -----------------------------------
            col1, col2, col3, col4 = st.columns(4)

            with col1:
                st.metric(
                    "Total Customers",
                    len(result)
                )

            with col2:
                high_risk = (
                    result["Risk Level"] == "High"
                ).sum()

                st.metric(
                    "High Risk Customers",
                    high_risk
                )

            with col3:
                average_risk = result[
                    "Churn Probability (%)"
                ].mean()

                st.metric(
                    "Average Churn Risk",
                    f"{average_risk:.1f}%"
                )

            with col4:
                critical = (
                    result["Retention Priority"] == "Critical"
                ).sum()

                st.metric(
                    "Critical Priority",
                    critical
                )

            # -----------------------------------
            # Priority Customers
            # -----------------------------------
            st.subheader(
                "🚨 Customers Requiring Immediate Attention"
            )

            priority_customers = result.sort_values(
                "Churn Probability (%)",
                ascending=False
            )

            st.dataframe(
                priority_customers.head(20),
                use_container_width=True
            )

            # -----------------------------------
            # Download Results
            # -----------------------------------
            csv = result.to_csv(index=False)

            st.download_button(
                label="⬇️ Download Retention Analysis",
                data=csv,
                file_name="retainai_customer_risk_analysis.csv",
                mime="text/csv"
            )

        else:

            st.warning(
                f"""
                ⚠️ The uploaded file contains {df.shape[1]} columns,
                but the model expects {expected_features} features.

                Your uploaded CSV needs to use the same preprocessing
                that was used during model training.
                """
            )

    except Exception as e:

        st.error(
            "❌ Prediction could not be completed."
        )

        st.code(str(e))
