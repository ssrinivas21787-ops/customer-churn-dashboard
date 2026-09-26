import streamlit as st
import pandas as pd
import joblib

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

# Load model
try:
    model = joblib.load("./models/churn_model.pkl")
    st.success("✅ Churn model loaded successfully!")
except Exception as e:
    st.error("❌ Could not load the churn model.")
    st.write(e)
    st.stop()

# Upload data
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

    st.subheader("🤖 RetainAI Analysis")

    try:

        expected_features = model.n_features_in_

        st.write(
            f"Model expects **{expected_features} features**."
        )

        # Remove Churn column
        if "Churn" in df.columns:
            X_input = df.drop("Churn", axis=1)
        else:
            X_input = df.copy()

        # Check features
        if X_input.shape[1] == expected_features:

            # Prediction
            probabilities = model.predict_proba(X_input)

            churn_probability = probabilities[:, 1] * 100

            result = df.copy()

            result["Churn Probability (%)"] = churn_probability

            # Risk
            result["Risk Level"] = pd.cut(
                result["Churn Probability (%)"],
                bins=[-1, 30, 70, 100],
                labels=[
                    "Low",
                    "Medium",
                    "High"
                ]
            )

            # Priority
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

            # Recommended action
            def retention_action(priority):

                if priority == "Critical":
                    return "Immediate personal contact + retention offer"

                elif priority == "High":
                    return "Contact customer + personalized offer"

                elif priority == "Medium":
                    return "Send personalized engagement offer"

                else:
                    return "Continue regular engagement"

            result["Recommended Action"] = (
                result["Retention Priority"]
                .astype(str)
                .apply(retention_action)
            )

            st.success(
                "✅ Churn analysis completed successfully!"
            )

            # Dashboard
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

            # Priority customers
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

            # Recommended actions
            st.subheader(
                "💡 Recommended Retention Actions"
            )

            action_view = result[
                [
                    "Churn Probability (%)",
                    "Risk Level",
                    "Retention Priority",
                    "Recommended Action"
                ]
            ].sort_values(
                "Churn Probability (%)",
                ascending=False
            )

            st.dataframe(
                action_view.head(20),
                use_container_width=True
            )

            # Download
            csv = result.to_csv(index=False)

            st.download_button(
                label="⬇️ Download Retention Analysis",
                data=csv,
                file_name="retainai_customer_risk_analysis.csv",
                mime="text/csv"
            )

        else:

            st.warning(
                f"⚠️ Uploaded data has {X_input.shape[1]} "
                f"features, but the model expects "
                f"{expected_features}."
            )

    except Exception as e:

        st.error(
            "❌ Prediction could not be completed."
        )

        st.code(str(e))
