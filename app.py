import json
import joblib
import pandas as pd
import numpy as np
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

from shap_analysis import get_feature_importance, plot_feature_importance
from customer_segmentation import create_customer_segments, plot_customer_segments, segment_summary

from utils import (
    load_data,
    clean_data,
    churn_risk_level,
    retention_recommendation,
    revenue_loss_estimation
)

DATA_PATH = "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_PATH = "models/best_churn_model.joblib"
MODEL_COMPARISON_PATH = "reports/model_comparison.csv"
CONFUSION_MATRIX_PATH = "reports/confusion_matrix.csv"
CLASSIFICATION_REPORT_PATH = "reports/classification_report.json"
METADATA_PATH = "reports/model_metadata.json"


st.set_page_config(
    page_title="ChurnIQ | Customer Churn Intelligence",
    page_icon="📊",
    layout="wide"
)


st.markdown("""
<style>
    .main {
        background-color: #f6f8fb;
    }

    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }

    .hero {
        background: linear-gradient(135deg, #0f172a, #1e3a8a);
        padding: 35px;
        border-radius: 18px;
        color: white;
        margin-bottom: 25px;
    }

    .hero h1 {
        font-size: 44px;
        margin-bottom: 5px;
    }

    .hero p {
        font-size: 18px;
        color: #dbeafe;
    }

    .metric-card {
        background-color: white;
        padding: 22px;
        border-radius: 16px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.08);
        border-left: 6px solid #2563eb;
    }

    .metric-title {
        color: #64748b;
        font-size: 15px;
        font-weight: 600;
    }

    .metric-value {
        color: #0f172a;
        font-size: 30px;
        font-weight: 800;
    }

    .section-card {
        background-color: white;
        padding: 25px;
        border-radius: 18px;
        box-shadow: 0 6px 18px rgba(15, 23, 42, 0.07);
        margin-bottom: 22px;
    }

    .risk-low {
        padding: 18px;
        border-radius: 14px;
        background-color: #dcfce7;
        color: #166534;
        font-weight: 700;
        font-size: 20px;
    }

    .risk-medium {
        padding: 18px;
        border-radius: 14px;
        background-color: #fef9c3;
        color: #854d0e;
        font-weight: 700;
        font-size: 20px;
    }

    .risk-high {
        padding: 18px;
        border-radius: 14px;
        background-color: #fee2e2;
        color: #991b1b;
        font-weight: 700;
        font-size: 20px;
    }

    .footer {
        text-align: center;
        color: #64748b;
        padding: 30px;
        font-size: 14px;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_data
def get_data():
    raw_df = load_data(DATA_PATH)
    clean_df = clean_data(raw_df)
    return raw_df, clean_df


@st.cache_resource
def get_model():
    return joblib.load(MODEL_PATH)


@st.cache_data
def get_reports():
    model_comparison = pd.read_csv(MODEL_COMPARISON_PATH)
    confusion_matrix = pd.read_csv(CONFUSION_MATRIX_PATH)

    with open(CLASSIFICATION_REPORT_PATH, "r") as f:
        classification_report = json.load(f)

    with open(METADATA_PATH, "r") as f:
        metadata = json.load(f)

    return model_comparison, confusion_matrix, classification_report, metadata


raw_df, df = get_data()
model = get_model()
model_comparison, confusion_matrix, classification_report, metadata = get_reports()


st.sidebar.image("https://cdn-icons-png.flaticon.com/512/3135/3135715.png", width=90)
st.sidebar.title("ChurnIQ")
st.sidebar.caption("Customer Churn Intelligence System")

page = st.sidebar.radio(
    "Navigation",
    [
        "Business Overview",
        "EDA Dashboard",
        "Model Performance",
        "Churn Prediction",
        "Retention Engine",
        "Feature Insights",
        "Explainable AI",
        "Customer Segmentation"
    ]
)

st.sidebar.markdown("---")
st.sidebar.info("Built with Python, Scikit-learn, XGBoost, Streamlit and Business Intelligence.")


st.markdown("""
<div class="hero">
    <h1>ChurnIQ</h1>
    <p>Customer Churn Intelligence and Retention Prediction System</p>
</div>
""", unsafe_allow_html=True)


total_customers = len(df)
churned_customers = int(df["Churn"].sum())
active_customers = total_customers - churned_customers
churn_rate = (churned_customers / total_customers) * 100
estimated_loss = revenue_loss_estimation(df)


col1, col2, col3, col4 = st.columns(4)

with col1:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Total Customers</div>
        <div class="metric-value">{total_customers:,}</div>
    </div>
    """, unsafe_allow_html=True)

with col2:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Churned Customers</div>
        <div class="metric-value">{churned_customers:,}</div>
    </div>
    """, unsafe_allow_html=True)

with col3:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Churn Rate</div>
        <div class="metric-value">{churn_rate:.2f}%</div>
    </div>
    """, unsafe_allow_html=True)

with col4:
    st.markdown(f"""
    <div class="metric-card">
        <div class="metric-title">Monthly Revenue at Risk</div>
        <div class="metric-value">${estimated_loss:,.0f}</div>
    </div>
    """, unsafe_allow_html=True)


st.markdown("<br>", unsafe_allow_html=True)


if page == "Business Overview":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Business Intelligence Overview")

    c1, c2 = st.columns(2)

    with c1:
        churn_count = df["Churn"].map({0: "No Churn", 1: "Churn"}).value_counts().reset_index()
        churn_count.columns = ["Status", "Customers"]

        fig = px.pie(
            churn_count,
            names="Status",
            values="Customers",
            hole=0.45,
            title="Customer Churn Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        contract_churn = df.groupby("Contract")["Churn"].mean().reset_index()
        contract_churn["Churn Rate"] = contract_churn["Churn"] * 100

        fig = px.bar(
            contract_churn,
            x="Contract",
            y="Churn Rate",
            title="Churn Rate by Contract Type",
            text=contract_churn["Churn Rate"].round(2)
        )
        st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Key Business Insights")

    st.write("""
    - Month-to-month contract customers have the highest churn tendency.
    - Customers with higher monthly charges are more likely to churn.
    - Short-tenure customers need stronger onboarding and engagement.
    - Electronic check payment users often show higher churn risk.
    - Long-term contracts and loyalty programs can reduce churn.
    """)

    st.markdown("</div>", unsafe_allow_html=True)


elif page == "EDA Dashboard":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Exploratory Data Analysis")

    selected_contract = st.multiselect(
        "Filter by Contract Type",
        options=df["Contract"].unique(),
        default=df["Contract"].unique()
    )

    filtered_df = df[df["Contract"].isin(selected_contract)]

    c1, c2 = st.columns(2)

    with c1:
        fig = px.histogram(
            filtered_df,
            x="MonthlyCharges",
            color=filtered_df["Churn"].map({0: "No Churn", 1: "Churn"}),
            nbins=40,
            title="Monthly Charges Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    with c2:
        fig = px.histogram(
            filtered_df,
            x="tenure",
            color=filtered_df["Churn"].map({0: "No Churn", 1: "Churn"}),
            nbins=40,
            title="Customer Tenure Distribution"
        )
        st.plotly_chart(fig, use_container_width=True)

    c3, c4 = st.columns(2)

    with c3:
        payment_churn = filtered_df.groupby("PaymentMethod")["Churn"].mean().reset_index()
        payment_churn["Churn Rate"] = payment_churn["Churn"] * 100

        fig = px.bar(
            payment_churn,
            x="PaymentMethod",
            y="Churn Rate",
            title="Churn Rate by Payment Method"
        )
        fig.update_layout(xaxis_tickangle=-30)
        st.plotly_chart(fig, use_container_width=True)

    with c4:
        internet_churn = filtered_df.groupby("InternetService")["Churn"].mean().reset_index()
        internet_churn["Churn Rate"] = internet_churn["Churn"] * 100

        fig = px.bar(
            internet_churn,
            x="InternetService",
            y="Churn Rate",
            title="Churn Rate by Internet Service"
        )
        st.plotly_chart(fig, use_container_width=True)

    numeric_df = filtered_df.select_dtypes(include=["int64", "float64"])
    corr = numeric_df.corr()

    fig = px.imshow(
        corr,
        text_auto=True,
        title="Correlation Heatmap",
        aspect="auto"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


elif page == "Model Performance":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Model Comparison")

    st.dataframe(model_comparison, use_container_width=True)

    fig = px.bar(
        model_comparison,
        x="Model",
        y=["Accuracy", "Precision", "Recall", "F1 Score", "ROC AUC"],
        barmode="group",
        title="Model Evaluation Metrics"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.success(f"Best Business-Focused Model: {metadata['best_model_name']}")

    st.markdown("</div>", unsafe_allow_html=True)

    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Confusion Matrix")

    fig = px.imshow(
        confusion_matrix.values,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Count"),
        x=["No Churn", "Churn"],
        y=["No Churn", "Churn"],
        title="Confusion Matrix"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.subheader("Classification Report")

    report_df = pd.DataFrame(classification_report).transpose()
    st.dataframe(report_df, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


elif page == "Churn Prediction":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Real-Time Customer Churn Prediction")

    c1, c2, c3 = st.columns(3)

    with c1:
        gender = st.selectbox("Gender", ["Male", "Female"])
        senior_citizen = st.selectbox("Senior Citizen", [0, 1])
        partner = st.selectbox("Partner", ["Yes", "No"])
        dependents = st.selectbox("Dependents", ["Yes", "No"])
        tenure = st.slider("Tenure Months", 0, 72, 12)

    with c2:
        phone_service = st.selectbox("Phone Service", ["Yes", "No"])
        multiple_lines = st.selectbox("Multiple Lines", ["No phone service", "No", "Yes"])
        internet_service = st.selectbox("Internet Service", ["DSL", "Fiber optic", "No"])
        online_security = st.selectbox("Online Security", ["No", "Yes", "No internet service"])
        online_backup = st.selectbox("Online Backup", ["No", "Yes", "No internet service"])

    with c3:
        device_protection = st.selectbox("Device Protection", ["No", "Yes", "No internet service"])
        tech_support = st.selectbox("Tech Support", ["No", "Yes", "No internet service"])
        streaming_tv = st.selectbox("Streaming TV", ["No", "Yes", "No internet service"])
        streaming_movies = st.selectbox("Streaming Movies", ["No", "Yes", "No internet service"])
        contract = st.selectbox("Contract Type", ["Month-to-month", "One year", "Two year"])

    c4, c5, c6 = st.columns(3)

    with c4:
        paperless_billing = st.selectbox("Paperless Billing", ["Yes", "No"])

    with c5:
        payment_method = st.selectbox(
            "Payment Method",
            [
                "Electronic check",
                "Mailed check",
                "Bank transfer (automatic)",
                "Credit card (automatic)"
            ]
        )

    with c6:
        monthly_charges = st.number_input("Monthly Charges", 0.0, 200.0, 70.0)

    total_charges = st.number_input("Total Charges", 0.0, 10000.0, monthly_charges * max(tenure, 1))

    input_data = pd.DataFrame({
        "gender": [gender],
        "SeniorCitizen": [senior_citizen],
        "Partner": [partner],
        "Dependents": [dependents],
        "tenure": [tenure],
        "PhoneService": [phone_service],
        "MultipleLines": [multiple_lines],
        "InternetService": [internet_service],
        "OnlineSecurity": [online_security],
        "OnlineBackup": [online_backup],
        "DeviceProtection": [device_protection],
        "TechSupport": [tech_support],
        "StreamingTV": [streaming_tv],
        "StreamingMovies": [streaming_movies],
        "Contract": [contract],
        "PaperlessBilling": [paperless_billing],
        "PaymentMethod": [payment_method],
        "MonthlyCharges": [monthly_charges],
        "TotalCharges": [total_charges]
    })

    if st.button("Predict Churn Risk"):
        probability = model.predict_proba(input_data)[0][1]
        prediction = model.predict(input_data)[0]
        risk = churn_risk_level(probability)
        recommendation = retention_recommendation(probability, contract, monthly_charges, tenure)

        st.markdown("### Prediction Result")

        r1, r2, r3 = st.columns(3)

        with r1:
            st.metric("Churn Probability", f"{probability * 100:.2f}%")

        with r2:
            st.metric("Prediction", "Churn" if prediction == 1 else "No Churn")

        with r3:
            st.metric("Risk Level", risk)

        if risk == "Low Risk":
            st.markdown(f'<div class="risk-low">{risk}</div>', unsafe_allow_html=True)
        elif risk == "Medium Risk":
            st.markdown(f'<div class="risk-medium">{risk}</div>', unsafe_allow_html=True)
        else:
            st.markdown(f'<div class="risk-high">{risk}</div>', unsafe_allow_html=True)

        st.markdown("### Recommended Retention Action")
        st.info(recommendation)

    st.markdown("</div>", unsafe_allow_html=True)


elif page == "Retention Engine":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Retention Recommendation Engine")

    st.write("""
    ChurnIQ converts machine learning predictions into business actions.
    The system classifies customers into risk levels and recommends practical retention strategies.
    """)

    retention_data = pd.DataFrame({
        "Risk Level": ["Low Risk", "Medium Risk", "High Risk"],
        "Customer Status": [
            "Stable customer",
            "Needs engagement",
            "Likely to churn"
        ],
        "Recommended Strategy": [
            "Continue loyalty communication",
            "Send offers, satisfaction surveys, and usage-based engagement",
            "Offer discount, plan upgrade, personal support, and loyalty reward"
        ]
    })

    st.dataframe(retention_data, use_container_width=True)

    high_risk_contract = df.groupby("Contract")["Churn"].mean().reset_index()
    high_risk_contract["Churn Rate"] = high_risk_contract["Churn"] * 100

    fig = px.bar(
        high_risk_contract,
        x="Contract",
        y="Churn Rate",
        title="Retention Priority by Contract Type",
        text=high_risk_contract["Churn Rate"].round(2)
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)


elif page == "Feature Insights":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Top Churn-Causing Factors")

    st.write("""
    This section shows business-level churn drivers based on churn rate patterns.
    These factors help teams understand why customers leave.
    """)

    factors = []

    for col in ["Contract", "PaymentMethod", "InternetService", "TechSupport", "OnlineSecurity"]:
        temp = df.groupby(col)["Churn"].mean().reset_index()
        temp["Churn Rate"] = temp["Churn"] * 100
        temp["Feature"] = col
        temp.rename(columns={col: "Category"}, inplace=True)
        factors.append(temp[["Feature", "Category", "Churn Rate"]])

    factor_df = pd.concat(factors).sort_values(by="Churn Rate", ascending=False)

    st.dataframe(factor_df, use_container_width=True)

    fig = px.bar(
        factor_df.head(10),
        x="Churn Rate",
        y="Category",
        color="Feature",
        orientation="h",
        title="Top 10 Churn Risk Categories"
    )
    st.plotly_chart(fig, use_container_width=True)

    st.markdown("</div>", unsafe_allow_html=True)

elif page == "Explainable AI":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Explainable AI - Feature Importance")

    importance_df = get_feature_importance(model, df.drop("Churn", axis=1))

    if importance_df is not None:
        st.dataframe(importance_df, use_container_width=True)
        fig = plot_feature_importance(importance_df)
        st.plotly_chart(fig, use_container_width=True)
        st.info("This section explains the most important customer attributes influencing churn prediction.")
    else:
        st.warning("Feature importance is not available for this model.")

    st.markdown("</div>", unsafe_allow_html=True)


elif page == "Customer Segmentation":
    st.markdown('<div class="section-card">', unsafe_allow_html=True)
    st.subheader("Customer Segmentation Dashboard")

    segment_df = create_customer_segments(df)
    summary_df = segment_summary(segment_df)

    st.dataframe(summary_df, use_container_width=True)

    fig = plot_customer_segments(segment_df)
    st.plotly_chart(fig, use_container_width=True)

    csv = summary_df.to_csv(index=False).encode("utf-8")

    st.download_button(
        label="Download Segmentation Report",
        data=csv,
        file_name="churniq_customer_segmentation_report.csv",
        mime="text/csv"
    )

    st.info("Customer segmentation groups users by tenure, charges, value, and churn behavior for targeted retention.")

    st.markdown("</div>", unsafe_allow_html=True)
    

st.markdown("""
<div class="footer">
    ChurnIQ — Customer Churn Intelligence and Retention Prediction System | Built for Business Intelligence and Machine Learning
</div>
""", unsafe_allow_html=True)
