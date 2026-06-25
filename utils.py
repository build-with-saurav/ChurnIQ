import pandas as pd
import numpy as np


def load_data(path):
    df = pd.read_csv(path)
    return df


def clean_data(df):
    df = df.copy()

    df.drop_duplicates(inplace=True)

    if "customerID" in df.columns:
        df.drop("customerID", axis=1, inplace=True)

    df["TotalCharges"] = pd.to_numeric(df["TotalCharges"], errors="coerce")
    df["TotalCharges"] = df["TotalCharges"].fillna(df["TotalCharges"].median())

    if "Churn" in df.columns:
        df["Churn"] = df["Churn"].map({"No": 0, "Yes": 1})

    return df


def get_feature_target(df):
    X = df.drop("Churn", axis=1)
    y = df["Churn"]
    return X, y


def get_numeric_categorical_columns(X):
    numeric_cols = X.select_dtypes(include=["int64", "float64"]).columns.tolist()
    categorical_cols = X.select_dtypes(include=["object"]).columns.tolist()
    return numeric_cols, categorical_cols


def detect_outliers_iqr(df, numeric_cols):
    outlier_summary = {}

    for col in numeric_cols:
        q1 = df[col].quantile(0.25)
        q3 = df[col].quantile(0.75)
        iqr = q3 - q1

        lower = q1 - 1.5 * iqr
        upper = q3 + 1.5 * iqr

        outliers = df[(df[col] < lower) | (df[col] > upper)]
        outlier_summary[col] = len(outliers)

    return outlier_summary


def revenue_loss_estimation(df):
    churned = df[df["Churn"] == 1]
    estimated_loss = churned["MonthlyCharges"].sum()
    return estimated_loss


def churn_risk_level(probability):
    if probability < 0.35:
        return "Low Risk"
    elif probability < 0.65:
        return "Medium Risk"
    else:
        return "High Risk"


def retention_recommendation(probability, contract_type, monthly_charges, tenure):
    if probability >= 0.65:
        if contract_type == "Month-to-month":
            return "Offer long-term contract discount and personalized retention support."
        elif monthly_charges > 70:
            return "Offer pricing discount or plan optimization."
        elif tenure < 12:
            return "Provide onboarding support and loyalty benefits."
        else:
            return "Assign customer success manager and provide loyalty rewards."

    elif probability >= 0.35:
        return "Send engagement offer, service satisfaction survey, and usage-based recommendation."

    else:
        return "Customer is stable. Continue regular engagement and loyalty communication."
