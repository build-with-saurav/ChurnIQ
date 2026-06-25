import pandas as pd
import plotly.express as px
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def create_customer_segments(df):
    segment_df = df[["tenure", "MonthlyCharges", "TotalCharges", "Churn"]].copy()

    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(segment_df[["tenure", "MonthlyCharges", "TotalCharges"]])

    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    segment_df["Segment"] = kmeans.fit_predict(scaled_data)

    segment_names = {
        0: "New / Low Value",
        1: "High Value Loyal",
        2: "Premium At Risk",
        3: "Moderate Value"
    }

    segment_df["Segment Name"] = segment_df["Segment"].map(segment_names)

    return segment_df


def plot_customer_segments(segment_df):
    fig = px.scatter(
        segment_df,
        x="tenure",
        y="MonthlyCharges",
        color="Segment Name",
        size="TotalCharges",
        hover_data=["Churn"],
        title="Customer Segmentation: Tenure vs Monthly Charges"
    )

    return fig


def segment_summary(segment_df):
    summary = segment_df.groupby("Segment Name").agg(
        Customers=("Segment Name", "count"),
        Avg_Tenure=("tenure", "mean"),
        Avg_Monthly_Charges=("MonthlyCharges", "mean"),
        Churn_Rate=("Churn", "mean")
    ).reset_index()

    summary["Churn_Rate"] = summary["Churn_Rate"] * 100

    return summary
