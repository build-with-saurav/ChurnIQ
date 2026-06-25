import os
import json
import joblib
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix, classification_report
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.pipeline import Pipeline

from utils import load_data, clean_data, get_feature_target, get_numeric_categorical_columns
from model import get_models


DATA_PATH = "dataset/WA_Fn-UseC_-Telco-Customer-Churn.csv"
MODEL_DIR = "models"
REPORT_DIR = "reports"


def train_models():
    os.makedirs(MODEL_DIR, exist_ok=True)
    os.makedirs(REPORT_DIR, exist_ok=True)

    df = load_data(DATA_PATH)
    df = clean_data(df)

    X, y = get_feature_target(df)
    numeric_cols, categorical_cols = get_numeric_categorical_columns(X)

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", StandardScaler(), numeric_cols),
            ("cat", OneHotEncoder(handle_unknown="ignore"), categorical_cols)
        ]
    )

    X_train, X_test, y_train, y_test = train_test_split(
        X, y,
        test_size=0.2,
        random_state=42,
        stratify=y
    )

    models = get_models()
    results = []
    best_model = None
    best_model_name = None
    best_score = -1

    for name, model in models.items():
        pipeline = Pipeline(
            steps=[
                ("preprocessor", preprocessor),
                ("model", model)
            ]
        )

        pipeline.fit(X_train, y_train)

        y_pred = pipeline.predict(X_test)
        y_prob = pipeline.predict_proba(X_test)[:, 1]

        metrics = {
            "Model": name,
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred),
            "Recall": recall_score(y_test, y_pred),
            "F1 Score": f1_score(y_test, y_pred),
            "ROC AUC": roc_auc_score(y_test, y_prob)
        }

        results.append(metrics)

        business_score = (0.45 * metrics["Recall"]) + (0.35 * metrics["ROC AUC"]) + (0.20 * metrics["F1 Score"])

        if business_score > best_score:
            best_score = business_score
            best_model = pipeline
            best_model_name = name
            best_y_pred = y_pred
            best_y_prob = y_prob

    results_df = pd.DataFrame(results).sort_values(by="ROC AUC", ascending=False)

    joblib.dump(best_model, f"{MODEL_DIR}/best_churn_model.joblib")

    results_df.to_csv(f"{REPORT_DIR}/model_comparison.csv", index=False)

    cm = confusion_matrix(y_test, best_y_pred)
    pd.DataFrame(cm).to_csv(f"{REPORT_DIR}/confusion_matrix.csv", index=False)

    report = classification_report(y_test, best_y_pred, output_dict=True)
    with open(f"{REPORT_DIR}/classification_report.json", "w") as f:
        json.dump(report, f, indent=4)

    metadata = {
        "best_model_name": best_model_name,
        "best_business_score": best_score,
        "features": X.columns.tolist(),
        "numeric_cols": numeric_cols,
        "categorical_cols": categorical_cols
    }

    with open(f"{REPORT_DIR}/model_metadata.json", "w") as f:
        json.dump(metadata, f, indent=4)

    print("Training completed successfully.")
    print(f"Best Model: {best_model_name}")
    print(results_df)


if __name__ == "__main__":
    train_models()
