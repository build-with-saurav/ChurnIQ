import pandas as pd
import plotly.express as px


def get_feature_importance(model_pipeline, sample_data):
    preprocessor = model_pipeline.named_steps["preprocessor"]
    model = model_pipeline.named_steps["model"]

    transformed_data = preprocessor.transform(sample_data)

    try:
        feature_names = preprocessor.get_feature_names_out()
    except Exception:
        feature_names = [f"Feature_{i}" for i in range(transformed_data.shape[1])]

    if hasattr(model, "feature_importances_"):
        importance = model.feature_importances_
    elif hasattr(model, "coef_"):
        importance = abs(model.coef_[0])
    else:
        return None

    importance_df = pd.DataFrame({
        "Feature": feature_names,
        "Importance": importance
    }).sort_values(by="Importance", ascending=False).head(15)

    return importance_df


def plot_feature_importance(importance_df):
    fig = px.bar(
        importance_df,
        x="Importance",
        y="Feature",
        orientation="h",
        title="Top Model Feature Importance"
    )
    fig.update_layout(yaxis={"categoryorder": "total ascending"})
    return fig
