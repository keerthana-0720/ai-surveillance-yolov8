import joblib
import pandas as pd
from sklearn.ensemble import IsolationForest


DATA_FILE = "combined_dataset.xlsx"
MODEL_FILE = "model.pkl"


def train_model() -> None:
    # Load the Excel dataset.
    df = pd.read_excel(DATA_FILE)

    # Keep numeric columns for system metrics.
    numeric_df = df.select_dtypes(include=["number"]).dropna()
    if numeric_df.empty:
        raise ValueError("No numeric columns found in combined_dataset.xlsx")

    # Train on all numeric metric columns.
    feature_columns = numeric_df.columns.tolist()
    X = numeric_df[feature_columns]

    model = IsolationForest(
        contamination=0.1,
        random_state=42,
    )
    model.fit(X)

    # Save model and feature list for consistent prediction later.
    joblib.dump(
        {
            "model": model,
            "feature_columns": feature_columns,
        },
        MODEL_FILE,
    )

    print(f"Training complete. Metric columns: {feature_columns}")
    print(f"Model saved as {MODEL_FILE}")


if __name__ == "__main__":
    train_model()
