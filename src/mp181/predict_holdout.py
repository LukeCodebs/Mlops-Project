from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml
from mlflow.tracking import MlflowClient


MODEL_NAME = "MP181_Power_Loom_Model"


def main():
    with open("params.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri("sqlite:///mlflow.db")

    client = MlflowClient()
    champion = client.get_model_version_by_alias(
        MODEL_NAME, "champion"
    )

    model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}@champion"
    )

    holdout = pd.read_csv(
        config["data"]["holdout_features"]
    )

    features = (
        config["features"]["numeric"]
        + config["features"]["categorical"]
    )

    # Normalize fabric categories
    holdout["fabric"] = (
        holdout["fabric"]
        .astype(str)
        .str.strip()
        .str.lower()
        .replace({
            "cot.": "cotton",
            "sil.": "silk",
            "pol.": "polyester"
        })
    )

    # Validate input
    assert "record_id" in holdout.columns
    assert all(col in holdout.columns for col in features)
    assert holdout["record_id"].is_unique

    predictions = model.predict(holdout[features])

    output = pd.DataFrame({
        "record_id": holdout["record_id"],
        "prediction": predictions
    })

    Path("predictions").mkdir(
        parents=True,
        exist_ok=True
    )

    output_path = "predictions/holdout_predictions.csv"
    output.to_csv(output_path, index=False)

    print("\n========== HOLDOUT PREDICTIONS ==========")
    print("Champion Version:", champion.version)
    print("Input Rows:", len(holdout))
    print("Output Rows:", len(output))
    print("Columns:", output.columns.tolist())
    print("Missing Predictions:", output["prediction"].isna().sum())
    print("\nFirst 5 predictions:")
    print(output.head())
    print("\nSaved:", output_path)


if __name__ == "__main__":
    main()