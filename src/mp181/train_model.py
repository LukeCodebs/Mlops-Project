import hashlib
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder


def main():
    # Read project configuration
    with open("params.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    data_path = Path(config["data"]["cleaned_train"])
    raw_path = Path(config["data"]["raw_train"])

    df = pd.read_csv(data_path)

    numeric = config["features"]["numeric"]
    categorical = config["features"]["categorical"]
    target = config["target"]
    features = numeric + categorical

    # Prevent target leakage
    assert "metres_checked_at_folding" not in features
    assert target not in features
    assert set(features + [target]).issubset(df.columns)

    X = df[features]
    y = df[target]

    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2,
        random_state=config["model"]["random_state"]
    )

    preprocessing = ColumnTransformer([
        ("numeric", SimpleImputer(strategy="median"), numeric),
        ("categorical", Pipeline([
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("encoder", OneHotEncoder(handle_unknown="ignore"))
        ]), categorical)
    ])

    # SHA256 of original raw training data
    sha256 = hashlib.sha256(raw_path.read_bytes()).hexdigest()

    # SQLite-backed MLflow tracking
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("MP181_Power_Loom_Regression")

    # Three experiments with different model configurations
    experiments = [
        {"n_estimators": 50, "max_depth": 5},
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 15},
    ]

    results = []

    for i, params in enumerate(experiments, start=1):
        model = Pipeline([
            ("preprocessor", preprocessing),
            ("regressor", RandomForestRegressor(
                n_estimators=params["n_estimators"],
                max_depth=params["max_depth"],
                random_state=config["model"]["random_state"],
                n_jobs=-1
            ))
        ])

        with mlflow.start_run(run_name=f"RF_Experiment_{i}") as run:
            model.fit(X_train, y_train)
            predictions = model.predict(X_val)
            mae = mean_absolute_error(y_val, predictions)

            mlflow.log_params(params)
            mlflow.log_param("random_state", config["model"]["random_state"])
            mlflow.log_param("raw_train_sha256", sha256)
            mlflow.log_metric("mae", mae)
            mlflow.sklearn.log_model(
                model, artifact_path="model"
            )

            results.append({
                "experiment": i,
                "run_id": run.info.run_id,
                "mae": mae
            })

            print(f"Experiment {i}: MAE = {mae:.4f}")

    best = min(results, key=lambda item: item["mae"])

    print("\nBest experiment:", best["experiment"])
    print("Best MAE:", round(best["mae"], 4))
    print("Best run ID:", best["run_id"])


if __name__ == "__main__":
    main()