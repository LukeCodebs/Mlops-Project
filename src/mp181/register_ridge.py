import hashlib
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd
import yaml

from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


def main():
    # 1. Load project configuration
    with open("params.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    df = pd.read_csv(config["data"]["cleaned_train"])

    numeric = config["features"]["numeric"]
    categorical = config["features"]["categorical"]
    target = config["target"]
    features = numeric + categorical

    # Prevent data leakage
    assert target not in features
    assert "metres_checked_at_folding" not in features

    X = df[features]
    y = df[target]

    # 2. Same split as original model comparison
    X_train, X_val, y_train, y_val = train_test_split(
        X,
        y,
        test_size=0.2,
        random_state=42
    )

    # 3. Feature preprocessing
    preprocessor = ColumnTransformer([
        (
            "numeric",
            StandardScaler(),
            numeric
        ),
        (
            "categorical",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False
            ),
            categorical
        )
    ])

    # 4. Original Ridge Regression (alpha = 10)
    model = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=10.0))
    ])

    model.fit(X_train, y_train)

    # 5. Evaluate model
    predictions = model.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    r2 = r2_score(y_val, predictions)

    # 6. Calculate SHA256 of raw training dataset
    raw_path = Path(config["data"]["raw_train"])

    raw_sha256 = hashlib.sha256(
        raw_path.read_bytes()
    ).hexdigest()

    # 7. Configure MLflow
    mlflow.set_tracking_uri("sqlite:///mlflow.db")
    mlflow.set_experiment("MP181_Power_Loom_Regression")

    model_name = "MP181_Power_Loom_Model"

    # 8. Log experiment and model
    with mlflow.start_run(
        run_name="Ridge_Alpha_10_Champion"
    ) as run:

        mlflow.log_param("algorithm", "Ridge Regression")
        mlflow.log_param("alpha", 10.0)
        mlflow.log_param("random_state", 42)
        mlflow.log_param("test_size", 0.2)
        mlflow.log_param("raw_train_sha256", raw_sha256)

        mlflow.log_metric("mae", mae)
        mlflow.log_metric("r2", r2)

        # Model signature and input example
        signature = infer_signature(
            X_train,
            model.predict(X_train)
        )

        mlflow.sklearn.log_model(
            sk_model=model,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(3)
        )

        run_id = run.info.run_id

    # 9. Register model in MLflow Registry
    registered = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=model_name
    )

    # 10. Assign champion alias
    client = MlflowClient()

    client.set_registered_model_alias(
        name=model_name,
        alias="champion",
        version=registered.version
    )

    # 11. Verify registration
    champion = client.get_model_version_by_alias(
        model_name,
        "champion"
    )

    print("\n========== MODEL REGISTRATION ==========")
    print("Model Name:", model_name)
    print("Algorithm: Ridge Regression")
    print("Alpha: 10.0")
    print("Validation MAE:", round(mae, 4))
    print("Validation R2:", round(r2, 4))
    print("Registered Version:", registered.version)
    print("Champion Version:", champion.version)
    print("Run ID:", run_id)
    print("Registration completed successfully!")


if __name__ == "__main__":
    main()