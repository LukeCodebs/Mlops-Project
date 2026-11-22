import hashlib
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd
import yaml

from mlflow.models import infer_signature
from mlflow.tracking import MlflowClient

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


MODEL_NAME = "MP181_Power_Loom_Model"
TRACKING_URI = "sqlite:///mlflow.db"


def normalize_fabric(series):
    """Apply the same fabric normalization as training."""
    mapping = {
        "cot.": "cotton",
        "sil.": "silk",
        "pol.": "polyester",
    }

    values = series.astype(str).str.strip().str.lower()
    return values.replace(mapping)


def main():
    # 1. Load configuration
    with open("params.yaml", "r", encoding="utf-8") as f:
        config = yaml.safe_load(f)

    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment("MP181_Power_Loom_Regression")

    client = MlflowClient()

    numeric = config["features"]["numeric"]
    categorical = config["features"]["categorical"]
    features = numeric + categorical
    target = config["target"]

    split_ratio = config["monitoring"]["later_split_ratio"]
    threshold = config["monitoring"][
        "promotion_improvement_threshold"
    ]

    # 2. Load existing champion
    champion = client.get_model_version_by_alias(
        MODEL_NAME,
        "champion"
    )

    old_version = champion.version

    v1 = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}@champion"
    )

    print("\n========== CURRENT CHAMPION ==========")
    print("Model:", MODEL_NAME)
    print("Version:", old_version)

    # 3. Load cleaned training data
    train_df = pd.read_csv(
        config["data"]["cleaned_train"]
    )

    # 4. Load and chronologically sort later.csv
    later_df = pd.read_csv(
        config["data"]["raw_later"]
    )

    later_df["date"] = pd.to_datetime(
        later_df["date"]
    )

    later_df = later_df.sort_values(
        ["date", "record_id"]
    ).reset_index(drop=True)

    # Normalize categorical values
    train_df["fabric"] = normalize_fabric(
        train_df["fabric"]
    )

    later_df["fabric"] = normalize_fabric(
        later_df["fabric"]
    )

    # 5. Chronological 70/30 split
    split_index = int(
        len(later_df) * split_ratio
    )

    later_train = later_df.iloc[:split_index].copy()
    later_test = later_df.iloc[split_index:].copy()

    print("\n========== DATA SPLIT ==========")
    print("Original cleaned train:", len(train_df))
    print("Later retraining rows:", len(later_train))
    print("Later testing rows:", len(later_test))

    # 6. Combine cleaned train + first 70% later
    combined_train = pd.concat(
        [
            train_df[features + [target]],
            later_train[features + [target]]
        ],
        ignore_index=True
    )

    X_train = combined_train[features]
    y_train = combined_train[target]

    X_test = later_test[features]
    y_test = later_test[target]

    # 7. Evaluate existing V1 champion
    v1_predictions = v1.predict(X_test)

    v1_mae = mean_absolute_error(
        y_test,
        v1_predictions
    )

    v1_r2 = r2_score(
        y_test,
        v1_predictions
    )

    # 8. Create V2 Ridge pipeline
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

    v2 = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", Ridge(alpha=10.0))
    ])

    # 9. Train V2
    v2.fit(X_train, y_train)

    # 10. Evaluate V2 on same test data
    v2_predictions = v2.predict(X_test)

    v2_mae = mean_absolute_error(
        y_test,
        v2_predictions
    )

    v2_r2 = r2_score(
        y_test,
        v2_predictions
    )

    # 11. Calculate improvement
    improvement = (
        (v1_mae - v2_mae) / v1_mae
    ) if v1_mae > 0 else 0.0

    should_promote = improvement >= threshold

    print("\n========== MODEL COMPARISON ==========")
    print("V1 MAE:", round(v1_mae, 4))
    print("V2 MAE:", round(v2_mae, 4))
    print("V1 R2:", round(v1_r2, 4))
    print("V2 R2:", round(v2_r2, 4))
    print(
        "Improvement (%):",
        round(improvement * 100, 2)
    )
    print(
        "Required Improvement (%):",
        threshold * 100
    )

    # 12. Calculate SHA256 of original raw train
    raw_sha256 = hashlib.sha256(
        Path(
            config["data"]["raw_train"]
        ).read_bytes()
    ).hexdigest()

    # 13. Log V2 experiment in MLflow
    with mlflow.start_run(
        run_name="MP181_Ridge_V2_Retraining"
    ) as run:

        mlflow.log_param(
            "algorithm",
            "Ridge Regression"
        )
        mlflow.log_param("alpha", 10.0)
        mlflow.log_param("data_sha256", raw_sha256)
        mlflow.log_param(
            "original_train_rows",
            len(train_df)
        )
        mlflow.log_param(
            "later_train_rows",
            len(later_train)
        )
        mlflow.log_param(
            "later_test_rows",
            len(later_test)
        )
        mlflow.log_param(
            "promotion_threshold",
            threshold
        )

        mlflow.log_metric("v1_mae_later", v1_mae)
        mlflow.log_metric("v2_mae_later", v2_mae)
        mlflow.log_metric("v1_r2_later", v1_r2)
        mlflow.log_metric("v2_r2_later", v2_r2)
        mlflow.log_metric(
            "improvement_pct",
            improvement * 100
        )

        mlflow.set_tag(
            "promotion_decision",
            "promote" if should_promote else "retain"
        )

        signature = infer_signature(
            X_train,
            v2.predict(X_train)
        )

        mlflow.sklearn.log_model(
            sk_model=v2,
            artifact_path="model",
            signature=signature,
            input_example=X_train.head(3)
        )

        run_id = run.info.run_id

    # 14. Register V2 as new MLflow version
    registered = mlflow.register_model(
        model_uri=f"runs:/{run_id}/model",
        name=MODEL_NAME
    )

    # 15. Apply client's 5% promotion rule
    if should_promote:
        client.set_registered_model_alias(
            name=MODEL_NAME,
            alias="champion",
            version=registered.version
        )

        decision = "PROMOTED"
    else:
        decision = "NOT PROMOTED"

    # 16. Verify final champion
    final_champion = client.get_model_version_by_alias(
        MODEL_NAME,
        "champion"
    )

    # 17. Save results for findings.yaml
    Path("reports").mkdir(
        parents=True,
        exist_ok=True
    )

    results = {
        "v1_registry_version": int(old_version),
        "v2_registry_version": int(
            registered.version
        ),
        "v1_mae_later": float(v1_mae),
        "v2_mae_later": float(v2_mae),
        "improvement_pct": float(
            improvement * 100
        ),
        "promotion_threshold_pct": float(
            threshold * 100
        ),
        "promoted": bool(should_promote),
        "final_champion_version": int(
            final_champion.version
        ),
    }

    with open(
        "reports/retraining_results.yaml",
        "w",
        encoding="utf-8"
    ) as f:
        yaml.safe_dump(
            results,
            f,
            sort_keys=False
        )

    print("\n========== PROMOTION DECISION ==========")
    print("Decision:", decision)
    print("Registered V2:", registered.version)
    print(
        "Final Champion:",
        final_champion.version
    )
    print("MLflow Run ID:", run_id)
    print(
        "Results saved to:",
        "reports/retraining_results.yaml"
    )
    print("\nRetraining pipeline completed!")


if __name__ == "__main__":
    main()