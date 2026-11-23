from pathlib import Path

import joblib
import pandas as pd
import yaml

ROOT = Path(__file__).resolve().parents[1]

FEATURES = [
    "loom_age_yr",
    "thread_count",
    "power_cuts_min",
    "weaver_exp_yr",
    "humidity_pct",
    "shift_hours",
    "fabric",
    "shed",
]


# Test 1: Cleaned dataset
def test_cleaned_dataset():
    df = pd.read_csv(ROOT / "data/cleaned_train.csv")

    assert len(df) == 1000
    assert not df.duplicated().any()
    assert df["metres_per_shift"].notna().all()


# Test 2: Target leakage removed
def test_no_data_leakage():
    df = pd.read_csv(ROOT / "data/cleaned_train.csv")

    assert "metres_checked_at_folding" not in df.columns
    assert "metres_per_shift" in df.columns


# Test 3: Data cleaning and feature validation
def test_cleaned_features():
    df = pd.read_csv(ROOT / "data/cleaned_train.csv")

    assert df["loom_age_yr"].between(0, 30).all()
    assert set(df["fabric"].unique()) == {
        "cotton", "silk", "polyester"
    }
    assert all(feature in df.columns for feature in FEATURES)


# Test 4: Champion model loading
def test_champion_model_loads():
    model = joblib.load(ROOT / "models/model.joblib")

    assert hasattr(model, "predict")


# Test 5: Champion model prediction
def test_champion_prediction():
    model = joblib.load(ROOT / "models/model.joblib")
    df = pd.read_csv(ROOT / "data/cleaned_train.csv")

    predictions = model.predict(df[FEATURES].head(5))

    assert len(predictions) == 5
    assert pd.notna(predictions).all()


# Test 6: Holdout prediction submission
def test_holdout_predictions():
    df = pd.read_csv(
        ROOT / "predictions/holdout_predictions.csv"
    )

    assert len(df) == 250
    assert list(df.columns) == ["record_id", "prediction"]
    assert df["record_id"].is_unique
    assert df["prediction"].notna().all()


# Test 7: Retraining and champion promotion
def test_retraining_and_promotion():
    with open(
        ROOT / "reports/findings.yaml",
        encoding="utf-8"
    ) as file:
        findings = yaml.safe_load(file)

    monitoring = findings["monitoring"]
    tracking = findings["tracking"]

    v1_mae = monitoring["mae_v1_on_check"]
    v2_mae = monitoring["mae_v2_on_check"]

    assert v2_mae <= v1_mae * 0.95
    assert monitoring["decision"] == "promote"
    assert tracking["champion_version"] == 3
    assert tracking["registered_model_name"] == (
        "MP181_Power_Loom_Model"
    )