import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.ensemble import (
    RandomForestRegressor,
    ExtraTreesRegressor,
    GradientBoostingRegressor,
    HistGradientBoostingRegressor,
)
from sklearn.linear_model import Ridge
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler

# Load configuration and cleaned data
with open("params.yaml", "r", encoding="utf-8") as f:
    config = yaml.safe_load(f)

df = pd.read_csv(config["data"]["cleaned_train"])

numeric = config["features"]["numeric"]
categorical = config["features"]["categorical"]
features = numeric + categorical
target = config["target"]

X = df[features]
y = df[target]

X_train, X_val, y_train, y_val = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42,
)

models = {
    "Random Forest": RandomForestRegressor(
        n_estimators=200,
        max_depth=15,
        random_state=42,
        n_jobs=-1,
    ),
    "Extra Trees": ExtraTreesRegressor(
        n_estimators=300,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1,
    ),
    "Gradient Boosting": GradientBoostingRegressor(
        n_estimators=300,
        learning_rate=0.05,
        max_depth=3,
        random_state=42,
    ),
    "Hist Gradient Boosting": HistGradientBoostingRegressor(
        max_iter=300,
        learning_rate=0.05,
        max_leaf_nodes=15,
        random_state=42,
    ),
    "Ridge Regression": Ridge(alpha=10.0),
}

results = []

for name, regressor in models.items():

    preprocessor = ColumnTransformer([
        ("numeric", StandardScaler(), numeric),
        ("categorical", OneHotEncoder(handle_unknown="ignore",
                                      sparse_output=False), categorical),
    ])

    pipeline = Pipeline([
        ("preprocessor", preprocessor),
        ("regressor", regressor),
    ])

    pipeline.fit(X_train, y_train)

    predictions = pipeline.predict(X_val)

    mae = mean_absolute_error(y_val, predictions)
    r2 = r2_score(y_val, predictions)

    results.append({
        "Model": name,
        "MAE": round(mae, 4),
        "R2": round(r2, 4),
    })

results_df = pd.DataFrame(results).sort_values("MAE")

print("\n========== MODEL COMPARISON ==========")
print(results_df.to_string(index=False))

print("\nBEST MODEL:", results_df.iloc[0]["Model"])
print("BEST MAE:", results_df.iloc[0]["MAE"])