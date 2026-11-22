import pandas as pd
import yaml

from sklearn.compose import ColumnTransformer
from sklearn.linear_model import Ridge
from sklearn.model_selection import (
    KFold, GridSearchCV, train_test_split
)
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score

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
    X, y, test_size=0.2, random_state=42
)

preprocessor = ColumnTransformer([
    ("numeric", StandardScaler(), numeric),
    ("categorical", OneHotEncoder(
        handle_unknown="ignore",
        sparse_output=False
    ), categorical),
])

pipeline = Pipeline([
    ("preprocessor", preprocessor),
    ("regressor", Ridge())
])

param_grid = {
    "regressor__alpha": [
        0.01, 0.1, 1, 5, 10, 20, 50, 100
    ]
}

cv = KFold(
    n_splits=5,
    shuffle=True,
    random_state=42
)

search = GridSearchCV(
    pipeline,
    param_grid,
    cv=cv,
    scoring="neg_mean_absolute_error",
    n_jobs=-1
)

search.fit(X_train, y_train)

predictions = search.best_estimator_.predict(X_val)

print("\nBEST ALPHA:", search.best_params_)
print("CROSS-VALIDATION MAE:", round(-search.best_score_, 4))
print("VALIDATION MAE:", round(
    mean_absolute_error(y_val, predictions), 4
))
print("VALIDATION R2:", round(
    r2_score(y_val, predictions), 4
))