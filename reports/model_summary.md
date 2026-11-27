# MP181 – Model Performance Summary

## Model Comparison

| Model | Validation MAE |
|---|---:|
| Ridge Regression (alpha=10) | 1.9964 |
| Fine-tuned Ridge (alpha=1) | 2.0003 |
| Gradient Boosting | 2.3281 |
| HistGradientBoosting | 2.3427 |
| Extra Trees | 2.9035 |
| Random Forest | 3.1541 |

## Hyperparameter Tuning

Ridge Regression with alpha=10 was retained because it achieved a lower validation MAE than the fine-tuned model.

## Retraining Results

The model was retrained using 1,000 cleaned training records and 210 additional records from later.csv. Both versions were evaluated on the remaining 90 records.

- V1 MAE: 4.8545
- V2 MAE: 4.0075
- MAE improvement: 17.45%
- Required improvement: 5%
- Decision: Promote V2

The initial validation MAE and retraining evaluation MAE are based on different datasets and are not directly comparable.

## Final Champion Model

- Algorithm: Ridge Regression V2
- MLflow registered model: MP181_Power_Loom_Model
- Registry version: 3
- Alias: champion
- Final MAE: 4.0075
- Exported model: models/model.joblib
- Holdout predictions: 250
- Automated tests: 7/7 passed

The exported model includes the preprocessing and regression pipeline for API integration.