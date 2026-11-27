# MP181 – Model Handover and API Integration

## Final Champion Model

- Algorithm: Ridge Regression V2
- MLflow Registry Version: 3
- Champion MAE: 4.0075 metres
- MAE improvement after retraining: 17.45%
- Model artifact: models/model.joblib

## Required Input Features

- loom_age_yr
- thread_count
- power_cuts_min
- weaver_exp_yr
- humidity_pct
- shift_hours
- fabric
- shed

## Model Loading and Prediction

```python
import joblib
import pandas as pd

model = joblib.load("models/model.joblib")

sample = pd.DataFrame([{
    "loom_age_yr": 10,
    "thread_count": 120,
    "power_cuts_min": 30,
    "weaver_exp_yr": 5,
    "humidity_pct": 60,
    "shift_hours": 8,
    "fabric": "cotton",
    "shed": "Shed 1"
}])

prediction = float(model.predict(sample)[0])
print(prediction)
```

## API Integration Requirements

- Integrate the exported model into the FastAPI prediction endpoint.
- Validate power_cuts_min <= 250.
- Validate shift_hours >= 3.
- Use the exact input feature names.
- No separate preprocessing is required because the exported model includes the preprocessing pipeline.
- Return the predicted metres_per_shift.
- Include the model artifact in the Docker image.

## Handover Status

- Champion model exported.
- 250 holdout predictions generated.
- All 7 model validation tests passed.
- Model Lead findings updated.
- Model ready for service integration and deployment testing.