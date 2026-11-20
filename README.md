# G037 — MP181: Power Loom Output Prediction (Jaipur Weavers' Cluster)

## Who We Are

| Enrolment No. | Name | Role |
|---|---|---|
| S24CSEU2715 | Rohan Dave |  |
| S24CSEU2479 | Lucky Kumar |  |
| S24CSEU2733 | Pranjal Yadav |  |
| S24CSEU2723 | Anishka Gangwar |  |

**Faculty:** Amit Soni &nbsp;|&nbsp; **Batch:** 72 &nbsp;|&nbsp; **Group:** G037 &nbsp;|&nbsp; **Problem:** MP181

---

## What We're Building

The Jaipur Weavers' Cluster takes cloth orders by the metre and needs reliable delivery estimates. We're building a full MLOps pipeline that predicts **how many metres of cloth a loom will produce in a single shift** (`metres_per_shift`). The model is a regression, and we measure it with **MAE** (Mean Absolute Error) — the average mistake in metres.

This isn't just a model — it's the whole pipeline: versioned data, tracked experiments, a live prediction API, a container, automated tests, Kubernetes configs, drift monitoring, and a retrain trigger.

---

## How to Get Started

### 1. Set Up Your Environment
```bash
python -m venv venv
venv\Scripts\activate        # Windows
# source venv/bin/activate   # Mac / Linux
pip install -r requirements.txt
```

### 2. Train the Model & Track with MLflow
```bash
python -m src.mp181.train
```

### 3. Run Drift Monitoring & Retraining
```bash
python -m src.mp181.monitoring
```

### 4. Start the Prediction API
```bash
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 5. Run the Tests
```bash
pytest -v
```

### 6. Build & Run the Docker Container
```bash
docker build -t mp181-api:1.0 .
docker run -p 8000:8000 mp181-api:1.0
```

---

## Pipeline at a Glance

- **Data versioning** — DVC tracks `train.csv` and `later.csv` so the exact files are always reproducible.
- **Data quality** — We hunt and fix leaking columns, unit mixups (Shed 1 loom age recorded in months, not years), duplicate rows, and messy fabric-name spellings.
- **Experiment tracking** — MLflow logs every training run to `mlflow.db` (SQLite). We log the SHA-256 hash of `train.csv`, MAE on validation, and register the best model under the `@champion` alias.
- **Prediction service** — A FastAPI app at `POST /predict` and `GET /health`. Business rules are enforced right in the schema: `power_cuts_min` can't exceed 250, and `shift_hours` must be at least 3.
- **Containerisation** — Docker image `mp181-api:1.0`, built from `python:3.12-slim`.
- **Kubernetes** — Deployment `mp181-api` with 3 replicas, Service `mp181-svc` on port 80 → 8000, readiness & liveness probes on `/health`.
- **CI/CD** — GitHub Actions runs `pytest` on every push and must stay green before submission.
- **Monitoring & retraining** — PSI drift scores per numeric column (train vs. later), MAE per shed group, and a controlled retrain that promotes v2 only if it beats v1 by at least 5% MAE.
