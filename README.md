# MLOps Anomaly Detection System

An anomaly detection service for system-metric data using **Isolation Forest**, exposed through a **FastAPI** REST API.

## Overview

This project trains an unsupervised anomaly detection model on numeric system metrics and serves predictions through an API.

**Workflow:**

`Excel dataset → preprocessing → Isolation Forest → saved model → FastAPI /predict endpoint`

## Tech Stack

- Python
- Pandas
- Scikit-learn
- Isolation Forest
- FastAPI
- Uvicorn
- Joblib
- OpenPyXL

## Project Structure

```text
.
├── app.py                 # FastAPI application
├── train.py               # Model training pipeline
├── model.pkl              # Trained model artifact
├── requirements.txt       # Python dependencies
└── deploy/
    ├── anomaly-api.service
    ├── ec2-setup.sh
    └── push-to-ec2.ps1
```

## How It Works

1. `train.py` reads the input Excel dataset.
2. Numeric columns are selected and missing rows are removed.
3. An `IsolationForest` model is trained with a contamination value of `0.1`.
4. The trained model and feature-column list are saved to `model.pkl`.
5. `app.py` loads the model and exposes a `/predict` endpoint.
6. The API checks that all expected metrics are supplied and returns either **Normal** or **Anomaly**.

## Running Locally

Create and activate a virtual environment:

```bash
python -m venv venv
```

Windows:

```bash
venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

### Train the model

Place your dataset as:

```text
combined_dataset.xlsx
```

in the project root, then run:

```bash
python train.py
```

This generates `model.pkl`.

### Start the API

```bash
uvicorn app:app --reload
```

The API will be available at:

```text
http://127.0.0.1:8000
```

FastAPI's interactive documentation is available at:

```text
http://127.0.0.1:8000/docs
```

## API

### `GET /`

Health check:

```json
{
  "message": "Anomaly Detection API Running"
}
```

### `POST /predict`

Example request:

```json
{
  "metrics": {
    "metric_1": 10.5,
    "metric_2": 25.1
  }
}
```

The exact metric names depend on the columns present in the training dataset. The API reports the required feature columns when input is incomplete.

## Deployment

The repository includes deployment scripts for an EC2-based setup under `deploy/`.

Review the deployment scripts and replace environment-specific values before using them in a new environment.

## Dataset

The original Excel dataset is **not included in this public repository**. This keeps the repository lightweight and avoids publishing project data unnecessarily.

To reproduce training, provide your own `combined_dataset.xlsx` with the required numeric metric columns.

## Notes

This repository contains the source code and deployment configuration for the project. Model performance depends on the dataset and feature distributions used for training.
