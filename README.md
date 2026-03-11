# Titanic MLOps Pipeline (Airflow + MLflow)

End-to-end Machine Learning pipeline for predicting survival on the Titanic dataset using:

- **Apache Airflow** for workflow orchestration (DAG with parallel tasks + branching)
- **MLflow** for experiment tracking and **Model Registry**
- **Scikit-learn** for ML model training/evaluation

## 1) Project Structure

Recommended structure:

```text
.
mlops_assignment/
├── airflow/
│   └── dags/
│       └── mlops_airflow_mlflow_pipeline.py
├── data/
│   └── Titanic-Dataset.csv
├── models/
│   └── generated at runtime
├── mlruns/
│   └── generated at runtime by MLflow
├── screenshots/
│   └── Important screenshots from the project
├── Technical_Report.pdf
├── requirements.txt
└── README.md
```

## 2) Prerequisites (Ubuntu)

- Ubuntu 20.04+ recommended
- Python 3.10+ (3.11 also works)
- `pip`, `venv`
- A browser for Airflow UI & MLflow UI

Install system dependencies:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv
python3 --version
```

## 3) Setup (Virtual Environment + Dependencies)

From the repository root:

```bash
python3 -m venv mlops_env
source mlops_env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

## 4) Dataset Setup

Download the dataset from Kaggle:

- Titanic dataset: https://www.kaggle.com/datasets/yasserh/titanic-dataset/data
- Download the file named **`Titanic-Dataset.csv`**

Place it here:

```text
data/Titanic-Dataset.csv
```

Example command (if it is in your Downloads folder):

```bash
mkdir -p data
cp ~/Downloads/Titanic-Dataset.csv data/Titanic-Dataset.csv
```

## 5) Airflow Initialization

### 5.1 Set AIRFLOW_HOME (important)

From repo root:

```bash
export AIRFLOW_HOME="$PWD/airflow"
mkdir -p "$AIRFLOW_HOME/dags"
```

Confirm the DAG file is in:

```text
airflow/dags/mlops_airflow_mlflow_pipeline.py
```

### 5.2 Initialize DB and create admin user

```bash
airflow db init

airflow users create \
  --username admin \
  --firstname Admin \
  --lastname User \
  --role Admin \
  --email admin@example.com \
  --password admin
```

## 6) Running the System (3 Terminals)

Open **three terminals**, and in each one:

```bash
cd <your-repo-folder>
source mlops_env/bin/activate
export AIRFLOW_HOME="$PWD/airflow"
```

### Terminal A — Airflow Webserver

```bash
airflow webserver --port 8080
```

Airflow UI: `http://localhost:8080`  
Login: `admin` / `admin`

### Terminal B — Airflow Scheduler

```bash
airflow scheduler
```

### Terminal C — MLflow UI

MLflow uses a local `mlruns/` folder (created automatically):

```bash
mlflow ui --backend-store-uri "file://$PWD/mlruns" --port 5000
```

MLflow UI: `http://localhost:5000`

## 7) Running the DAG

1. Go to Airflow UI: `http://localhost:8080`
2. Find the DAG: **`mlops_airflow_mlflow_pipeline`**
3. Toggle it **ON** (unpause)
4. Click **Trigger DAG** (play button ▶)

You can also trigger from CLI:

```bash
airflow dags trigger mlops_airflow_mlflow_pipeline
```

## 8) What the DAG Does (Task Mapping)

- **Task 2** `data_ingestion`: loads CSV, prints shape, logs missing values, pushes dataset path using XCom
- **Task 3** `data_validation`: checks missing % for Age and Embarked, fails if > 30% (configured with retries)
- **Task 4 (Parallel)**:
  - `handle_missing_values`: fills missing Age/Embarked
  - `feature_engineering`: creates FamilySize, IsAlone
- **Task 5** `data_encoding`: encodes Sex & Embarked, drops irrelevant columns
- **Task 6** `model_training`: starts MLflow run, logs params, trains model, logs model artifact
- **Task 7** `model_evaluation`: computes accuracy/precision/recall/F1, logs metrics to MLflow, pushes accuracy via XCom
- **Task 8** `check_accuracy` (BranchPythonOperator): accuracy ≥ 0.80 → register else reject
- **Task 9** `register_model` / `reject_model`: registers in MLflow Model Registry or logs rejection reason

No cyclic dependencies: tasks move forward and converge after parallel steps.

## 9) Experiment Comparison (Run DAG 3 Times)

To complete Task 10, run the DAG **at least 3 times** with different hyperparameters.

### Approach (simple for assignment)
Edit these variables inside `airflow/dags/mlops_airflow_mlflow_pipeline.py`:

- `MODEL_TYPE = "LogisticRegression"` or `"RandomForest"`
- `HYPERPARAMS = {...}`

Then re-run the DAG each time.

**Example runs to try:**
1. LogisticRegression: `C=1.0`
2. RandomForest: `n_estimators=100`, `max_depth=10`
3. RandomForest: `n_estimators=200`, `max_depth=5`

In MLflow UI:
- Go to experiment: `Titanic_Survival_Prediction`
- Select all 3 runs → click **Compare**
- Identify best run by **Accuracy** and justify in report.

## 10) Retry Experiment (Intentional Failure Demo)

To demonstrate retries for validation:
- Temporarily increase missing values in `Age` to exceed 30%, then trigger DAG.
- `data_validation` will fail and show retries (up_for_retry) in Airflow logs.

**Important:** Do this only for experimentation purpose, then revert dataset to normal.

## 11) Notes / Troubleshooting

### DAG not showing in Airflow UI
- Confirm:
  - `export AIRFLOW_HOME="$PWD/airflow"`
  - DAG is inside: `airflow/dags/`
- Check for syntax errors:
  - Airflow UI → browse logs, or run:
    ```bash
    python airflow/dags/mlops_airflow_mlflow_pipeline.py
    ```

### Port already in use
Try different ports:
- Airflow: `airflow webserver --port 8081`
- MLflow: `mlflow ui --port 5001`

## 12) License
This repository is for academic purposes.
