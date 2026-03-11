# Titanic Survival Prediction — MLOps Pipeline (Airflow + MLflow)

This project builds a complete (but lightweight) MLOps workflow to predict passenger survival on the Titanic dataset. The main idea is to run the full machine learning lifecycle as an automated **Airflow DAG** while tracking experiments and models using **MLflow** (including the **Model Registry**).

It’s designed to be easy to run locally on Ubuntu with three terminals: Airflow webserver, Airflow scheduler, and MLflow UI.

---

## Tech Stack

- **Apache Airflow** — orchestration (parallel tasks + branching)
- **MLflow** — experiment tracking + Model Registry
- **Scikit-learn** — training and evaluation (Logistic Regression / Random Forest)

---

## Project Layout

Make sure your repository looks like the structure below (some folders will appear only after you run the pipeline):

```text
.
Titanic Survival Prediction/
├── airflow/
│   └── dags/
│       └── mlops_airflow_mlflow_pipeline.py
├── data/
│   └── Titanic-Dataset.csv
├── models/
│   └── (created at runtime)
├── mlruns/
│   └── (created automatically by MLflow)
└── requirements.txt
```

---

## 1. Requirements (Ubuntu)

Recommended:
- Ubuntu 20.04+
- Python 3.10+ (3.11 works fine)
- `pip` and `venv`
- Browser access for Airflow UI + MLflow UI

Install system packages:

```bash
sudo apt update && sudo apt upgrade -y
sudo apt install -y python3 python3-pip python3-venv
python3 --version
```

---

## 2. Environment Setup (venv + dependencies)

From the repo root:

```bash
python3 -m venv mlops_env
source mlops_env/bin/activate

pip install --upgrade pip
pip install -r requirements.txt
```

---

## 3. Dataset Setup

Download the dataset from Kaggle:

- Dataset page: https://www.kaggle.com/datasets/yasserh/titanic-dataset/data  
- File needed: **Titanic-Dataset.csv**

Place it here:

```text
data/Titanic-Dataset.csv
```

Example (if the file is in `~/Downloads`):

```bash
mkdir -p data
cp ~/Downloads/Titanic-Dataset.csv data/Titanic-Dataset.csv
```

---

## 4. Airflow Initialization

### 4.1 Set `AIRFLOW_HOME`

From your repository root:

```bash
export AIRFLOW_HOME="$PWD/airflow"
mkdir -p "$AIRFLOW_HOME/dags"
```

Confirm that the DAG exists at:

```text
airflow/dags/mlops_airflow_mlflow_pipeline.py
```

### 4.2 Initialize the Airflow DB and create an admin account

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

---

## 5. Run Everything (3 terminals)

Open **three terminals**. In each terminal, run:

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

MLflow creates a local `mlruns/` directory automatically:

```bash
mlflow ui --backend-store-uri "file://$PWD/mlruns" --port 5000
```

MLflow UI: `http://localhost:5000`

---

## 6. Trigger the DAG

From the Airflow UI:
1. Open `http://localhost:8080`
2. Find the DAG: **mlops_airflow_mlflow_pipeline**
3. Turn it **ON** (unpause)
4. Click **Trigger DAG**

You can also trigger from CLI:

```bash
airflow dags trigger mlops_airflow_mlflow_pipeline
```

---

## 7. What the DAG Does (high-level flow)

The DAG follows a clean “data → preprocessing → training → evaluation → decision” structure.

### Task breakdown

- **data_ingestion**  
  Loads the CSV, prints dataset shape, shows missing values summary, and passes the dataset path using XCom.

- **data_validation**  
  Checks missing percentage for `Age` and `Embarked`. If missing values exceed **30%**, the task fails (and retries are enabled so you can see retry behavior in Airflow).

- **Parallel preprocessing (runs at the same time)**  
  - **handle_missing_values**: fills missing values for key columns  
  - **feature_engineering**: creates features like `FamilySize` and `IsAlone`

- **data_encoding**  
  Encodes `Sex` and `Embarked`, removes columns not needed for training.

- **model_training**  
  Starts an MLflow run, logs parameters, trains the selected model, and logs the trained model artifact.

- **model_evaluation**  
  Calculates accuracy, precision, recall, and F1-score, logs metrics to MLflow, and sends accuracy via XCom.

- **check_accuracy** (Branching step)  
  If accuracy is **≥ 0.80**, the pipeline continues to model registration. Otherwise, it goes to rejection.

- **register_model / reject_model**  
  Registers the model to MLflow Model Registry if it meets the threshold, otherwise stores a rejection reason.

This DAG does not have cyclic dependencies—tasks progress forward and the pipeline merges properly after the parallel preprocessing step.

---

## 8. Experiment Comparison (run at least 3 times)

To compare multiple experiments, trigger the DAG **three separate times** with different model choices / hyperparameters.

### Simple method (edit values in the DAG file)

Inside:

```text
airflow/dags/mlops_airflow_mlflow_pipeline.py
```

Update:
- `MODEL_TYPE` (example: `"LogisticRegression"` or `"RandomForest"`)
- `HYPERPARAMS` (dictionary of parameters)

Suggested runs:
1. LogisticRegression with `C=1.0`
2. RandomForest with `n_estimators=100`, `max_depth=10`
3. RandomForest with `n_estimators=200`, `max_depth=5`

### Checking results in MLflow

In MLflow UI:
- Open experiment: `Titanic_Survival_Prediction`
- Select the 3 runs
- Click **Compare**
- Identify the best run based on **Accuracy** (and include your reasoning in the report)

---

## 9. Retry Demonstration (Intentional failure)

To show Airflow retry behavior for validation:
1. Temporarily modify the dataset so the `Age` column has missing values above **30%**
2. Trigger the DAG
3. `data_validation` should fail and show retries (state: `up_for_retry`) in Airflow

After testing, revert the dataset back to its original state.

---

## 10. Troubleshooting

### DAG is not visible in Airflow
Check these first:
- `AIRFLOW_HOME` is set correctly:
  ```bash
  export AIRFLOW_HOME="$PWD/airflow"
  ```
- DAG file is inside:
  ```text
  airflow/dags/
  ```
- Quick syntax check:
  ```bash
  python airflow/dags/mlops_airflow_mlflow_pipeline.py
  ```

### Ports already in use
Use different ports if needed:
- Airflow:
  ```bash
  airflow webserver --port 8081
  ```
- MLflow:
  ```bash
  mlflow ui --port 5001
  ```

---

## License

Academic / learning use only.
