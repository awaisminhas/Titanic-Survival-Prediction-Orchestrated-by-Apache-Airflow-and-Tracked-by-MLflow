"""
Pipeline: Titanic Survival Prediction 
Orchestrated by Apache Airflow and Tracked by MLflow
"""

# importing all libraries
import os
import json
import pickle
import warnings
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator, BranchPythonOperator
from airflow.operators.empty import EmptyOperator
import mlflow
import mlflow.sklearn
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
from sklearn.preprocessing import LabelEncoder

warnings.filterwarnings("ignore")

# 
# CONFIGURATION -These are the changes we will be making in 3 experiment runs
# ============================================================
# --- RUN 1 (default): LogisticRegression, C=1.0 ---
# --- RUN 2: Change to RandomForest below ---
# --- RUN 3: Change hyperparameters -- woth RandomForest

# first run
# MODEL_TYPE = "LogisticRegression"       # "LogisticRegression"
#2nd run
MODEL_TYPE = "RandomForest"       # "RandomForest"
HYPERPARAMS = {
    "LogisticRegression": {"C": 1.0, "max_iter": 200, "solver": "lbfgs"},
    "RandomForest": {"n_estimators": 200, "max_depth": 5, "random_state": 42},
}

BASE_DIR = os.path.expanduser("~/mlops_assignment")
DATA_PATH = os.path.join(BASE_DIR, "data", "Titanic-Dataset.csv")
PROCESSED_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
MLFLOW_TRACKING_URI = f"file://{os.path.join(BASE_DIR, 'mlruns')}"
EXPERIMENT_NAME = "Titanic_Survival_Prediction"
ACCURACY_THRESHOLD = 0.80

# Ensuring that directories exist
os.makedirs(PROCESSED_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

# ============================================================
# TASK FUNCTIONS
# ============================================================

# ---- TASK 2: Data Ingestion ----
def data_ingestion(**context):
    """Load Titanic CSV, print shape, log missing values, push path via XCom."""
    print(f"Loading dataset from: {DATA_PATH}")
    df = pd.read_csv(DATA_PATH)

    print(f"Dataset Shape: {df.shape}")
    print(f"Rows: {df.shape[0]}, Columns: {df.shape[1]}")

    missing = df.isnull().sum()
    print("\n--- Missing Values Count ---")
    print(missing[missing > 0].to_string())
    print(f"\nTotal missing values: {df.isnull().sum().sum()}")

    # Save raw data path and push via XCom
    context['ti'].xcom_push(key='dataset_path', value=DATA_PATH)
    context['ti'].xcom_push(key='dataset_shape', value=str(df.shape))
    print(f"\nXCom pushed dataset_path: {DATA_PATH}")
    return DATA_PATH


# ---- TASK 3: Data Validation ----
def data_validation(**context):
    """Validate missing percentages in Age and Embarked. Raise if > 30%."""
    dataset_path = context['ti'].xcom_pull(task_ids='data_ingestion', key='dataset_path')
    df = pd.read_csv(dataset_path)

    total_rows = len(df)
    age_missing_pct = (df['Age'].isnull().sum() / total_rows) * 100
    embarked_missing_pct = (df['Embarked'].isnull().sum() / total_rows) * 100

    print(f"Age missing: {age_missing_pct:.2f}%")
    print(f"Embarked missing: {embarked_missing_pct:.2f}%")

    if age_missing_pct > 30:
        raise ValueError(f"VALIDATION FAILED: Age has {age_missing_pct:.2f}% missing (> 30%)")
    if embarked_missing_pct > 30:
        raise ValueError(f"VALIDATION FAILED: Embarked has {embarked_missing_pct:.2f}% missing (> 30%)")

    print("Data validation PASSED.")
    context['ti'].xcom_push(key='validation_status', value='passed')


# ---- TASK 4a: Handle Missing Values ----
def handle_missing_values(**context):
    """Fill missing values in Age and Embarked."""
    dataset_path = context['ti'].xcom_pull(task_ids='data_ingestion', key='dataset_path')
    df = pd.read_csv(dataset_path)

    print(f"Before - Age nulls: {df['Age'].isnull().sum()}, Embarked nulls: {df['Embarked'].isnull().sum()}")

    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    print(f"After  - Age nulls: {df['Age'].isnull().sum()}, Embarked nulls: {df['Embarked'].isnull().sum()}")

    cleaned_path = os.path.join(PROCESSED_DIR, "titanic_cleaned.csv")
    df.to_csv(cleaned_path, index=False)
    context['ti'].xcom_push(key='cleaned_path', value=cleaned_path)
    print(f"Cleaned data saved to: {cleaned_path}")


# ---- TASK 4b: Feature Engineering ----
def feature_engineering(**context):
    """Create FamilySize and IsAlone features."""
    dataset_path = context['ti'].xcom_pull(task_ids='data_ingestion', key='dataset_path')
    df = pd.read_csv(dataset_path)

    df['FamilySize'] = df['SibSp'] + df['Parch'] + 1
    df['IsAlone'] = (df['FamilySize'] == 1).astype(int)

    print(f"FamilySize stats:\n{df['FamilySize'].describe()}")
    print(f"\nIsAlone distribution:\n{df['IsAlone'].value_counts().to_string()}")

    fe_path = os.path.join(PROCESSED_DIR, "titanic_features.csv")
    df.to_csv(fe_path, index=False)
    context['ti'].xcom_push(key='features_path', value=fe_path)
    print(f"Feature engineered data saved to: {fe_path}")


# ---- TASK 5: Data Encoding (Merge + Encode) ----
def data_encoding(**context):
    """Merge cleaned and feature-engineered data, encode categoricals, drop irrelevant columns."""
    cleaned_path = context['ti'].xcom_pull(task_ids='handle_missing_values', key='cleaned_path')
    features_path = context['ti'].xcom_pull(task_ids='feature_engineering', key='features_path')

    df_cleaned = pd.read_csv(cleaned_path)
    df_features = pd.read_csv(features_path)

    # Use cleaned data as base (it has no missing values)
    df = df_cleaned.copy()
    df['FamilySize'] = df_features['FamilySize']
    df['IsAlone'] = df_features['IsAlone']

    # Fill any remaining missing values
    df['Age'].fillna(df['Age'].median(), inplace=True)
    df['Embarked'].fillna(df['Embarked'].mode()[0], inplace=True)

    # Encode categorical variables
    le_sex = LabelEncoder()
    df['Sex'] = le_sex.fit_transform(df['Sex'])

    le_embarked = LabelEncoder()
    df['Embarked'] = le_embarked.fit_transform(df['Embarked'])

    print(f"Sex encoded: {dict(zip(le_sex.classes_, le_sex.transform(le_sex.classes_)))}")
    print(f"Embarked encoded: {dict(zip(le_embarked.classes_, le_embarked.transform(le_embarked.classes_)))}")

    # Drop irrelevant columns
    columns_to_drop = ['PassengerId', 'Name', 'Ticket', 'Cabin']
    df.drop(columns=columns_to_drop, inplace=True, errors='ignore')
    print(f"Dropped columns: {columns_to_drop}")
    print(f"Final columns: {list(df.columns)}")
    print(f"Final shape: {df.shape}")

    encoded_path = os.path.join(PROCESSED_DIR, "titanic_encoded.csv")
    df.to_csv(encoded_path, index=False)
    context['ti'].xcom_push(key='encoded_path', value=encoded_path)


# ---- TASK 6: Model Training with MLflow ----
def model_training(**context):
    """Train model with MLflow tracking."""
    encoded_path = context['ti'].xcom_pull(task_ids='data_encoding', key='encoded_path')
    df = pd.read_csv(encoded_path)

    X = df.drop('Survived', axis=1)
    y = df['Survived']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

    # Save test data for evaluation
    test_data = {"X_test_path": os.path.join(PROCESSED_DIR, "X_test.csv"),
                 "y_test_path": os.path.join(PROCESSED_DIR, "y_test.csv")}
    pd.DataFrame(X_test).to_csv(test_data["X_test_path"], index=False)
    pd.DataFrame(y_test).to_csv(test_data["y_test_path"], index=False)

    # Set MLflow tracking
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    params = HYPERPARAMS[MODEL_TYPE]

    with mlflow.start_run(run_name=f"{MODEL_TYPE}_run") as run:
        # Log model type and hyperparameters
        mlflow.log_param("model_type", MODEL_TYPE)
        for param_name, param_value in params.items():
            mlflow.log_param(param_name, param_value)

        mlflow.log_param("train_size", len(X_train))
        mlflow.log_param("test_size", len(X_test))
        mlflow.log_param("dataset_total_size", len(df))
        mlflow.log_param("num_features", X_train.shape[1])

        # Train model
        if MODEL_TYPE == "LogisticRegression":
            model = LogisticRegression(**params)
        else:
            model = RandomForestClassifier(**params)

        model.fit(X_train, y_train)
        print(f"Model trained: {MODEL_TYPE} with params: {params}")

        # Log model artifact
        mlflow.sklearn.log_model(model, "model")

        # Log dataset as artifact
        mlflow.log_artifact(encoded_path)

        # Save model locally too
        model_path = os.path.join(MODELS_DIR, "trained_model.pkl")
        with open(model_path, 'wb') as f:
            pickle.dump(model, f)

        # Push info via XCom
        context['ti'].xcom_push(key='model_path', value=model_path)
        context['ti'].xcom_push(key='run_id', value=run.info.run_id)
        context['ti'].xcom_push(key='X_test_path', value=test_data["X_test_path"])
        context['ti'].xcom_push(key='y_test_path', value=test_data["y_test_path"])
        context['ti'].xcom_push(key='model_type', value=MODEL_TYPE)

        print(f"MLflow Run ID: {run.info.run_id}")


# ---- TASK 7: Model Evaluation ----
def model_evaluation(**context):
    """Evaluate model and log metrics to MLflow."""
    model_path = context['ti'].xcom_pull(task_ids='model_training', key='model_path')
    X_test_path = context['ti'].xcom_pull(task_ids='model_training', key='X_test_path')
    y_test_path = context['ti'].xcom_pull(task_ids='model_training', key='y_test_path')
    run_id = context['ti'].xcom_pull(task_ids='model_training', key='run_id')

    # Load model and test data
    with open(model_path, 'rb') as f:
        model = pickle.load(f)

    X_test = pd.read_csv(X_test_path)
    y_test = pd.read_csv(y_test_path).values.ravel()

    # Predictions
    y_pred = model.predict(X_test)

    # Compute metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')

    print(f"\n{'='*50}")
    print(f"MODEL EVALUATION RESULTS")
    print(f"{'='*50}")
    print(f"Accuracy:  {accuracy:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall:    {recall:.4f}")
    print(f"F1-Score:  {f1:.4f}")
    print(f"{'='*50}\n")

    # Log metrics to MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    with mlflow.start_run(run_id=run_id):
        mlflow.log_metric("accuracy", accuracy)
        mlflow.log_metric("precision", precision)
        mlflow.log_metric("recall", recall)
        mlflow.log_metric("f1_score", f1)

    # Push accuracy via XCom for branching
    context['ti'].xcom_push(key='accuracy', value=accuracy)
    context['ti'].xcom_push(key='run_id', value=run_id)
    print(f"Accuracy pushed to XCom: {accuracy}")


# ---- TASK 8: Branching Logic ----
def check_accuracy(**context):
    """Branch based on accuracy threshold."""
    accuracy = context['ti'].xcom_pull(task_ids='model_evaluation', key='accuracy')
    print(f"Retrieved accuracy: {accuracy}")
    print(f"Threshold: {ACCURACY_THRESHOLD}")

    if accuracy >= ACCURACY_THRESHOLD:
        print(f"Accuracy {accuracy:.4f} >= {ACCURACY_THRESHOLD} → REGISTERING MODEL")
        return 'register_model'
    else:
        print(f"Accuracy {accuracy:.4f} < {ACCURACY_THRESHOLD} → REJECTING MODEL")
        return 'reject_model'


# ---- TASK 9a: Register Model ----
def register_model(**context):
    """Register model in MLflow Model Registry."""
    run_id = context['ti'].xcom_pull(task_ids='model_evaluation', key='run_id')
    accuracy = context['ti'].xcom_pull(task_ids='model_evaluation', key='accuracy')
    model_type = context['ti'].xcom_pull(task_ids='model_training', key='model_type')

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    model_uri = f"runs:/{run_id}/model"
    model_name = "TitanicSurvivalModel"

    result = mlflow.register_model(model_uri=model_uri, name=model_name)

    print(f"\n{'='*50}")
    print(f"MODEL REGISTERED SUCCESSFULLY")
    print(f"{'='*50}")
    print(f"Model Name:    {result.name}")
    print(f"Model Version: {result.version}")
    print(f"Model Type:    {model_type}")
    print(f"Accuracy:      {accuracy:.4f}")
    print(f"Run ID:        {run_id}")
    print(f"{'='*50}\n")

    # Log registration info
    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("registered", True)
        mlflow.log_param("registry_version", result.version)


# ---- TASK 9b: Reject Model ----
def reject_model(**context):
    """Log rejection reason to MLflow."""
    run_id = context['ti'].xcom_pull(task_ids='model_evaluation', key='run_id')
    accuracy = context['ti'].xcom_pull(task_ids='model_evaluation', key='accuracy')

    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)

    rejection_reason = (
        f"Model REJECTED. Accuracy {accuracy:.4f} is below threshold {ACCURACY_THRESHOLD}. "
        f"Model does not meet minimum quality standards for deployment."
    )

    print(f"\n{'='*50}")
    print(f"MODEL REJECTED")
    print(f"{'='*50}")
    print(f"Reason: {rejection_reason}")
    print(f"{'='*50}\n")

    with mlflow.start_run(run_id=run_id):
        mlflow.log_param("registered", False)
        mlflow.log_param("rejection_reason", rejection_reason)


# ============================================================
# DAG DEFINITION
# ============================================================

default_args = {
    'owner': 'mlops_student',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(seconds=10),
    'start_date': datetime(2025, 1, 1),
}

with DAG(
    dag_id='mlops_airflow_mlflow_pipeline',
    default_args=default_args,
    description='End-to-end MLOps pipeline: Titanic Survival Prediction with Airflow + MLflow',
    schedule_interval=None,
    catchup=False,
    tags=['mlops', 'mlflow', 'titanic'],
) as dag:

    # Task 2: Data Ingestion
    t_ingest = PythonOperator(
        task_id='data_ingestion',
        python_callable=data_ingestion,
    )

    # Task 3: Data Validation (with retries to demonstrate retry behavior)
    t_validate = PythonOperator(
        task_id='data_validation',
        python_callable=data_validation,
        retries=3,
        retry_delay=timedelta(seconds=5),
    )

    # Task 4a: Handle Missing Values (PARALLEL)
    t_missing = PythonOperator(
        task_id='handle_missing_values',
        python_callable=handle_missing_values,
    )

    # Task 4b: Feature Engineering (PARALLEL)
    t_features = PythonOperator(
        task_id='feature_engineering',
        python_callable=feature_engineering,
    )

    # Task 5: Data Encoding
    t_encode = PythonOperator(
        task_id='data_encoding',
        python_callable=data_encoding,
    )

    # Task 6: Model Training
    t_train = PythonOperator(
        task_id='model_training',
        python_callable=model_training,
    )

    # Task 7: Model Evaluation
    t_evaluate = PythonOperator(
        task_id='model_evaluation',
        python_callable=model_evaluation,
    )

    # Task 8: Branching Logic
    t_branch = BranchPythonOperator(
        task_id='check_accuracy',
        python_callable=check_accuracy,
    )

    # Task 9a: Register Model
    t_register = PythonOperator(
        task_id='register_model',
        python_callable=register_model,
    )

    # Task 9b: Reject Model
    t_reject = PythonOperator(
        task_id='reject_model',
        python_callable=reject_model,
    )

    # End node
    t_end = EmptyOperator(
        task_id='end',
        trigger_rule='none_failed_min_one_success',
    )

    t_ingest >> t_validate >> [t_missing, t_features]
    [t_missing, t_features] >> t_encode
    t_encode >> t_train >> t_evaluate >> t_branch
    t_branch >> [t_register, t_reject] >> t_end
