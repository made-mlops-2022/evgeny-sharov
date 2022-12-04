import json
import pathlib
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.metrics import f1_score, accuracy_score
import pickle

import airflow
import requests
import requests.exceptions as requests_exceptions
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

TRAIN_FILE_NAME = "train_data.csv"


def _prepare_data(path_to_data: str, path_to_preprocessed: str, path_to_models: str):
    """Function reads data for current day, normalizes it and saves data for training"""
    pathlib.Path(path_to_models).mkdir(parents=True, exist_ok=True)
    pathlib.Path(path_to_preprocessed).mkdir(parents=True, exist_ok=True)

    path_to_target = pathlib.Path(path_to_data) / "target.csv"
    path_to_data = pathlib.Path(path_to_data) / "data.csv"
    path_to_preprocessed = pathlib.Path(path_to_preprocessed) / TRAIN_FILE_NAME
    path_to_model = pathlib.Path(path_to_models) / "normalizer.pickle"

    data = pd.read_csv(path_to_data)
    target = pd.read_csv(path_to_target)

    normalizer = Normalizer()
    normed_data = normalizer.fit_transform(data)
    normed_data = pd.DataFrame(normed_data, columns=data.columns)

    with open(path_to_model, 'wb') as file:
        pickle.dump(normalizer, file)

    normed_data = pd.concat([normed_data, target], axis=1)
    normed_data.to_csv(path_to_preprocessed, index=False)


def _split_train_val(path_to_preprocessed: str, path_to_train_val: str):
    """
    Split preprocessed data randomly to training and validation datasets.
    Write them to directory path_to_train_val.
    """
    pathlib.Path(path_to_train_val).mkdir(parents=True, exist_ok=True)

    path_to_preprocessed = pathlib.Path(path_to_preprocessed) / TRAIN_FILE_NAME
    data = pd.read_csv(path_to_preprocessed)

    train, val = train_test_split(data, test_size=0.2)

    path_to_train = pathlib.Path(path_to_train_val) / "train_data.csv"
    path_to_val = pathlib.Path(path_to_train_val) / "val_data.csv"

    train.to_csv(path_to_train, index=False)
    val.to_csv(path_to_val, index=False)


def _train_model(path_to_train_val: str, path_to_models: str):
    """
    Trains Logistic Regression on training data and save model to path_to_model
    """
    pathlib.Path(path_to_models).mkdir(parents=True, exist_ok=True)
    path_to_model = pathlib.Path(path_to_models) / "logreg.pickle"

    path_to_train = pathlib.Path(path_to_train_val) / "train_data.csv"
    train = pd.read_csv(path_to_train)
    target = train['target'].copy()
    train.drop(['target'], axis=1, inplace=True)

    # TODO: save param grid in config file
    parameter_grid = [
        {'penalty': ['l2', 'l1'], 'C': np.arange(0.1, 1, 0.01),
         'solver': ['saga'], 'max_iter': [1000]},
        {'penalty': ['elasticnet'], 'l1_ratio': np.arange(0, 1, 0.01),
         'solver': ['saga'], 'max_iter': [1000]}
    ]
    model = LogisticRegression()
    grid_model = GridSearchCV(model, parameter_grid, refit=True)
    grid_model.fit(train, target)

    with open(path_to_model, 'wb') as file:
        pickle.dump(grid_model.best_estimator_, file)


def _validate_model(path_to_train_val: str, path_to_models: str, path_to_metrics: str):
    """
    Calculates f1 score and accuracy for model in path_to_models,
    writes calculated metrics to path_to_metrics.
    """
    pathlib.Path(path_to_metrics).mkdir(parents=True, exist_ok=True)
    path_to_metrics = pathlib.Path(path_to_metrics) / "metrics.json"

    path_to_model = pathlib.Path(path_to_models) / "logreg.pickle"
    path_to_val = pathlib.Path(path_to_train_val) / "val_data.csv"

    validation_data = pd.read_csv(path_to_val)
    target = validation_data['target'].copy()
    validation_data.drop(['target'], axis=1, inplace=True)

    with open(path_to_model, 'rb') as file:
        model = pickle.load(file)

    prediction = model.predict(validation_data)
    metrics = {
        "accuracy_score": accuracy_score(target, prediction),
        "f1_score": f1_score(target, prediction)
    }
    with open(path_to_metrics, 'w', encoding='utf-8') as metric_file:
        json.dump(metrics, metric_file, ensure_ascii=False, indent=4)


with DAG(
        dag_id="model_training",
        start_date=airflow.utils.dates.days_ago(2),
        schedule="59 23 * * 6",
        catchup=True
) as dag:
    preprocess_data = PythonOperator(
        task_id="preprocess_data",
        python_callable=_prepare_data,
        op_args=("data/raw/{{ ds }}", "data/preprocessed/{{ ds }}", "data/models/{{ ds }}")
    )

    split_train_test = PythonOperator(
        task_id="split_train_test",
        python_callable=_split_train_val,
        op_args=("data/preprocessed/{{ ds }}", "data/train_val/{{ ds }}")
    )

    train_model = PythonOperator(
        task_id="train_model",
        python_callable=_train_model,
        op_args=("data/train_val/{{ ds }}", "data/models/{{ ds }}")
    )

    validate_model = PythonOperator(
        task_id="validate_model",
        python_callable=_validate_model,
        op_args=("data/train_val/{{ ds }}", "data/models/{{ ds }}", "data/metrics/{{ ds }}")
    )

    preprocess_data >> split_train_test >> train_model >> validate_model
