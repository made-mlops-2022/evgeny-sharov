import json
import pathlib
import pandas as pd
import numpy as np
import random
from sklearn.preprocessing import Normalizer
from sklearn.linear_model import LogisticRegression
import pickle

import airflow
import requests
import requests.exceptions as requests_exceptions
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator
from airflow.models import Variable


def _predict_data(path_to_data: str, path_to_prediction: str):
    path_to_models = f"data/models/{Variable.get('model_name')}"
    path_to_normalizer = pathlib.Path(path_to_models) / "normalizer.pickle"
    path_to_model = pathlib.Path(path_to_models) / "logreg.pickle"

    pathlib.Path(path_to_prediction).mkdir(parents=True, exist_ok=True)
    path_to_prediction = pathlib.Path(path_to_prediction) / "prediction.csv"
    path_to_data = pathlib.Path(path_to_data) / "data.csv"
    if not path_to_data.exists():
        return None
    data = pd.read_csv(path_to_data)

    with open(path_to_normalizer, 'rb') as file:
        normalizer = pickle.load(file)

    with open(path_to_model, 'rb') as file:
        model = pickle.load(file)

    normed_data = normalizer.transform(data)
    prediction = pd.DataFrame(model.predict(normed_data), columns=['prediction'], index=data.index)

    prediction.to_csv(path_to_prediction, index=False)


with DAG(
    dag_id="predict_data",
    start_date=airflow.utils.dates.days_ago(2),
    schedule_interval="@daily",
    max_active_runs=1,
) as dag:
    predict_data = PythonOperator(
        task_id="predict_data",
        python_callable=_predict_data,
        op_args=("data/raw/{{ ds }}", "data/prediction/{{ ds }}")
    )

    predict_data
