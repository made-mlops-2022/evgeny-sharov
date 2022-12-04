import json
import pathlib
import pandas as pd
import random
from sklearn.datasets import load_breast_cancer

import airflow
import requests
import requests.exceptions as requests_exceptions
from airflow import DAG
from airflow.operators.bash import BashOperator
from airflow.operators.python import PythonOperator

SHAPE = 100


def _generate_data(path_to_data: str) -> None:
    """Generate random data that has same values like train data
    Function will create file /data/raw/data.csv with test data and
    file /data/raw/target.csv with target data."""
    pathlib.Path(path_to_data).mkdir(parents=True, exist_ok=True)

    path_to_target = pathlib.Path(path_to_data) / "target.csv"
    path_to_data = pathlib.Path(path_to_data) / "data.csv"

    if pathlib.Path(path_to_data).exists():
        old_data = pd.read_csv(path_to_data)
    else:
        old_data = pd.DataFrame()

    if pathlib.Path(path_to_target).exists():
        old_target = pd.read_csv(path_to_target)
    else:
        old_target = pd.DataFrame()

    data, target = load_breast_cancer(return_X_y=True, as_frame=True)
    data = data.sample(SHAPE)
    target = pd.DataFrame(target.loc[data.index], columns=['target'])
    generated_data = {}
    for column_name in data:
        generated_data[column_name] = random.choices(data[column_name].unique(), k=SHAPE)

    data = pd.concat([old_data, pd.DataFrame(generated_data)], axis=0, ignore_index=True)
    data.to_csv(path_to_data, index=False)

    target = pd.concat([old_target, target], axis=0, ignore_index=True)
    target.to_csv(path_to_target, index=False)


with DAG(
    dag_id="data_generation",
    start_date=airflow.utils.dates.days_ago(4),
    schedule_interval="@hourly",
    max_active_runs=1,
) as dag:
    load_data = PythonOperator(
        task_id="generate_data",
        python_callable=_generate_data,
        op_args=("data/raw/{{ ds }}",)
    )

    notify = BashOperator(task_id="notification", bash_command="echo received data at {{ ds }}")

    load_data >> notify
