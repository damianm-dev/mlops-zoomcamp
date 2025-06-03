# LABELS: dag, airflow (it's required for airflow dag-processor)

# from __future__ import annotations
import datetime


from airflow.sdk import dag, task
from duration import run


@dag(start_date=datetime.datetime(2025, 6, 2), schedule="@daily")
def test_dag_mlflow():
    @task
    def my_task():
        run(year=2023, month=3)

    my_task()


test_dag_mlflow()
