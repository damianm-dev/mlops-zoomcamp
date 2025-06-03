FROM apache/airflow:3.0.1
#USER root
#RUN apt-get update && apt-get install -y git && apt-get clean
USER airflow
COPY requirements.txt /requirements.txt
#RUN pip install --user -r /requirements.txt
RUN pip install xgboost mlflow==2.12.1
