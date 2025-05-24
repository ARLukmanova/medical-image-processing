FROM apache/airflow:2.9.3-python3.10

#WORKDIR /app

COPY requirements.txt /app/requirements.txt

USER root
RUN apt-get update && apt-get install -y git
USER airflow

RUN pip install --upgrade pip
RUN pip install --no-cache-dir -r /app/requirements.txt