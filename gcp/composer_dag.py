"""
Cloud Composer (Airflow) DAG — orchestrates the lakehouse pipeline:
  medallion_etl (Dataproc Serverless Spark)  ->  refresh_gold_table (BigQuery/BigLake)

Deploy by copying this file into the Composer environment's dags/ GCS folder. The PySpark
job (gcp/dataproc_medallion.py) must be uploaded to gs://<bucket>/code/ first.

Config via Airflow env vars: GCP_PROJECT, GCP_REGION, SNN_BUCKET.
"""
import datetime
import os

from airflow import DAG
from airflow.providers.google.cloud.operators.dataproc import DataprocCreateBatchOperator
from airflow.providers.google.cloud.operators.bigquery import BigQueryInsertJobOperator

PROJECT = os.environ.get("GCP_PROJECT", "snn-data-lake-prod")
REGION = os.environ.get("GCP_REGION", "us-central1")
BUCKET = os.environ.get("SNN_BUCKET", f"{PROJECT}-snn-lake")
CONNECTION = f"{REGION}.biglake-snn"

REFRESH_GOLD_SQL = f"""
CREATE OR REPLACE EXTERNAL TABLE `{PROJECT}.snn_lake.gold`
WITH CONNECTION `{CONNECTION}`
OPTIONS (format = 'PARQUET', uris = ['gs://{BUCKET}/gold/*']);
"""

with DAG(
    dag_id="snn_medallion",
    schedule="@daily",
    start_date=datetime.datetime(2026, 1, 1),
    catchup=False,
    default_args={"retries": 1, "retry_delay": datetime.timedelta(minutes=5)},
    tags=["snn", "lakehouse"],
) as dag:

    medallion_etl = DataprocCreateBatchOperator(
        task_id="medallion_etl",
        project_id=PROJECT,
        region=REGION,
        batch_id="medallion-{{ ds_nodash }}",
        batch={
            "pyspark_batch": {
                "main_python_file_uri": f"gs://{BUCKET}/code/dataproc_medallion.py",
                "args": [BUCKET],
            },
            "runtime_config": {"version": "2.2"},
        },
    )

    refresh_gold_table = BigQueryInsertJobOperator(
        task_id="refresh_gold_table",
        location=REGION,
        configuration={"query": {"query": REFRESH_GOLD_SQL, "useLegacySql": False}},
    )

    medallion_etl >> refresh_gold_table
