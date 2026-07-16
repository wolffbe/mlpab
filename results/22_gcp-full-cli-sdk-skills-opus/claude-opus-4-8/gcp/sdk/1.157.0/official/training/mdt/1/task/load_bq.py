import os
from google.cloud import bigquery

PROJECT = os.environ["GCP_PROJECT"]
DATASET = os.environ["GCP_BQ_DATASET"]
LOCATION = os.environ["GCP_LOCATION"]

bq = bigquery.Client(project=PROJECT, location=LOCATION)
ds = f"{PROJECT}.{DATASET}"

schema = [
    bigquery.SchemaField("row_id", "STRING"),
    bigquery.SchemaField("f1", "FLOAT64"),
    bigquery.SchemaField("f2", "FLOAT64"),
    bigquery.SchemaField("f3", "FLOAT64"),
    bigquery.SchemaField("f4", "FLOAT64"),
]


def load(csv_path, table):
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        skip_leading_rows=1,
        source_format=bigquery.SourceFormat.CSV,
        write_disposition="WRITE_TRUNCATE",
    )
    with open(csv_path, "rb") as f:
        job = bq.load_table_from_file(f, f"{ds}.{table}", job_config=job_config)
    job.result()
    t = bq.get_table(f"{ds}.{table}")
    print(f"loaded {table}: {t.num_rows} rows")


load("data/features_train.csv", "features_train")
load("data/features_serve.csv", "features_serve")
