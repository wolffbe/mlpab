import os
from google.cloud import bigquery

project = os.environ["GCP_PROJECT"]
dataset = os.environ["GCP_BQ_DATASET"]
loc = os.environ["GCP_LOCATION"]
client = bigquery.Client(project=project, location=loc)


def load(csv_path, table, schema):
    tid = f"{project}.{dataset}.{table}"
    job_config = bigquery.LoadJobConfig(
        source_format=bigquery.SourceFormat.CSV,
        skip_leading_rows=1,
        schema=schema,
        write_disposition="WRITE_TRUNCATE",
    )
    with open(csv_path, "rb") as f:
        job = client.load_table_from_file(f, tid, job_config=job_config)
    job.result()
    t = client.get_table(tid)
    print(f"loaded {table}: {t.num_rows} rows")


load("data/requests.csv", "requests_raw", [
    bigquery.SchemaField("request_id", "STRING"),
    bigquery.SchemaField("account_id", "STRING"),
    bigquery.SchemaField("request_lat", "FLOAT64"),
    bigquery.SchemaField("request_lon", "FLOAT64"),
    bigquery.SchemaField("requested_at", "TIMESTAMP"),
])
load("data/profiles.csv", "profiles_raw", [
    bigquery.SchemaField("account_id", "STRING"),
    bigquery.SchemaField("home_lat", "FLOAT64"),
    bigquery.SchemaField("home_lon", "FLOAT64"),
    bigquery.SchemaField("base_score", "FLOAT64"),
])
print("done")
