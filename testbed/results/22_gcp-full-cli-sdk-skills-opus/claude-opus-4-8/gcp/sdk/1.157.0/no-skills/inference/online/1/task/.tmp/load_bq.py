import os
from google.cloud import bigquery

PROJECT = os.environ["GCP_PROJECT"]
DATASET = os.environ["GCP_BQ_DATASET"]
client = bigquery.Client(project=PROJECT)

raw = f"{PROJECT}.{DATASET}.profilesaf22bf_raw"
schema = [
    bigquery.SchemaField("account_id", "STRING"),
    bigquery.SchemaField("f1", "FLOAT64"),
    bigquery.SchemaField("f2", "FLOAT64"),
    bigquery.SchemaField("f3", "FLOAT64"),
    bigquery.SchemaField("f4", "FLOAT64"),
]
job_config = bigquery.LoadJobConfig(
    schema=schema, skip_leading_rows=1,
    source_format=bigquery.SourceFormat.CSV, write_disposition="WRITE_TRUNCATE",
)
with open("data/features.csv", "rb") as f:
    job = client.load_table_from_file(f, raw, job_config=job_config)
job.result()
print("loaded raw rows:", client.get_table(raw).num_rows)

final = f"{PROJECT}.{DATASET}.profilesaf22bf"
q = f"""CREATE OR REPLACE TABLE `{final}` AS
SELECT account_id, f1, f2, f3, f4, CURRENT_TIMESTAMP() AS feature_timestamp
FROM `{raw}`"""
client.query(q).result()
t = client.get_table(final)
print("final table rows:", t.num_rows, "schema:", [(s.name, s.field_type) for s in t.schema])
