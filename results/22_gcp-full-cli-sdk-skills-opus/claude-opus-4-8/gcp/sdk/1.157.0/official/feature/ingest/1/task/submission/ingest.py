import os
from google.cloud import bigquery

PROJECT = os.environ["GCP_PROJECT"]
DS = os.environ["GCP_BQ_DATASET"]
client = bigquery.Client(project=PROJECT)
dsref = f"{PROJECT}.{DS}"
print("dataset location:", client.get_dataset(dsref).location)

schema = [
    bigquery.SchemaField("row_id", "STRING"),
    bigquery.SchemaField("account_id", "STRING"),
    bigquery.SchemaField("event_time", "INT64"),
    bigquery.SchemaField("amount", "FLOAT64"),
    bigquery.SchemaField("category", "STRING"),
]
stg = f"{dsref}.transactions85a07a_staging"
job_cfg = bigquery.LoadJobConfig(
    schema=schema, skip_leading_rows=1,
    source_format=bigquery.SourceFormat.CSV,
    write_disposition="WRITE_TRUNCATE",
)
with open("data/transactions_export_1.csv", "rb") as f:
    client.load_table_from_file(f, stg, job_config=job_cfg).result()
job_cfg.write_disposition = "WRITE_APPEND"
with open("data/transactions_export_2.csv", "rb") as f:
    client.load_table_from_file(f, stg, job_config=job_cfg).result()
print("staging rows:", client.get_table(stg).num_rows)

final = f"{dsref}.transactions85a07a"
sql = f"""CREATE OR REPLACE TABLE `{final}` AS
SELECT row_id, account_id, event_time, amount, category FROM (
  SELECT *, ROW_NUMBER() OVER(PARTITION BY row_id ORDER BY event_time) rn
  FROM `{stg}`
) WHERE rn = 1"""
client.query(sql).result()
t = client.get_table(final)
print("FINAL rows:", t.num_rows, "cols:", [f.name for f in t.schema])
r = list(client.query(
    f"SELECT COUNT(*) c, COUNT(DISTINCT row_id) d FROM `{final}`").result())[0]
print("count/distinct:", r.c, r.d)
