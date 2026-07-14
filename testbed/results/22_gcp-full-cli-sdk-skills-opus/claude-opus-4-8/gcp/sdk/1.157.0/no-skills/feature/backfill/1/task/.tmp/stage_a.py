import os
from google.cloud import bigquery

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
DATASET = os.environ["GCP_BQ_DATASET"]

client = bigquery.Client(project=PROJECT)

ds = client.get_dataset(f"{PROJECT}.{DATASET}")
print("dataset location:", ds.location)

staging = f"{PROJECT}.{DATASET}.accountsed4daa_staging"
final = f"{PROJECT}.{DATASET}.accountsed4daa"

schema = [
    bigquery.SchemaField("row_id", "STRING"),
    bigquery.SchemaField("status", "STRING"),
    bigquery.SchemaField("balance", "FLOAT64"),
    bigquery.SchemaField("updated_at", "INT64"),
]

# Fresh staging table each run
client.query(f"DROP TABLE IF EXISTS `{staging}`", location=ds.location).result()

for i, path in enumerate(["data/batch_1.csv", "data/batch_2.csv", "data/batch_3.csv"]):
    job_config = bigquery.LoadJobConfig(
        schema=schema,
        skip_leading_rows=1,
        source_format=bigquery.SourceFormat.CSV,
        write_disposition=bigquery.WriteDisposition.WRITE_APPEND if i > 0 else bigquery.WriteDisposition.WRITE_TRUNCATE,
    )
    with open(path, "rb") as f:
        job = client.load_table_from_file(f, staging, job_config=job_config, location=ds.location)
    job.result()
    print("loaded", path, "->", job.output_rows, "rows")

# Dedup to latest revision per row_id, entirely in BigQuery.
dedup_sql = f"""
CREATE OR REPLACE TABLE `{final}` AS
SELECT row_id, status, balance, updated_at
FROM (
  SELECT row_id, status, balance, updated_at,
         ROW_NUMBER() OVER (PARTITION BY row_id ORDER BY updated_at DESC) AS rn
  FROM `{staging}`
)
WHERE rn = 1
"""
client.query(dedup_sql, location=ds.location).result()

# Verify
counts = list(client.query(
    f"SELECT COUNT(*) AS n, COUNT(DISTINCT row_id) AS d FROM `{final}`",
    location=ds.location).result())[0]
staged = list(client.query(
    f"SELECT COUNT(*) AS n, COUNT(DISTINCT row_id) AS d FROM `{staging}`",
    location=ds.location).result())[0]
print(f"staging rows={staged.n} distinct_row_id={staged.d}")
print(f"final rows={counts.n} distinct_row_id={counts.d}")

# Spot check a row that appears in multiple batches keeps the max updated_at
sample = list(client.query(f"""
SELECT s.row_id, MAX(s.updated_at) AS max_ts, f.updated_at AS final_ts
FROM `{staging}` s JOIN `{final}` f USING(row_id)
GROUP BY s.row_id, f.updated_at
HAVING COUNT(*) OVER () >= 0
ORDER BY s.row_id LIMIT 5
""", location=ds.location).result())
for r in sample:
    print("check", r.row_id, "max_ts", r.max_ts, "final_ts", r.final_ts, "OK" if r.max_ts == r.final_ts else "MISMATCH")

# How many row_ids had >1 revision
multi = list(client.query(f"""
SELECT COUNTIF(c>1) AS dup_keys, MAX(c) AS max_rev FROM (
  SELECT row_id, COUNT(*) c FROM `{staging}` GROUP BY row_id)
""", location=ds.location).result())[0]
print(f"row_ids with >1 revision: {multi.dup_keys}, max revisions for a key: {multi.max_rev}")
print("STAGE_A_DONE")
