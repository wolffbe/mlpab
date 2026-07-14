import os, json
from google.cloud import bigquery

PROJECT = os.environ["GCP_PROJECT"]
DATASET = os.environ["GCP_BQ_DATASET"]
LOCATION = os.environ["GCP_LOCATION"]

bq = bigquery.Client(project=PROJECT, location=LOCATION)

raw = f"{PROJECT}.{DATASET}.eventsd3c188_raw"
final = f"{PROJECT}.{DATASET}.eventsd3c188"

# Load CSV as all-STRING to inspect/validate ourselves in SQL
schema = [
    bigquery.SchemaField("row_id", "STRING"),
    bigquery.SchemaField("account_id", "STRING"),
    bigquery.SchemaField("event_time", "STRING"),
    bigquery.SchemaField("amount", "STRING"),
    bigquery.SchemaField("category", "STRING"),
]
job_config = bigquery.LoadJobConfig(
    schema=schema,
    skip_leading_rows=1,
    source_format=bigquery.SourceFormat.CSV,
    write_disposition="WRITE_TRUNCATE",
    allow_quoted_newlines=True,
)
with open("data/events.csv", "rb") as f:
    load_job = bq.load_table_from_file(f, raw, job_config=job_config)
load_job.result()
print("raw loaded rows:", bq.get_table(raw).num_rows)

# Contract validity predicate
valid_pred = """
  amount IS NOT NULL AND TRIM(amount) != ''
  AND SAFE_CAST(amount AS FLOAT64) IS NOT NULL
  AND SAFE_CAST(amount AS FLOAT64) BETWEEN 0 AND 10000
  AND category IN ('grocery','travel','salary','rent','other')
"""

# Build final typed feature table with ONLY valid rows
create_sql = f"""
CREATE OR REPLACE TABLE `{final}` AS
SELECT
  row_id,
  account_id,
  SAFE_CAST(event_time AS INT64) AS event_time,
  SAFE_CAST(amount AS FLOAT64) AS amount,
  category,
  TIMESTAMP_MILLIS(SAFE_CAST(event_time AS INT64)) AS feature_timestamp
FROM `{raw}`
WHERE {valid_pred}
"""
bq.query(create_sql, location=LOCATION).result()
n_final = bq.get_table(final).num_rows
print("final valid rows:", n_final)

# Rejected ids
rej_sql = f"SELECT row_id FROM `{raw}` WHERE NOT ({valid_pred}) OR ({valid_pred}) IS NULL ORDER BY row_id"
rejected = [r["row_id"] for r in bq.query(rej_sql, location=LOCATION).result()]
print("rejected count:", len(rejected))

os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump({"rejected": rejected}, f, indent=2)
print("wrote submission/answers.json")
print("sample rejected:", rejected[:10])
