import os
import json
from google.cloud import bigquery

PROJECT = os.environ["GCP_PROJECT"]
LOCATION = os.environ["GCP_LOCATION"]
DATASET = os.environ["GCP_BQ_DATASET"]
T = 1773478800000

with open("data/model.json") as f:
    model = json.load(f)
w = model["weights"]
b = model["bias"]
print("model:", w, b)

client = bigquery.Client(project=PROJECT, location=LOCATION)

raw_table = f"{PROJECT}.{DATASET}.feature_history_raw"
scores_table = f"{PROJECT}.{DATASET}.scores36e30a"

# 1. Load feature history CSV into BigQuery (ingestion onto the platform)
schema = [
    bigquery.SchemaField("account_id", "STRING"),
    bigquery.SchemaField("event_time", "INT64"),
    bigquery.SchemaField("f1", "FLOAT64"),
    bigquery.SchemaField("f2", "FLOAT64"),
    bigquery.SchemaField("f3", "FLOAT64"),
]
job_config = bigquery.LoadJobConfig(
    schema=schema,
    skip_leading_rows=1,
    source_format=bigquery.SourceFormat.CSV,
    write_disposition="WRITE_TRUNCATE",
)
with open("data/feature_history.csv", "rb") as fh:
    load_job = client.load_table_from_file(fh, raw_table, job_config=job_config)
load_job.result()
n_raw = client.get_table(raw_table).num_rows
print("loaded raw rows:", n_raw)

# 2. Point-in-time-correct scoring, entirely in BigQuery.
#    sigmoid(z) = 1/(1+exp(-z)); pick most recent revision at or before T.
query = f"""
CREATE OR REPLACE TABLE `{scores_table}` AS
WITH valid AS (
  SELECT account_id, f1, f2, f3,
         ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY event_time DESC) AS rn
  FROM `{raw_table}`
  WHERE event_time <= {T}
)
SELECT
  account_id,
  ROUND(1.0 / (1.0 + EXP(-(({w['f1']}) * f1 + ({w['f2']}) * f2 + ({w['f3']}) * f3 + ({b})))), 6) AS score
FROM valid
WHERE rn = 1
"""
client.query(query, location=LOCATION).result()

# 3. Validate
n_scores = client.get_table(scores_table).num_rows
n_accounts = list(client.query(
    f"SELECT COUNT(DISTINCT account_id) c FROM `{raw_table}`", location=LOCATION).result())[0].c
n_valid_accounts = list(client.query(
    f"SELECT COUNT(DISTINCT account_id) c FROM `{raw_table}` WHERE event_time <= {T}",
    location=LOCATION).result())[0].c
print("distinct accounts total:", n_accounts, "| with revision<=T:", n_valid_accounts, "| scored rows:", n_scores)
for r in client.query(f"SELECT * FROM `{scores_table}` ORDER BY account_id LIMIT 5", location=LOCATION).result():
    print(r.account_id, r.score)
