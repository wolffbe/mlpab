import os
from google.cloud import bigquery

PROJECT = os.environ["GCP_PROJECT"]
DATASET = os.environ["GCP_BQ_DATASET"]
T = 1773478800000
W1, W2, W3, BIAS = -0.4331, -0.4499, 1.9204, -0.0086

client = bigquery.Client(project=PROJECT)

raw_tbl = f"{PROJECT}.{DATASET}.feature_history_raw"
scores_tbl = f"{PROJECT}.{DATASET}.scores36e30a"

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
with open("data/feature_history.csv", "rb") as f:
    load_job = client.load_table_from_file(f, raw_tbl, job_config=job_config)
load_job.result()
print("loaded raw rows:", client.get_table(raw_tbl).num_rows)

# Point-in-time-correct scoring done entirely in BigQuery.
sql = f"""
CREATE OR REPLACE TABLE `{scores_tbl}` AS
WITH valid AS (
  SELECT account_id, f1, f2, f3
  FROM (
    SELECT account_id, f1, f2, f3,
           ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY event_time DESC) AS rn
    FROM `{raw_tbl}`
    WHERE event_time <= {T}
  )
  WHERE rn = 1
)
SELECT
  account_id,
  ROUND(1.0 / (1.0 + EXP(-({W1}*f1 + {W2}*f2 + {W3}*f3 + ({BIAS})))), 6) AS score
FROM valid
"""
client.query(sql).result()
tbl = client.get_table(scores_tbl)
print("scores rows:", tbl.num_rows)
print("columns:", [f.name for f in tbl.schema])
for row in client.query(f"SELECT account_id, score FROM `{scores_tbl}` ORDER BY account_id LIMIT 5").result():
    print(dict(row))
