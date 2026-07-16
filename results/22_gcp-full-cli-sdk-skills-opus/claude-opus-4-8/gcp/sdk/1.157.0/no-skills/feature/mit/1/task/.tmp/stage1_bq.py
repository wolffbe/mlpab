import os
from google.cloud import bigquery

PROJECT = os.environ["GCP_PROJECT"]
DATASET = os.environ["GCP_BQ_DATASET"]
LOCATION = os.environ["GCP_LOCATION"]

client = bigquery.Client(project=PROJECT, location=LOCATION)

ds_ref = f"{PROJECT}.{DATASET}"
tx_tbl = f"{ds_ref}._stg_transactions"
fx_tbl = f"{ds_ref}._stg_fx_rates"
feat_tbl = f"{ds_ref}.features347afc"

# --- Load transactions.csv ---
tx_schema = [
    bigquery.SchemaField("row_id", "STRING"),
    bigquery.SchemaField("account_id", "STRING"),
    bigquery.SchemaField("event_time", "INT64"),
    bigquery.SchemaField("amount", "FLOAT64"),
    bigquery.SchemaField("currency", "STRING"),
]
job_cfg = bigquery.LoadJobConfig(
    schema=tx_schema, skip_leading_rows=1, source_format=bigquery.SourceFormat.CSV,
    write_disposition="WRITE_TRUNCATE",
)
with open("data/transactions.csv", "rb") as f:
    client.load_table_from_file(f, tx_tbl, job_config=job_cfg).result()
print("loaded transactions:", client.get_table(tx_tbl).num_rows)

# --- Load fx_rates.csv ---
fx_schema = [
    bigquery.SchemaField("currency", "STRING"),
    bigquery.SchemaField("fx_rate", "FLOAT64"),
]
job_cfg = bigquery.LoadJobConfig(
    schema=fx_schema, skip_leading_rows=1, source_format=bigquery.SourceFormat.CSV,
    write_disposition="WRITE_TRUNCATE",
)
with open("data/fx_rates.csv", "rb") as f:
    client.load_table_from_file(f, fx_tbl, job_config=job_cfg).result()
print("loaded fx_rates:", client.get_table(fx_tbl).num_rows)

# --- Compute feature table (all work in BigQuery) ---
# 7 days in epoch milliseconds
WIN_MS = 7 * 24 * 60 * 60 * 1000  # 604800000
sql = f"""
CREATE OR REPLACE TABLE `{feat_tbl}` AS
WITH joined AS (
  SELECT
    t.row_id,
    t.account_id,
    t.event_time,
    t.amount,
    t.amount * f.fx_rate AS amount_usd,
    CASE WHEN EXTRACT(DAYOFWEEK FROM TIMESTAMP_MILLIS(t.event_time)) IN (1, 7)
         THEN 1 ELSE 0 END AS is_weekend
  FROM `{tx_tbl}` t
  JOIN `{fx_tbl}` f ON t.currency = f.currency
)
SELECT
  row_id,
  account_id,
  event_time,
  amount_usd,
  is_weekend,
  SUM(amount) OVER (
    PARTITION BY account_id
    ORDER BY event_time
    RANGE BETWEEN {WIN_MS} PRECEDING AND CURRENT ROW
  ) AS amount_7d
FROM joined
"""
client.query(sql, location=LOCATION).result()
t = client.get_table(feat_tbl)
print("feature table rows:", t.num_rows)
print("columns:", [f.name for f in t.schema])

# sanity preview
rows = list(client.query(
    f"SELECT row_id, account_id, event_time, amount_usd, is_weekend, amount_7d "
    f"FROM `{feat_tbl}` ORDER BY event_time LIMIT 5", location=LOCATION).result())
for r in rows:
    print(dict(r))
