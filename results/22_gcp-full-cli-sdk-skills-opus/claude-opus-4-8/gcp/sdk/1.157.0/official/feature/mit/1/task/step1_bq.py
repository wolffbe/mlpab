import os
from google.cloud import bigquery

PROJECT = os.environ["GCP_PROJECT"]
DATASET = os.environ["GCP_BQ_DATASET"]
client = bigquery.Client(project=PROJECT)

ds_ref = f"{PROJECT}.{DATASET}"

# --- load transactions.csv ---
tx_schema = [
    bigquery.SchemaField("row_id", "STRING"),
    bigquery.SchemaField("account_id", "STRING"),
    bigquery.SchemaField("event_time", "INT64"),
    bigquery.SchemaField("amount", "FLOAT64"),
    bigquery.SchemaField("currency", "STRING"),
]
job_cfg = bigquery.LoadJobConfig(
    schema=tx_schema, skip_leading_rows=1,
    source_format=bigquery.SourceFormat.CSV,
    write_disposition="WRITE_TRUNCATE",
)
with open("data/transactions.csv", "rb") as f:
    client.load_table_from_file(f, f"{ds_ref}.stg_transactions", job_config=job_cfg).result()
print("loaded transactions")

# --- load fx_rates.csv ---
fx_schema = [
    bigquery.SchemaField("currency", "STRING"),
    bigquery.SchemaField("fx_rate", "FLOAT64"),
]
job_cfg = bigquery.LoadJobConfig(
    schema=fx_schema, skip_leading_rows=1,
    source_format=bigquery.SourceFormat.CSV,
    write_disposition="WRITE_TRUNCATE",
)
with open("data/fx_rates.csv", "rb") as f:
    client.load_table_from_file(f, f"{ds_ref}.stg_fx_rates", job_config=job_cfg).result()
print("loaded fx_rates")

# --- compute features table ---
WEEK_MS = 7 * 24 * 60 * 60 * 1000
sql = f"""
CREATE OR REPLACE TABLE `{ds_ref}.features347afc` AS
SELECT
  t.row_id,
  t.account_id,
  t.event_time,
  t.amount * fx.fx_rate AS amount_usd,
  CASE WHEN EXTRACT(DAYOFWEEK FROM TIMESTAMP_MILLIS(t.event_time)) IN (1, 7)
       THEN 1 ELSE 0 END AS is_weekend,
  SUM(t.amount) OVER (
    PARTITION BY t.account_id
    ORDER BY t.event_time
    RANGE BETWEEN {WEEK_MS} PRECEDING AND CURRENT ROW
  ) AS amount_7d
FROM `{ds_ref}.stg_transactions` t
JOIN `{ds_ref}.stg_fx_rates` fx USING (currency)
"""
client.query(sql).result()
print("created features347afc")

# verify
q = f"SELECT COUNT(*) n, COUNT(DISTINCT row_id) k FROM `{ds_ref}.features347afc`"
for r in client.query(q).result():
    print("rows:", r.n, "distinct row_id:", r.k)
q2 = f"SELECT * FROM `{ds_ref}.features347afc` ORDER BY event_time LIMIT 5"
for r in client.query(q2).result():
    print(dict(r))
