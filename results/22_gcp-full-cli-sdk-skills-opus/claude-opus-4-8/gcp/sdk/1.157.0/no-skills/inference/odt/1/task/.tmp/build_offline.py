import os
from google.cloud import bigquery

proj = os.environ['GCP_PROJECT']; ds = os.environ['GCP_BQ_DATASET']
c = bigquery.Client(project=proj)

def load(csv, table, schema):
    ref = f"{proj}.{ds}.{table}"
    jc = bigquery.LoadJobConfig(source_format=bigquery.SourceFormat.CSV,
                                skip_leading_rows=1, schema=schema,
                                write_disposition="WRITE_TRUNCATE")
    with open(csv, "rb") as f:
        c.load_table_from_file(f, ref, job_config=jc).result()
    print("loaded", ref, c.get_table(ref).num_rows, "rows")

load("data/requests.csv", "requests", [
    bigquery.SchemaField("request_id", "STRING"),
    bigquery.SchemaField("account_id", "STRING"),
    bigquery.SchemaField("request_lat", "FLOAT64"),
    bigquery.SchemaField("request_lon", "FLOAT64"),
    bigquery.SchemaField("requested_at", "TIMESTAMP"),
])
load("data/profiles.csv", "profiles", [
    bigquery.SchemaField("account_id", "STRING"),
    bigquery.SchemaField("home_lat", "FLOAT64"),
    bigquery.SchemaField("home_lon", "FLOAT64"),
    bigquery.SchemaField("base_score", "FLOAT64"),
])

sql = f"""
CREATE OR REPLACE TABLE `{proj}.{ds}.scored3ace95` AS
WITH j AS (
  SELECT r.request_id, r.account_id,
         ROUND(SQRT(POW(r.request_lat - p.home_lat,2)+POW(r.request_lon - p.home_lon,2)),6) AS distance_deg,
         p.base_score
  FROM `{proj}.{ds}.requests` r
  JOIN `{proj}.{ds}.profiles` p USING (account_id)
)
SELECT request_id, account_id, distance_deg,
       ROUND(base_score - 0.1*distance_deg, 6) AS score,
       CURRENT_TIMESTAMP() AS feature_timestamp
FROM j
"""
c.query(sql).result()
t = c.get_table(f"{proj}.{ds}.scored3ace95")
print("scored3ace95 rows:", t.num_rows, "cols:", [f.name for f in t.schema])
for row in c.query(f"SELECT request_id,account_id,distance_deg,score FROM `{proj}.{ds}.scored3ace95` ORDER BY request_id LIMIT 3").result():
    print(dict(row))
