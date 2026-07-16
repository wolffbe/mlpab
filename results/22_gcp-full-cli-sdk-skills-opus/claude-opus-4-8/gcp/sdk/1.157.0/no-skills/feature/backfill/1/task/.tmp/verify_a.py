import os
from google.cloud import bigquery
PROJECT = os.environ["GCP_PROJECT"]; DATASET = os.environ["GCP_BQ_DATASET"]
LOC = os.environ["GCP_LOCATION"]
client = bigquery.Client(project=PROJECT)
staging = f"{PROJECT}.{DATASET}.accountsed4daa_staging"
final = f"{PROJECT}.{DATASET}.accountsed4daa"

# For every row_id, final.updated_at must equal max(updated_at) in staging
bad = list(client.query(f"""
SELECT COUNT(*) AS mism FROM (
  SELECT s.row_id, MAX(s.updated_at) AS max_ts, ANY_VALUE(f.updated_at) AS final_ts
  FROM `{staging}` s JOIN `{final}` f USING(row_id)
  GROUP BY s.row_id
) WHERE max_ts != final_ts
""", location=LOC).result())[0]
print("mismatched keys (should be 0):", bad.mism)

# also verify the balance/status come from the latest revision row
bad2 = list(client.query(f"""
WITH latest AS (
  SELECT row_id, status, balance, updated_at,
         ROW_NUMBER() OVER (PARTITION BY row_id ORDER BY updated_at DESC) rn
  FROM `{staging}`)
SELECT COUNT(*) AS mism FROM latest l
JOIN `{final}` f USING(row_id)
WHERE l.rn=1 AND (l.status!=f.status OR l.balance!=f.balance OR l.updated_at!=f.updated_at)
""", location=LOC).result())[0]
print("value mismatches vs latest revision (should be 0):", bad2.mism)

rows = list(client.query(f"SELECT * FROM `{final}` ORDER BY row_id LIMIT 3", location=LOC).result())
for r in rows: print(dict(r))
