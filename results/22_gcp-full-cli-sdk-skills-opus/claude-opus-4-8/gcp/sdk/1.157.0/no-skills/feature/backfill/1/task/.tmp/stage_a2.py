import os
from google.cloud import bigquery
PROJECT = os.environ["GCP_PROJECT"]; DATASET = os.environ["GCP_BQ_DATASET"]; LOC = os.environ["GCP_LOCATION"]
client = bigquery.Client(project=PROJECT)
staging = f"{PROJECT}.{DATASET}.accountsed4daa_staging"
final = f"{PROJECT}.{DATASET}.accountsed4daa"

# Rebuild final: latest revision per row_id, plus feature_timestamp (TIMESTAMP) that the
# BigQuery-backed Vertex Feature Store requires as its event-time column, derived from updated_at.
sql = f"""
CREATE OR REPLACE TABLE `{final}` AS
SELECT row_id, status, balance, updated_at, TIMESTAMP_MILLIS(updated_at) AS feature_timestamp
FROM (
  SELECT row_id, status, balance, updated_at,
         ROW_NUMBER() OVER (PARTITION BY row_id ORDER BY updated_at DESC) AS rn
  FROM `{staging}`
)
WHERE rn = 1
"""
client.query(sql, location=LOC).result()
c = list(client.query(f"SELECT COUNT(*) n, COUNT(DISTINCT row_id) d FROM `{final}`", location=LOC).result())[0]
print("final rows", c.n, "distinct", c.d)
for r in list(client.query(f"SELECT * FROM `{final}` ORDER BY row_id LIMIT 2", location=LOC).result()):
    print(dict(r))
print("STAGE_A2_DONE")
