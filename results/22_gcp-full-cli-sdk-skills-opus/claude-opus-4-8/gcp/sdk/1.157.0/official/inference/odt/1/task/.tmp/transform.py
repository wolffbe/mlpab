import os
from google.cloud import bigquery

project = os.environ["GCP_PROJECT"]
dataset = os.environ["GCP_BQ_DATASET"]
loc = os.environ["GCP_LOCATION"]
client = bigquery.Client(project=project, location=loc)

ds = f"{project}.{dataset}"

# On-demand transformation: combine request-time coordinates with the stored
# profile. distance_deg rounded to 6dp; score uses the ROUNDED distance.
sql = f"""
CREATE OR REPLACE TABLE `{ds}.scored3ace95` AS
WITH j AS (
  SELECT
    r.request_id,
    r.account_id,
    ROUND(SQRT(POW(r.request_lat - p.home_lat, 2) +
               POW(r.request_lon - p.home_lon, 2)), 6) AS distance_deg,
    p.base_score
  FROM `{ds}.requests_raw` r
  JOIN `{ds}.profiles_raw` p USING (account_id)
)
SELECT
  request_id,
  account_id,
  distance_deg,
  ROUND(base_score - 0.1 * distance_deg, 6) AS score
FROM j
"""
client.query(sql, location=loc).result()

t = client.get_table(f"{ds}.scored3ace95")
print("scored3ace95 rows:", t.num_rows)
print("columns:", [f.name for f in t.schema])

# quick sanity peek
for row in client.query(
    f"SELECT * FROM `{ds}.scored3ace95` ORDER BY request_id LIMIT 5", location=loc
).result():
    print(dict(row))

# integrity checks
chk = list(client.query(
    f"SELECT COUNT(*) c, COUNT(DISTINCT request_id) d, "
    f"COUNTIF(distance_deg IS NULL OR score IS NULL) nulls "
    f"FROM `{ds}.scored3ace95`", location=loc).result())[0]
print("count/distinct/nulls:", chk.c, chk.d, chk.nulls)
