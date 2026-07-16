import os, json
from google.cloud import bigquery

PROJECT = os.environ["GCP_PROJECT"]
DATASET = os.environ["GCP_BQ_DATASET"]
LOCATION = os.environ["GCP_LOCATION"]
PREFIX = os.environ["MLPAB_GCP_PREFIX"]

client = bigquery.Client(project=PROJECT, location=LOCATION)

table_id = f"{PROJECT}.{DATASET}.{PREFIX}_prediction_log"

# 1) Load the CSV into BigQuery (ingestion on-platform)
schema = [
    bigquery.SchemaField("ts", "TIMESTAMP"),
    bigquery.SchemaField("prediction", "FLOAT"),
]
job_config = bigquery.LoadJobConfig(
    schema=schema,
    skip_leading_rows=1,
    source_format=bigquery.SourceFormat.CSV,
    write_disposition="WRITE_TRUNCATE",
)
with open("data/prediction_log.csv", "rb") as f:
    load_job = client.load_table_from_file(f, table_id, job_config=job_config)
load_job.result()
tbl = client.get_table(table_id)
print(f"Loaded {tbl.num_rows} rows into {table_id}")

# 2) Daily statistics (monitoring / statistics)
daily_sql = f"""
SELECT DATE(ts) AS d,
       COUNT(*) AS n,
       AVG(prediction) AS mean,
       STDDEV(prediction) AS sd,
       MIN(prediction) AS mn,
       MAX(prediction) AS mx
FROM `{table_id}`
GROUP BY d
ORDER BY d
"""
daily = list(client.query(daily_sql).result())
print(f"Days: {len(daily)}  range {daily[0].d} .. {daily[-1].d}")

# 3) Change-point detection on the platform via SQL.
# For each candidate onset date k, compute a two-sample Welch t-statistic
# comparing all predictions with DATE(ts) < k vs >= k. The onset is the
# candidate maximizing |t|.
cp_sql = f"""
WITH data AS (
  SELECT DATE(ts) AS d, prediction AS p FROM `{table_id}`
),
days AS (
  SELECT DISTINCT d FROM data
),
stats AS (
  SELECT
    k.d AS onset,
    COUNTIF(x.d <  k.d) AS n1,
    COUNTIF(x.d >= k.d) AS n2,
    AVG(IF(x.d <  k.d, x.p, NULL)) AS m1,
    AVG(IF(x.d >= k.d, x.p, NULL)) AS m2,
    VAR_SAMP(IF(x.d <  k.d, x.p, NULL)) AS v1,
    VAR_SAMP(IF(x.d >= k.d, x.p, NULL)) AS v2
  FROM days k CROSS JOIN data x
  GROUP BY k.d
)
SELECT
  onset, n1, n2, m1, m2,
  ABS(m2 - m1) AS abs_mean_diff,
  ABS(m2 - m1) / SQRT(v1/n1 + v2/n2) AS welch_t
FROM stats
WHERE n1 >= 150 AND n2 >= 150
ORDER BY welch_t DESC
LIMIT 10
"""
rows = list(client.query(cp_sql).result())
print("\nTop candidate onset dates by Welch t-statistic:")
for r in rows:
    print(f"  {r.onset}  t={r.welch_t:.2f}  m1={r.m1:.3f} m2={r.m2:.3f} "
          f"diff={r.abs_mean_diff:.3f} n1={r.n1} n2={r.n2}")

best = rows[0]
onset = str(best.onset)
print(f"\nDetected onset: {onset}")

# Save daily stats and per-day mean around the onset for a sanity check
window_sql = f"""
SELECT DATE(ts) AS d, COUNT(*) n, AVG(prediction) mean
FROM `{table_id}`
GROUP BY d ORDER BY d
"""
w = list(client.query(window_sql).result())
# print means around detected onset
onset_idx = next(i for i,x in enumerate(w) if str(x.d)==onset)
print("\nDaily means around onset:")
for x in w[max(0,onset_idx-4):onset_idx+4]:
    marker = " <== onset" if str(x.d)==onset else ""
    print(f"  {x.d}  n={x.n:3d}  mean={x.mean:.3f}{marker}")

os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump({"onset": onset}, f)
print("\nWrote submission/answers.json:", {"onset": onset})
