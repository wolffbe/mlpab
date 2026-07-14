from google.cloud import bigquery
import os
proj=os.environ['GCP_PROJECT']; ds=os.environ['GCP_BQ_DATASET']; loc=os.environ['GCP_LOCATION']
c=bigquery.Client(project=proj)
D=f"`{proj}.{ds}`"

FEAT_CTE = f"""
WITH allt AS (
  SELECT transaction_id, cc_num, datetime, amount, merchant, category, lat, long, is_fraud, 'train' AS split
  FROM {D}.stg_transactions
  UNION ALL
  SELECT transaction_id, cc_num, datetime, amount, merchant, category, lat, long, CAST(NULL AS INT64) AS is_fraud, 'score' AS split
  FROM {D}.stg_score
),
base AS (SELECT *, UNIX_SECONDS(datetime) AS ts FROM allt),
cardstats AS (SELECT cc_num, AVG(lat) AS clat, AVG(long) AS clong, AVG(amount) AS avg_amt FROM base GROUP BY cc_num),
feat AS (
  SELECT
    b.transaction_id, b.cc_num, b.datetime, b.split, b.is_fraud,
    b.amount,
    LN(b.amount+1) AS log_amount,
    EXTRACT(HOUR FROM b.datetime) AS hour,
    EXTRACT(DAYOFWEEK FROM b.datetime) AS dow,
    b.category,
    COUNT(*) OVER (PARTITION BY b.cc_num ORDER BY b.ts RANGE BETWEEN 3600 PRECEDING AND 1 PRECEDING) AS velocity_1h,
    COUNT(*) OVER (PARTITION BY b.cc_num ORDER BY b.ts RANGE BETWEEN 86400 PRECEDING AND 1 PRECEDING) AS velocity_24h,
    COALESCE(b.ts - LAG(b.ts) OVER (PARTITION BY b.cc_num ORDER BY b.ts), 999999) AS time_since_last,
    SAFE_DIVIDE(b.amount, cs.avg_amt) AS amount_ratio,
    6371*ACOS(GREATEST(-1,LEAST(1,
      SIN(cs.clat*ACOS(-1)/180)*SIN(b.lat*ACOS(-1)/180)+
      COS(cs.clat*ACOS(-1)/180)*COS(b.lat*ACOS(-1)/180)*COS((b.long-cs.clong)*ACOS(-1)/180)))) AS geo_dist
  FROM base b JOIN cardstats cs USING(cc_num)
)
"""

def run(sql):
    job=c.query(sql, location=loc); job.result(); return job

# Feature group: engineered features for the labelled history
run(f"""
CREATE OR REPLACE TABLE {D}.cctxn76ccb2 AS
{FEAT_CTE}
SELECT transaction_id, cc_num, datetime, amount, log_amount, hour, dow, category,
       velocity_1h, velocity_24h, time_since_last, amount_ratio, geo_dist, is_fraud
FROM feat WHERE split='train'
""")
print("cctxn76ccb2 created")

# Score features table (intermediate, for prediction)
run(f"""
CREATE OR REPLACE TABLE {D}.score_feat AS
{FEAT_CTE}
SELECT transaction_id, cc_num, datetime, amount, log_amount, hour, dow, category,
       velocity_1h, velocity_24h, time_since_last, amount_ratio, geo_dist
FROM feat WHERE split='score'
""")
print("score_feat created")

for t in ["cctxn76ccb2","score_feat"]:
    tb=c.get_table(f"{proj}.{ds}.{t}"); print(t, tb.num_rows, "rows")

# sanity: fraud rate + a peek
for row in c.query(f"SELECT is_fraud, COUNT(*) n FROM {D}.cctxn76ccb2 GROUP BY is_fraud ORDER BY is_fraud", location=loc).result():
    print("label", row.is_fraud, row.n)
print("done")
