from google.cloud import bigquery
import os
proj=os.environ['GCP_PROJECT']; ds=os.environ['GCP_BQ_DATASET']; loc=os.environ['GCP_LOCATION']
c=bigquery.Client(project=proj)
D=f"`{proj}.{ds}`"
def run(sql):
    j=c.query(sql, location=loc); j.result(); return j

run(f"""
CREATE OR REPLACE TABLE {D}.ccpred76ccb2 AS
SELECT
  transaction_id,
  (SELECT prob FROM UNNEST(predicted_is_fraud_probs) WHERE CAST(label AS INT64)=1) AS fraud_probability
FROM ML.PREDICT(MODEL {D}.ccmodel76ccb2,
  (SELECT transaction_id, amount, log_amount, hour, dow, category,
          velocity_1h, velocity_24h, time_since_last, amount_ratio, geo_dist
   FROM {D}.score_feat))
""")
t=c.get_table(f"{proj}.{ds}.ccpred76ccb2")
print("ccpred76ccb2 rows", t.num_rows)
rows=list(c.query(f"SELECT MIN(fraud_probability) mn, MAX(fraud_probability) mx, AVG(fraud_probability) av, COUNTIF(fraud_probability IS NULL) nulls FROM {D}.ccpred76ccb2", location=loc).result())[0]
print("prob stats", rows.mn, rows.mx, rows.av, "nulls", rows.nulls)
for r in c.query(f"SELECT * FROM {D}.ccpred76ccb2 LIMIT 3", location=loc).result():
    print(dict(r))
print("done")
