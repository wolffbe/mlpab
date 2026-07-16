import os
from google.cloud import bigquery

project = os.environ["GCP_PROJECT"]
dataset = os.environ["GCP_BQ_DATASET"]
client = bigquery.Client(project=project)

# Compare serving f2 against candidate transforms of training f2, all in BigQuery
q = f"""
SELECT
  CORR(s.f2, EXP(t.f2))               AS corr_exp,
  CORR(LN(s.f2), t.f2)                AS corr_ln,
  CORR(s.f2, t.f2*t.f2)               AS corr_sq,
  AVG(ABS(s.f2 - EXP(t.f2)))          AS mad_exp,
  AVG(s.f2 - EXP(t.f2))               AS bias_exp
FROM `{project}.{dataset}.skew_train` t
JOIN `{project}.{dataset}.skew_serve` s USING(entity_id)
"""
row = list(client.query(q).result())[0]
for k in row.keys():
    print(f"{k}: {row[k]:.5f}")

q2 = f"""
SELECT t.entity_id, t.f2 AS train_f2, s.f2 AS serve_f2, EXP(t.f2) AS exp_train_f2
FROM `{project}.{dataset}.skew_train` t
JOIN `{project}.{dataset}.skew_serve` s USING(entity_id)
ORDER BY t.entity_id LIMIT 8
"""
print("entity   train_f2  serve_f2  exp(train_f2)")
for r in client.query(q2).result():
    print(f"{r['entity_id']}  {r['train_f2']:8.4f}  {r['serve_f2']:8.4f}  {r['exp_train_f2']:8.4f}")
