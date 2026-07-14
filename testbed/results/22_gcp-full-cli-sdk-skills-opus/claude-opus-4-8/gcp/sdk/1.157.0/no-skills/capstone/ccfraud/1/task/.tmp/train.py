from google.cloud import bigquery
import os
proj=os.environ['GCP_PROJECT']; ds=os.environ['GCP_BQ_DATASET']; loc=os.environ['GCP_LOCATION']
pref=os.environ['MLPAB_GCP_PREFIX']
c=bigquery.Client(project=proj)
D=f"`{proj}.{ds}`"
def run(sql):
    j=c.query(sql, location=loc); j.result(); return j

# Training dataset: model-ready feature columns + label
run(f"""
CREATE OR REPLACE TABLE {D}.cctd76ccb2 AS
SELECT transaction_id, amount, log_amount, hour, dow, category,
       velocity_1h, velocity_24h, time_since_last, amount_ratio, geo_dist, is_fraud
FROM {D}.cctxn76ccb2
""")
print("cctd76ccb2 created", c.get_table(f"{proj}.{ds}.cctd76ccb2").num_rows)

# Train BQML logistic regression, register to Vertex AI Model Registry
vid = f"{pref}-ccmodel76ccb2".replace("_","-").lower()
run(f"""
CREATE OR REPLACE MODEL {D}.ccmodel76ccb2
OPTIONS(
  model_type='LOGISTIC_REG',
  input_label_cols=['is_fraud'],
  auto_class_weights=TRUE,
  data_split_method='RANDOM',
  data_split_eval_fraction=0.2,
  model_registry='vertex_ai',
  vertex_ai_model_id='{vid}',
  vertex_ai_model_version_aliases=['default']
) AS
SELECT amount, log_amount, hour, dow, category,
       velocity_1h, velocity_24h, time_since_last, amount_ratio, geo_dist,
       CAST(is_fraud AS INT64) AS is_fraud
FROM {D}.cctd76ccb2
""")
print("model ccmodel76ccb2 trained; vertex id", vid)

# Evaluation metrics
row=list(c.query(f"SELECT * FROM ML.EVALUATE(MODEL {D}.ccmodel76ccb2)", location=loc).result())[0]
metrics={k:row[k] for k in row.keys()}
print("EVAL METRICS:", metrics)
import json
with open(".tmp/metrics.json","w") as f: json.dump({k:(float(v) if v is not None else None) for k,v in metrics.items()}, f)
print("done")
