import os
from google.cloud import bigquery
proj=os.environ['GCP_PROJECT']; ds=os.environ['GCP_BQ_DATASET']; loc=os.environ['GCP_LOCATION']
c=bigquery.Client(project=proj)
base=f"{proj}.{ds}"

# Add feature_timestamp column (required by Vertex Feature Store BQ source)
stmts = [
  f"CREATE OR REPLACE TABLE `{base}.rawa55c41b` AS SELECT row_id, a_val, CURRENT_TIMESTAMP() AS feature_timestamp FROM `{base}.rawa55c41b`",
  f"CREATE OR REPLACE TABLE `{base}.rawb55c41b` AS SELECT row_id, b_val, CURRENT_TIMESTAMP() AS feature_timestamp FROM `{base}.rawb55c41b`",
  f"""CREATE OR REPLACE TABLE `{base}.derived55c41b` AS
      SELECT a.row_id AS row_id, ROUND(a.a_val + b.b_val, 6) AS col_sum, CURRENT_TIMESTAMP() AS feature_timestamp
      FROM `{base}.rawa55c41b` a JOIN `{base}.rawb55c41b` b ON a.row_id=b.row_id""",
]
for s in stmts:
    c.query(s, location=loc).result()
for t in ["rawa55c41b","rawb55c41b","derived55c41b"]:
    tb=c.get_table(f"{base}.{t}")
    print(t, tb.num_rows, [f.name for f in tb.schema])
