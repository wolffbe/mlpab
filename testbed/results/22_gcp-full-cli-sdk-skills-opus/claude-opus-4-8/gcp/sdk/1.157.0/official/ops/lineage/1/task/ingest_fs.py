import os
from google.cloud import bigquery

proj = os.environ['GCP_PROJECT']; ds = os.environ['GCP_BQ_DATASET']; loc = os.environ['GCP_LOCATION']
c = bigquery.Client(project=proj)

def load(table, path, valcol):
    tid = f"{proj}.{ds}.{table}"
    schema = [
        bigquery.SchemaField("row_id", "STRING"),
        bigquery.SchemaField(valcol, "FLOAT64"),
    ]
    job_config = bigquery.LoadJobConfig(
        schema=schema, skip_leading_rows=1,
        source_format=bigquery.SourceFormat.CSV,
        write_disposition="WRITE_TRUNCATE",
    )
    with open(path, "rb") as f:
        job = c.load_table_from_file(f, tid, job_config=job_config, location=loc)
    job.result()
    t = c.get_table(tid)
    print(f"loaded {table}: {t.num_rows} rows, schema={[s.name for s in t.schema]}")

load("rawa55c41b", "data/raw_a.csv", "a_val")
load("rawb55c41b", "data/raw_b.csv", "b_val")

derived_tid = f"{proj}.{ds}.derived55c41b"
sql = f"""
CREATE OR REPLACE TABLE `{derived_tid}` AS
SELECT a.row_id AS row_id,
       ROUND(a.a_val + b.b_val, 6) AS col_sum
FROM `{proj}.{ds}.rawa55c41b` a
JOIN `{proj}.{ds}.rawb55c41b` b
  ON a.row_id = b.row_id
"""
c.query(sql, location=loc).result()
t = c.get_table(derived_tid)
print(f"derived55c41b: {t.num_rows} rows, schema={[s.name for s in t.schema]}")
rows = list(c.query(f"SELECT row_id, col_sum FROM `{derived_tid}` ORDER BY row_id LIMIT 3", location=loc).result())
for r in rows: print(dict(r))
