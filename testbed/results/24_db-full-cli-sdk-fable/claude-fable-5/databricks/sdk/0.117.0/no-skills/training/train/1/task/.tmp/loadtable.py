import os, time
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
catalog, schema = os.environ["MLPAB_DATABRICKS_SCHEMA"].split(".")
vol = f"/Volumes/{catalog}/{schema}/taskvol"
tbl = f"{catalog}.{schema}.predictions178367"

# confirm the job output exists
files = [f.path for f in w.files.list_directory_contents(f"{vol}")]
print("volume files:", files)

wh = next(x for x in w.warehouses.list() if "Serverless" in (x.name or "") or True)
print("warehouse:", wh.name, wh.id)

def sql(stmt):
    r = w.statement_execution.execute_statement(statement=stmt, warehouse_id=wh.id, wait_timeout="50s")
    while r.status.state.value in ("PENDING", "RUNNING"):
        time.sleep(3)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state.value != "SUCCEEDED":
        raise RuntimeError(f"{r.status.state}: {r.status.error}")
    return r

sql(f"""
CREATE OR REPLACE TABLE {tbl} (
  row_id STRING NOT NULL,
  score DOUBLE,
  CONSTRAINT predictions178367_pk PRIMARY KEY (row_id)
) TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
print("table created")

sql(f"""
INSERT INTO {tbl}
SELECT CAST(row_id AS STRING), CAST(score AS DOUBLE)
FROM read_files('{vol}/predictions.csv', format => 'csv', header => true)
""")
print("data inserted")

r = sql(f"SELECT COUNT(*), MIN(score), MAX(score) FROM {tbl}")
print("verify:", r.result.data_array)
r = sql(f"SELECT * FROM {tbl} ORDER BY row_id LIMIT 3")
print("sample:", r.result.data_array)
