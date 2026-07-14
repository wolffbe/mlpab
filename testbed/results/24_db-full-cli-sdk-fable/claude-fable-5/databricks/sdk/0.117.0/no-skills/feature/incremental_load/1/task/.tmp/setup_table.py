import os, glob
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
catalog, schema = os.environ["MLPAB_DATABRICKS_SCHEMA"].split(".")
TABLE = f"{catalog}.{schema}.incrementalf3c1bf"
VOLUME_PATH = f"/Volumes/{catalog}/{schema}/raw_increments/events"

wh = next(x for x in w.warehouses.list() if x.name == "Serverless Starter Warehouse")

def sql(stmt):
    r = w.statement_execution.execute_statement(
        statement=stmt, warehouse_id=wh.id, wait_timeout="50s")
    if r.status.state.value != "SUCCEEDED":
        raise RuntimeError(f"SQL failed: {r.status}\n{stmt}")
    return r

# 1. Volume + upload increments
sql(f"CREATE VOLUME IF NOT EXISTS {catalog}.{schema}.raw_increments")
for f in sorted(glob.glob("data/increment_*.csv")):
    with open(f, "rb") as fh:
        w.files.upload(f"{VOLUME_PATH}/{os.path.basename(f)}", fh, overwrite=True)
    print("uploaded", f)

# 2. Feature table: record key row_id, event-time event_time (time-series PK), CDF for online sync
sql(f"""
CREATE TABLE IF NOT EXISTS {TABLE} (
  row_id STRING NOT NULL,
  account_id STRING,
  event_time BIGINT NOT NULL,
  amount DOUBLE,
  category STRING,
  CONSTRAINT incrementalf3c1bf_pk PRIMARY KEY (row_id, event_time TIMESERIES)
) TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

# 3. Load ALL increments (COPY INTO is idempotent per-file)
sql(f"""
COPY INTO {TABLE}
FROM (SELECT row_id, account_id, CAST(event_time AS BIGINT) AS event_time,
             CAST(amount AS DOUBLE) AS amount, category
      FROM '{VOLUME_PATH}/')
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true')
""")
r = sql(f"SELECT COUNT(*), MIN(event_time), MAX(event_time) FROM {TABLE}")
print("table row count / min / max event_time:", r.result.data_array[0])
