import time
import databricks.sdk as dsdk
from databricks.sdk.service import sql

w = dsdk.WorkspaceClient()
WH = "4dfab06c923fe3cc"
CATALOG, SCHEMA = "workspace", "mlpab2138eb"
TABLE = "predictionsa834e5"
FQN = f"{CATALOG}.{SCHEMA}.{TABLE}"
VOLCSV = f"/Volumes/{CATALOG}/{SCHEMA}/trainjoba834e5_data/predictions.csv"

def run(stmt):
    r = w.statement_execution.execute_statement(
        warehouse_id=WH, statement=stmt, catalog=CATALOG, schema=SCHEMA,
        wait_timeout="50s")
    # poll if needed
    while r.status.state in (sql.StatementState.PENDING, sql.StatementState.RUNNING):
        time.sleep(2)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != sql.StatementState.SUCCEEDED:
        raise RuntimeError(f"FAILED: {r.status.state} :: {getattr(r.status,'error',None)}")
    return r

run(f"DROP TABLE IF EXISTS {FQN}")
run(f"""CREATE TABLE {FQN} (
  row_id STRING NOT NULL,
  score DOUBLE,
  CONSTRAINT {TABLE}_pk PRIMARY KEY(row_id)
) TBLPROPERTIES (delta.enableChangeDataFeed = true)""")
print("table created")

run(f"""COPY INTO {FQN} FROM (
  SELECT CAST(row_id AS STRING) AS row_id, CAST(score AS DOUBLE) AS score
  FROM '{VOLCSV}'
) FILEFORMAT = CSV FORMAT_OPTIONS ('header'='true')""")
print("data loaded")

r = run(f"SELECT COUNT(*) AS c, MIN(score), MAX(score) FROM {FQN}")
print("rows:", r.result.data_array)
r = run(f"SELECT * FROM {FQN} ORDER BY row_id LIMIT 3")
print("sample:", r.result.data_array)
