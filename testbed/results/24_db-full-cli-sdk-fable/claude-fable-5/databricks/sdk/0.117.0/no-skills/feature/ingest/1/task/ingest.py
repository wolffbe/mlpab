import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
CAT, SCH = "workspace", "mlpabce02a9"
FQ = f"{CAT}.{SCH}"
WH = "8a93fc195da2ceb1"

def sql(stmt, timeout="50s"):
    r = w.statement_execution.execute_statement(statement=stmt, warehouse_id=WH, wait_timeout=timeout)
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(3)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL failed: {r.status.state} {r.status.error}\n{stmt}")
    return r

# 1. Volume + upload CSVs
sql(f"CREATE SCHEMA IF NOT EXISTS {FQ}")
sql(f"CREATE VOLUME IF NOT EXISTS {FQ}.ingest_staging")
for f in ("transactions_export_1.csv", "transactions_export_2.csv"):
    with open(f"data/{f}", "rb") as fh:
        w.files.upload(f"/Volumes/{CAT}/{SCH}/ingest_staging/{f}", fh, overwrite=True)
    print("uploaded", f)

# 2. Create feature table: PK row_id, CDF enabled (needed for online table)
sql(f"DROP TABLE IF EXISTS {FQ}.transactionsf10ad0")
sql(f"""
CREATE TABLE {FQ}.transactionsf10ad0 (
  row_id STRING NOT NULL,
  account_id STRING,
  event_time BIGINT,
  amount DOUBLE,
  category STRING,
  CONSTRAINT transactionsf10ad0_pk PRIMARY KEY (row_id)
) TBLPROPERTIES (delta.enableChangeDataFeed = true)
COMMENT 'Feature table: transactions; record key row_id, event-time event_time (epoch ms)'
""")

# 3. Load both exports, dedupe by row_id (files overlap)
sql(f"""
INSERT INTO {FQ}.transactionsf10ad0
SELECT row_id, account_id, event_time, amount, category FROM (
  SELECT *, row_number() OVER (PARTITION BY row_id ORDER BY event_time) rn
  FROM read_files(
    '/Volumes/{CAT}/{SCH}/ingest_staging/transactions_export_*.csv',
    format => 'csv', header => true,
    schema => 'row_id STRING, account_id STRING, event_time BIGINT, amount DOUBLE, category STRING'
  )
) WHERE rn = 1
""", timeout="0s")

r = sql(f"SELECT count(*), count(DISTINCT row_id) FROM {FQ}.transactionsf10ad0")
print("rows, distinct row_ids:", r.result.data_array)
