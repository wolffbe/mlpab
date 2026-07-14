import csv, sys, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
SCHEMA = "workspace.mlpabc5b156"
WAREHOUSES = ["a832b544eb7dc3fe", "8a93fc195da2ceb1"]

def try_sql(stmt, wh, deadline):
    r = w.statement_execution.execute_statement(
        statement=stmt, warehouse_id=wh, wait_timeout="30s")
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        if time.time() > deadline:
            w.statement_execution.cancel_execution(r.statement_id)
            raise TimeoutError(f"statement still {r.status.state} at deadline")
        time.sleep(8)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"{r.status.state}: {r.status.error}")
    return r

# Warm-up: find a warehouse that can actually run a statement.
WH = None
for cand in WAREHOUSES:
    try:
        try_sql("SELECT 1", cand, time.time() + 420)
        WH = cand
        print("warehouse usable:", cand, flush=True)
        break
    except Exception as e:
        print("warehouse", cand, "unusable:", e, flush=True)
if WH is None:
    print("NO WAREHOUSE AVAILABLE", flush=True)
    sys.exit(2)

def sql(stmt):
    return try_sql(stmt, WH, time.time() + 300)

def load(csv_path, table, valcol):
    sql(f"DROP TABLE IF EXISTS {SCHEMA}.{table}")
    sql(f"""CREATE TABLE {SCHEMA}.{table} (
        row_id STRING NOT NULL,
        {valcol} DOUBLE,
        CONSTRAINT {table}_pk PRIMARY KEY (row_id)
    ) TBLPROPERTIES (delta.enableChangeDataFeed = true)""")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    vals = ",".join(f"('{r['row_id']}',{r[valcol]})" for r in rows)
    sql(f"INSERT INTO {SCHEMA}.{table} (row_id, {valcol}) VALUES {vals}")
    n = sql(f"SELECT count(*) FROM {SCHEMA}.{table}").result.data_array[0][0]
    print(f"{table}: {n} rows loaded", flush=True)

load("data/raw_a.csv", "rawa5a60b6", "a_val")
load("data/raw_b.csv", "rawb5a60b6", "b_val")

# Derived table: PK + CDF for online use; INSERT..SELECT records UC lineage
sql(f"DROP TABLE IF EXISTS {SCHEMA}.derived5a60b6")
sql(f"""CREATE TABLE {SCHEMA}.derived5a60b6 (
    row_id STRING NOT NULL,
    col_sum DOUBLE,
    CONSTRAINT derived5a60b6_pk PRIMARY KEY (row_id)
) TBLPROPERTIES (delta.enableChangeDataFeed = true)""")
sql(f"""INSERT INTO {SCHEMA}.derived5a60b6 (row_id, col_sum)
    SELECT a.row_id, round(a.a_val + b.b_val, 6) AS col_sum
    FROM {SCHEMA}.rawa5a60b6 a
    INNER JOIN {SCHEMA}.rawb5a60b6 b ON a.row_id = b.row_id""")
r = sql(f"SELECT count(*) FROM {SCHEMA}.derived5a60b6")
print("derived5a60b6 rows:", r.result.data_array[0][0], flush=True)
r = sql(f"SELECT * FROM {SCHEMA}.derived5a60b6 ORDER BY row_id LIMIT 3")
for row in r.result.data_array:
    print(row, flush=True)
print("TABLES DONE", flush=True)
