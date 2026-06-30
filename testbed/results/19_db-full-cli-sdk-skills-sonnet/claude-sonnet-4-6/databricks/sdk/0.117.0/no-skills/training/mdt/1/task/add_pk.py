"""Add primary key and reinsert data into feature table."""
import os
import time
import csv

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState, ExecuteStatementRequestOnWaitTimeout

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
TABLE_NAME = "scaled7ecfaf"
FULL_TABLE = f"{SCHEMA}.{TABLE_NAME}"

w = WorkspaceClient()
warehouses = list(w.warehouses.list())
warehouse_id = warehouses[0].id

def run_sql(sql, timeout=120):
    stmt = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        wait_timeout="50s",
        on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE,
    )
    stmt_id = stmt.statement_id
    deadline = time.time() + timeout
    while time.time() < deadline:
        result = w.statement_execution.get_statement(stmt_id)
        state = result.status.state
        if state in (StatementState.SUCCEEDED, StatementState.FAILED,
                     StatementState.CANCELED, StatementState.CLOSED):
            if state != StatementState.SUCCEEDED:
                raise RuntimeError(f"SQL failed ({state}): {result.status.error}")
            return result
        time.sleep(2)
    raise TimeoutError("SQL timed out")

# Try adding PK constraint to existing table
print("Trying to add primary key constraint...")
try:
    run_sql(f"ALTER TABLE {FULL_TABLE} ADD CONSTRAINT pk_row_id PRIMARY KEY (row_id)")
    print("Primary key constraint added.")
except Exception as e:
    print(f"ALTER failed: {e}")
    print("Will recreate table with PK...")

    # Read training data
    train_data = []
    with open("data/features_train.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            train_data.append({
                "row_id": row["row_id"],
                "f1": float(row["f1"]),
                "f2": float(row["f2"]),
                "f3": float(row["f3"]),
                "f4": float(row["f4"]),
            })

    serve_data = []
    with open("data/features_serve.csv") as f:
        reader = csv.DictReader(f)
        for row in reader:
            serve_data.append({
                "row_id": row["row_id"],
                "f1": float(row["f1"]),
                "f2": float(row["f2"]),
                "f3": float(row["f3"]),
                "f4": float(row["f4"]),
            })

    # Compute training stats
    n = len(train_data)
    features = ["f1", "f2", "f3", "f4"]
    means = {}
    stds = {}
    for feat in features:
        vals = [r[feat] for r in train_data]
        mean = sum(vals) / n
        variance = sum((v - mean) ** 2 for v in vals) / n
        std = variance ** 0.5
        means[feat] = mean
        stds[feat] = std

    def standardize(val, mean, std):
        return round((val - mean) / std, 6)

    all_rows = []
    for row in train_data:
        all_rows.append({
            "row_id": row["row_id"],
            "split": "train",
            "f1": standardize(row["f1"], means["f1"], stds["f1"]),
            "f2": standardize(row["f2"], means["f2"], stds["f2"]),
            "f3": standardize(row["f3"], means["f3"], stds["f3"]),
            "f4": standardize(row["f4"], means["f4"], stds["f4"]),
        })
    for row in serve_data:
        all_rows.append({
            "row_id": row["row_id"],
            "split": "serve",
            "f1": standardize(row["f1"], means["f1"], stds["f1"]),
            "f2": standardize(row["f2"], means["f2"], stds["f2"]),
            "f3": standardize(row["f3"], means["f3"], stds["f3"]),
            "f4": standardize(row["f4"], means["f4"], stds["f4"]),
        })

    # Drop and recreate with PK
    run_sql(f"DROP TABLE IF EXISTS {FULL_TABLE}")
    run_sql(f"""CREATE TABLE {FULL_TABLE} (
  row_id STRING NOT NULL,
  split STRING,
  f1 DOUBLE,
  f2 DOUBLE,
  f3 DOUBLE,
  f4 DOUBLE,
  CONSTRAINT pk_row_id PRIMARY KEY (row_id)
) TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')""")
    print("Table recreated with PK")

    # Re-insert data
    BATCH_SIZE = 100
    for i in range(0, len(all_rows), BATCH_SIZE):
        batch = all_rows[i:i+BATCH_SIZE]
        values = ", ".join(
            f"('{r['row_id']}', '{r['split']}', {r['f1']}, {r['f2']}, {r['f3']}, {r['f4']})"
            for r in batch
        )
        run_sql(f"INSERT INTO {FULL_TABLE} VALUES {values}")
        print(f"Inserted rows {i} to {i+len(batch)-1}")

    print(f"Total rows: {len(all_rows)}")

# Verify
count = run_sql(f"SELECT COUNT(*) FROM {FULL_TABLE}")
print(f"Row count: {count.result.data_array}")
print("Done - table ready with primary key.")
