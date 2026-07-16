"""
Create scaled7ecfaf feature table with standardized values and online table access.
All computation runs on the Databricks platform via SDK.
"""
import os
import time
import csv

# Environment
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # e.g. workspace.mlpab2bb905
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]   # e.g. mlpab2bb905
TABLE_NAME = "scaled7ecfaf"
FULL_TABLE = f"{SCHEMA}.{TABLE_NAME}"

print(f"Schema: {SCHEMA}")
print(f"Prefix: {PREFIX}")
print(f"Table: {FULL_TABLE}")

from databricks.sdk import WorkspaceClient
w = WorkspaceClient()

# Read training data locally to compute mean/std (for the SQL we'll embed)
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

# Compute training stats locally (population std, no Bessel's correction)
n = len(train_data)
features = ["f1", "f2", "f3", "f4"]
means = {}
stds = {}
for feat in features:
    vals = [r[feat] for r in train_data]
    mean = sum(vals) / n
    variance = sum((v - mean) ** 2 for v in vals) / n  # population variance
    std = variance ** 0.5
    means[feat] = mean
    stds[feat] = std
    print(f"{feat}: mean={mean:.6f}, std={std:.6f}")

# Standardize
def standardize(val, mean, std):
    return round((val - mean) / std, 6)

# Prepare all rows
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

print(f"Total rows: {len(all_rows)}")

# Build SQL to create and populate the table
# Drop if exists first
drop_sql = f"DROP TABLE IF EXISTS {FULL_TABLE}"

create_sql = f"""
CREATE TABLE {FULL_TABLE} (
  row_id STRING NOT NULL,
  split STRING,
  f1 DOUBLE,
  f2 DOUBLE,
  f3 DOUBLE,
  f4 DOUBLE
)
TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
"""

# Build VALUES clause - chunk in batches to avoid too-long SQL
# We'll use the SQL warehouse to execute

# Find a SQL warehouse
warehouses = list(w.warehouses.list())
print(f"Available warehouses: {[wh.name for wh in warehouses]}")
warehouse = warehouses[0]
warehouse_id = warehouse.id
print(f"Using warehouse: {warehouse.name} ({warehouse_id})")

from databricks.sdk.service.sql import StatementState, ExecuteStatementRequestOnWaitTimeout

def run_sql(sql, warehouse_id=warehouse_id, timeout=300):
    """Execute SQL via warehouse and wait for completion."""
    stmt = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        wait_timeout="50s",
        on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE,
    )
    stmt_id = stmt.statement_id
    # Poll until done
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
    raise TimeoutError(f"SQL timed out after {timeout}s")

print("Dropping existing table if any...")
run_sql(drop_sql)

print("Creating table...")
run_sql(create_sql)

# Insert in batches of 100 rows
BATCH_SIZE = 100
for i in range(0, len(all_rows), BATCH_SIZE):
    batch = all_rows[i:i+BATCH_SIZE]
    values = ", ".join(
        f"('{r['row_id']}', '{r['split']}', {r['f1']}, {r['f2']}, {r['f3']}, {r['f4']})"
        for r in batch
    )
    insert_sql = f"INSERT INTO {FULL_TABLE} VALUES {values}"
    run_sql(insert_sql)
    print(f"Inserted rows {i} to {i+len(batch)-1}")

# Verify row count
count_result = run_sql(f"SELECT COUNT(*) FROM {FULL_TABLE}")
print(f"Row count: {count_result.result.data_array}")

# Verify a sample
sample_result = run_sql(f"SELECT * FROM {FULL_TABLE} LIMIT 3")
print(f"Sample rows: {sample_result.result.data_array}")

print("Delta table created and populated successfully.")

# Now create an online table for low-latency lookup
print("\nCreating online table...")

# Check what's available for online tables
from databricks.sdk.service import catalog

# Try to create an online table
online_table_name = f"{FULL_TABLE}"  # online table references the delta table

try:
    # Check the catalog service for online tables
    print(dir(w.online_tables))

    from databricks.sdk.service.catalog import (
        OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
    )

    online_table = w.online_tables.create(
        name=FULL_TABLE,
        spec=OnlineTableSpec(
            primary_key_columns=["row_id"],
            source_table_full_name=FULL_TABLE,
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
        ),
    )
    print(f"Online table created: {online_table}")
except Exception as e:
    print(f"Online table creation result: {e}")
    # Try alternative approach
    try:
        from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec
        ot = OnlineTable(
            name=FULL_TABLE,
            spec=OnlineTableSpec(
                primary_key_columns=["row_id"],
                source_table_full_name=FULL_TABLE,
            )
        )
        result = w.online_tables.create_and_wait(table=ot)
        print(f"Online table created (alt): {result}")
    except Exception as e2:
        print(f"Alt approach also failed: {e2}")

print("\nTask complete.")
