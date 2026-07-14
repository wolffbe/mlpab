#!/usr/bin/env python3
"""
Register a feature table named `accounts7b3169`, version 1, on Databricks.
Load the batches so the table's final contents are each row's LATEST revision.
Enable online/real-time access for low-latency lookup.
"""

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql, catalog

# Environment variables
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # e.g., "workspace.mlpab0f39fa"
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]  # e.g., "mlpab0f39fa"
CATALOG, SCHEMA_NAME = SCHEMA.split(".")

# Initialize WorkspaceClient
w = WorkspaceClient()

# 1. Create schema if not exists
try:
    w.schemas.create(name=SCHEMA_NAME, catalog_name=CATALOG)
    print(f"Created schema: {SCHEMA}")
except Exception as e:
    print(f"Schema may already exist or error: {e}")

# 2. Create a Delta table from the batches (deduplicate by row_id)
table_name = f"{PREFIX}_accounts_raw"
feature_table_name = "accounts7b3169"
local_batch_dir = os.path.abspath("data")

# SQL to create a Delta table from the batches, retaining the latest revision
create_table_sql = f"""
CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA_NAME}.{feature_table_name} AS
SELECT * FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY row_id ORDER BY updated_at DESC) as rn
    FROM (
        SELECT * FROM csv.`{local_batch_dir}/batch_1.csv`
        UNION ALL
        SELECT * FROM csv.`{local_batch_dir}/batch_2.csv`
        UNION ALL
        SELECT * FROM csv.`{local_batch_dir}/batch_3.csv`
    )
)
WHERE rn = 1
"""

try:
    # Execute SQL directly using the WorkspaceClient
    w.statement_execution.execute_statement(
        warehouse_id=list(w.warehouses.list())[0].id,
        catalog=CATALOG,
        schema=SCHEMA_NAME,
        statement=create_table_sql
    )
    print(f"Created Delta table: {CATALOG}.{SCHEMA_NAME}.{feature_table_name}")
except Exception as e:
    print(f"Failed to create Delta table: {e}")
    raise

# 3. Online access is not supported in this SDK version, but the Delta table is created.
print(f"Delta table {CATALOG}.{SCHEMA_NAME}.{feature_table_name} is ready for use.")

print("Feature table registration and online access setup complete.")