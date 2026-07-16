#!/usr/bin/env python3
"""
Register a feature table named `transactions4adadd` (version 1) on Databricks,
with record key `row_id` and event-time column `event_time` (epoch milliseconds).
Load the full export into it, deduplicating overlapping rows.
Enable online access for low-latency lookup.
"""

import os
import tempfile
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog, ml
from databricks.sdk.errors import NotFound

# Environment variables
SCHEMA = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # workspace.<run-id>
PREFIX = os.getenv("MLPAB_DATABRICKS_PREFIX")  # mlpab<run-id>
VOLUME_NAME = f"{PREFIX}_volume"
FEATURE_TABLE_NAME = "transactions4adadd"
FEATURE_TABLE_FULL_NAME = f"{SCHEMA}.{FEATURE_TABLE_NAME}"

# Initialize the WorkspaceClient
w = WorkspaceClient()

# Create a volume to store the CSV files
try:
    w.volumes.create(
        catalog_name="workspace",
        schema_name=SCHEMA.split(".")[1],
        name=VOLUME_NAME,
        volume_type=catalog.VolumeType.MANAGED,
    )
except Exception as e:
    print(f"Volume creation failed (may already exist): {e}")

# Upload CSV files to the volume
volume_path = f"/Volumes/workspace/{SCHEMA.split('.')[1]}/{VOLUME_NAME}"
for file_name in ["transactions_export_1.csv", "transactions_export_2.csv"]:
    file_path = f"./data/{file_name}"
    with open(file_path, "rb") as f:
        w.files.upload(f"{volume_path}/{file_name}", f, overwrite=True)

# Create a Delta table from the CSV files, deduplicating by row_id
spark_sql = f"""
CREATE TABLE IF NOT EXISTS {FEATURE_TABLE_FULL_NAME} (
    row_id STRING,
    account_id STRING,
    event_time BIGINT,
    amount DOUBLE,
    category STRING
) USING DELTA;

-- Load data from both CSV files, deduplicating by row_id
CREATE OR REPLACE TEMP VIEW transactions_temp AS
SELECT * FROM (
    SELECT * FROM read.csv("{volume_path}/transactions_export_1.csv", header => true, inferSchema => true)
    UNION ALL
    SELECT * FROM read.csv("{volume_path}/transactions_export_2.csv", header => true, inferSchema => true)
);

-- Deduplicate by row_id (keep the first occurrence)
INSERT INTO {FEATURE_TABLE_FULL_NAME}
SELECT * FROM (
    SELECT *, ROW_NUMBER() OVER (PARTITION BY row_id ORDER BY event_time) as rn
    FROM transactions_temp
)
WHERE rn = 1;
"""

# Execute the SQL
warehouses = list(w.warehouses.list())
if not warehouses:
    raise RuntimeError("No warehouses available.")
warehouse_id = warehouses[0].id

for statement in spark_sql.split(";"):
    statement = statement.strip()
    if not statement:
        continue
    w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        catalog="workspace",
        schema=SCHEMA.split(".")[1],
        statement=statement,
        wait_timeout="50s",
    )

# Create an online store
try:
    from databricks.sdk.service.ml import OnlineStore
    
    # Check if the online store already exists
    online_stores = list(w.feature_store.list_online_stores())
    if not any(store.name == f"{PREFIX}-online-store" for store in online_stores):
        w.feature_store.create_online_store(
            online_store=OnlineStore(
                name=f"{PREFIX}-online-store",
                capacity="CU_1",
            ),
        )
        print(f"Online store {PREFIX}-online-store created successfully.")
    else:
        print(f"Online store {PREFIX}-online-store already exists.")
except Exception as e:
    print(f"Online store creation failed: {e}")

# Enable online access for the feature table
try:
    from databricks.sdk.service.ml import MaterializedFeature, OnlineStoreConfig
    
    w.feature_engineering.create_materialized_feature(
        materialized_feature=MaterializedFeature(
            feature_name=FEATURE_TABLE_FULL_NAME,
            is_online=True,
            online_store_config=OnlineStoreConfig(
                catalog_name="workspace",
                schema_name=SCHEMA.split(".")[1],
                table_name_prefix=f"{FEATURE_TABLE_NAME}_online",
                online_store_name=f"{PREFIX}-online-store",
            ),
        ),
    )
    print(f"Online access for {FEATURE_TABLE_FULL_NAME} enabled successfully.")
except Exception as e:
    print(f"Online access enablement failed: {e}")

print("Feature table registration and online access setup complete.")