#!/usr/bin/env python3
"""
Create a feature table named `profiles395e7c`, version 1, in the specified schema.
Load data from data/features.csv, enable online access, and retrieve feature vectors
for the keys in data/lookup_keys.txt via the online store.
"""

import os
import json
import io
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog
from databricks.sdk.service import sql

# Environment variables
SCHEMA = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # workspace.<run-id>
PREFIX = os.getenv("MLPAB_DATABRICKS_PREFIX")  # mlpab05c624

# Feature table details
FEATURE_TABLE_NAME = "profiles395e7c"
RECORD_KEY = "account_id"

# Paths
data_path = "data/features.csv"
lookup_keys_path = "data/lookup_keys.txt"
output_path = "submission/answers.json"

# Initialize WorkspaceClient
w = WorkspaceClient()


def read_lookup_keys():
    """Read account_ids from lookup_keys.txt."""
    with open(lookup_keys_path, "r") as f:
        return [line.strip() for line in f if line.strip()]


def create_feature_table():
    """Create a feature table in Unity Catalog and enable online access."""
    full_table_name = f"{SCHEMA}.{FEATURE_TABLE_NAME}"
    
    # Check if the table already exists
    try:
        w.tables.get(full_table_name)
        print(f"Table {full_table_name} already exists. Skipping creation.")
        return full_table_name, f"{PREFIX}_online_{FEATURE_TABLE_NAME}"
    except Exception:
        pass
    
    # Create a temporary volume to stage the CSV
    volume_name = f"{PREFIX}_temp_volume"
    full_volume_name = f"{SCHEMA}.{volume_name}"
    
    try:
        w.volumes.create(
            catalog_name=SCHEMA.split(".")[0],
            schema_name=SCHEMA.split(".")[1],
            name=volume_name,
            volume_type=catalog.VolumeType.MANAGED
        )
    except Exception as e:
        print(f"Volume may already exist: {e}")
    
    # Upload the CSV to the volume
    volume_path = f"/Volumes/{SCHEMA.split('.')[0]}/{SCHEMA.split('.')[1]}/{volume_name}"
    volume_file_path = f"{volume_path}/features.csv"
    
    # Read the CSV content and write to the volume
    with open(data_path, "r") as f:
        csv_content = f.read()
    
    csv_bytes = io.BytesIO(csv_content.encode('utf-8'))
    w.files.upload(volume_file_path, csv_bytes, overwrite=True)
    
    # Get a SQL warehouse
    warehouses = list(w.warehouses.list())
    if not warehouses:
        raise RuntimeError("No SQL warehouses available in the workspace.")
    
    # Create the Delta table
    create_table_sql = f"""
    CREATE TABLE {full_table_name} (
        account_id STRING,
        f1 DOUBLE,
        f2 DOUBLE,
        f3 DOUBLE,
        f4 DOUBLE
    )
    USING DELTA
    """
    
    # Execute the Delta table SQL
    result = w.statement_execution.execute_statement(
        warehouse_id=warehouses[0].id,
        catalog=SCHEMA.split(".")[0],
        schema=SCHEMA.split(".")[1],
        statement=create_table_sql
    )
    
    # Poll for completion
    statement_id = result.statement_id
    while True:
        status = w.statement_execution.get_statement(statement_id)
        if status.status.state == sql.StatementState.SUCCEEDED:
            break
        elif status.status.state in [sql.StatementState.FAILED, sql.StatementState.CANCELED]:
            error_msg = getattr(status.status, 'message', 'No error message available')
            raise RuntimeError(f"Delta table creation failed: {error_msg}")
        time.sleep(2)
    
    # Load data into the Delta table using COPY INTO
    copy_into_sql = f"""
    COPY INTO {full_table_name}
    FROM '{volume_file_path}'
    FILEFORMAT = CSV
    FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'true')
    """
    
    # Execute the COPY INTO SQL
    result = w.statement_execution.execute_statement(
        warehouse_id=warehouses[0].id,
        catalog=SCHEMA.split(".")[0],
        schema=SCHEMA.split(".")[1],
        statement=copy_into_sql
    )
    
    # Poll for completion
    statement_id = result.statement_id
    while True:
        status = w.statement_execution.get_statement(statement_id)
        if status.status.state == sql.StatementState.SUCCEEDED:
            break
        elif status.status.state in [sql.StatementState.FAILED, sql.StatementState.CANCELED]:
            error_msg = getattr(status.status, 'message', 'No error message available')
            raise RuntimeError(f"COPY INTO failed: {error_msg}")
        time.sleep(2)
    
    # Enable online table for low-latency access
    w.online_tables.create(
        primary_key_columns=[RECORD_KEY],
        source_table_full_name=full_table_name,
        run_triggered=True
    )
    
    # Use the full table name as the online table identifier
    online_table_name = full_table_name
    
    return full_table_name, online_table_name


def retrieve_online_features(account_ids, online_table_name):
    """Retrieve feature vectors for the given account_ids via the online store."""
    vectors = {}
    for account_id in account_ids:
        try:
            row = w.online_tables.query(
                online_table_name=online_table_name,
                keys=[account_id]
            )
            if row and row.data_array:
                # Extract feature values in order f1, f2, f3, f4
                feature_values = [
                    float(row.data_array[0].values[1]),  # f1
                    float(row.data_array[0].values[2]),  # f2
                    float(row.data_array[0].values[3]),  # f3
                    float(row.data_array[0].values[4])   # f4
                ]
                vectors[account_id] = feature_values
        except Exception as e:
            print(f"Error retrieving features for {account_id}: {e}")
    
    return vectors


def main():
    # Create the feature table and enable online access
    _, online_table_name = create_feature_table()
    
    # Read lookup keys
    account_ids = read_lookup_keys()
    
    # Retrieve feature vectors via online store
    vectors = retrieve_online_features(account_ids, online_table_name)
    
    # Write the output
    os.makedirs("submission", exist_ok=True)
    with open(output_path, "w") as f:
        json.dump({"vectors": vectors}, f, indent=2)
    
    print(f"Feature vectors written to {output_path}")


if __name__ == "__main__":
    main()