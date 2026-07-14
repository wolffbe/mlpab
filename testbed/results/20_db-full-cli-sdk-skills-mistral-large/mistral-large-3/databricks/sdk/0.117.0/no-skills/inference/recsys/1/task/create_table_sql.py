#!/usr/bin/env python3
"""
Create the feature table using SQL statement execution.
"""

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog

# Initialize Databricks WorkspaceClient
w = WorkspaceClient()

# Get schema and prefix from environment variables
full_schema_name = os.environ['MLPAB_DATABRICKS_SCHEMA']
schema_name = full_schema_name.split(".")[-1]
table_name = "recse3a36e"
full_table_name = f"{full_schema_name}.{table_name}"
volume_name = f"{os.environ['MLPAB_DATABRICKS_PREFIX']}_volume"
full_volume_name = f"{full_schema_name}.{volume_name}"

# Path to the recommendations CSV in the volume
volume_path = f"/Volumes/workspace/{schema_name}/{volume_name}/recommendations.csv"

# SQL to create the table
spark_sql = f"""
CREATE TABLE IF NOT EXISTS {full_table_name} 
USING CSV
OPTIONS (path '{volume_path}', header 'true', inferSchema 'true')
"""

# Execute the SQL statement
try:
    w.statement_execution.execute_statement(
        warehouse_id=os.environ.get("DATABRICKS_SQL_WAREHOUSE_ID"),
        catalog="workspace",
        schema=schema_name,
        statement=spark_sql
    ).result()
    print(f"Feature table {full_table_name} created.")
except Exception as e:
    print(f"Failed to create table: {e}")
    raise

# Enable online access for low-latency lookup
try:
    w.online_tables.create(
        name=full_table_name,
        spec=catalog.OnlineTableSpec(
            source_table_full_name=full_table_name,
            primary_key_columns=["rec_id"]
        )
    )
    print(f"Online table {full_table_name} created successfully.")
except Exception as e:
    if "already exists" not in str(e):
        raise

print(f"Feature table {full_table_name} is ready for online access.")