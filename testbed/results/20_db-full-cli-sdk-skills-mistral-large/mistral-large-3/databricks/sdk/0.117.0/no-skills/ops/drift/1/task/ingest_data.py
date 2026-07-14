#!/usr/bin/env python3
"""
Ingest data/features.csv into a Unity Catalog table using CTAS.
"""
import os
from databricks.sdk import WorkspaceClient

# Initialize WorkspaceClient
w = WorkspaceClient()

# Environment variables
schema_name = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # workspace.<run-id>
catalog_name, schema = schema_name.split(".")
table_name = f"{os.getenv('MLPAB_DATABRICKS_PREFIX')}_features"

# Create the table using CTAS
print(f"Creating table {catalog_name}.{schema}.{table_name} using CTAS...")
ctas_sql = f"""
CREATE TABLE {catalog_name}.{schema}.{table_name} AS
SELECT
    entity_id,
    CAST(event_time AS TIMESTAMP) AS event_time,
    CAST(f1 AS FLOAT) AS f1,
    CAST(f2 AS FLOAT) AS f2,
    CAST(f3 AS FLOAT) AS f3,
    CAST(f4 AS FLOAT) AS f4,
    CAST(f5 AS FLOAT) AS f5,
    CAST(f6 AS FLOAT) AS f6
FROM
    csv.`dbfs:/mnt/data/features.csv`
WHERE
    entity_id IS NOT NULL
    AND event_time IS NOT NULL
"""

w.statement_execution.execute_statement(
    catalog=catalog_name,
    schema=schema,
    warehouse_id=list(w.warehouses.list())[0].id,
    statement=ctas_sql
)

print("Data ingestion complete.")