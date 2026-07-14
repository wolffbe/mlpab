"""Script to load predictions into a feature table on Databricks."""
import os
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog

# Environment variables
SCHEMA = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # e.g., workspace.<run-id>
PREFIX = os.getenv("MLPAB_DATABRICKS_PREFIX")
TABLE_NAME = "predictionsac536a"

# Initialize the WorkspaceClient
w = WorkspaceClient()

# Read predictions.csv
predictions = pd.read_csv("./data/predictions.csv")
print(f"Loaded predictions with {len(predictions)} rows")

# Extract schema and table names
catalog_name = "workspace"
schema_name = SCHEMA.split(".")[1]

# Create the schema if it doesn't exist
try:
    w.schemas.create(name=schema_name, catalog_name=catalog_name)
    print(f"Created schema {catalog_name}.{schema_name}")
except Exception as e:
    print(f"Schema may already exist or error occurred: {e}")

# Create the table using the SDK
try:
    w.tables.create(
        name=TABLE_NAME,
        catalog_name=catalog_name,
        schema_name=schema_name,
        table_type=catalog.TableType.MANAGED,
        data_source_format=catalog.DataSourceFormat.DELTA,
        columns=[
            {"name": "row_id", "type_name": "STRING"},
            {"name": "score", "type_name": "DOUBLE"},
        ],
    )
    print(f"Created table {catalog_name}.{schema_name}.{TABLE_NAME}")
except Exception as e:
    print(f"Table may already exist or error occurred: {e}")

# Write data to the table
for record in predictions.to_dict("records"):
    try:
        w.table_data.insert(
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_name=TABLE_NAME,
            data=[record],
        )
    except Exception as e:
        print(f"Error inserting record: {e}")

# Enable online access for the table
try:
    w.online_tables.create(
        catalog_name=catalog_name,
        schema_name=schema_name,
        table_name=TABLE_NAME,
        primary_key_columns=["row_id"],
    )
    print(f"Enabled online access for {catalog_name}.{schema_name}.{TABLE_NAME}")
except Exception as e:
    print(f"Error enabling online access: {e}")

print("Feature table creation process completed.")