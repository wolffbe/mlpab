#!/usr/bin/env python3
"""
Use the Databricks SDK to:
1. Upload the valid rows to DBFS.
2. Create a Delta table using a SQL query.
3. Register the feature table.
4. Enable online access.
"""

import os
import tempfile
import csv
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import *


def main():
    # Initialize Databricks WorkspaceClient
    w = WorkspaceClient()
    
    # Schema details
    catalog_name = "workspace"
    schema_name = os.environ["MLPAB_DATABRICKS_SCHEMA"]
    table_name = "events45bd4b"
    full_table_name = f"{catalog_name}.{schema_name}.{table_name}"
    
    # Prefix for DBFS paths
    prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]
    dbfs_path = f"/tmp/{prefix}_valid_events.csv"
    delta_path = f"/tmp/{prefix}_events45bd4b_delta"
    
    # Read valid rows from the filtered CSV
    valid_rows = []
    with open('valid_rows.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            valid_rows.append(row)
    
    # Create a temporary file with the valid rows
    with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
        fieldnames = ['row_id', 'account_id', 'event_time', 'amount', 'category']
        writer = csv.DictWriter(tmp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(valid_rows)
        tmp_path = tmp.name
    
    # Upload the valid rows CSV to DBFS
    w.dbfs.upload(tmp_path, dbfs_path, overwrite=True)
    
    # Create a SQL warehouse if none exists
    warehouses = list(w.warehouses.list())
    if not warehouses:
        warehouse = w.warehouses.create(
            name=f"{prefix}_warehouse",
            cluster_size="2X-Small",
            auto_stop_mins=10,
            max_num_clusters=1
        ).result()
    else:
        warehouse = warehouses[0]
    
    # Create a Delta table from the CSV using a SQL query
    spark_sql_query = f"""
    CREATE TABLE {full_table_name} USING DELTA LOCATION 'dbfs:{delta_path}' AS
    SELECT
        row_id,
        account_id,
        CAST(event_time AS BIGINT) AS event_time,
        CAST(amount AS DOUBLE) AS amount,
        category
    FROM csv.`dbfs:{dbfs_path}`
    WHERE row_id IS NOT NULL
    """
    
    w.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        catalog=catalog_name,
        schema=schema_name,
        statement=spark_sql_query
    ).result()
    
    # Register the feature table
    w.tables.create(
        name=table_name,
        catalog_name=catalog_name,
        schema_name=schema_name,
        table_type=TableType.MANAGED,
        data_source_format=DataSourceFormat.DELTA,
        storage_location=f"dbfs:{delta_path}",
        table_constraints=[
            TableConstraint(
                primary_key_constraint=PrimaryKeyConstraint(name="row_id_pk", columns=["row_id"])
            )
        ],
        properties={
            "delta.enableChangeDataFeed": "true",
            "pipelines.autoOptimize.managed": "true"
        }
    )
    
    # Enable online table for real-time access
    w.online_tables.create(
        name=full_table_name,
        spec=OnlineTableSpec(
            source_table_full_name=full_table_name,
            primary_key_columns=["row_id"],
            timeseries_key="event_time"
        )
    )
    
    print("Feature table registered and online access enabled.")


if __name__ == "__main__":
    main()