#!/usr/bin/env python3
"""
Use the Databricks SDK to:
1. Create a Delta table using a SQL query.
2. Register the feature table.
3. Enable online access.
"""

import os
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
    delta_path = f"/tmp/{prefix}_events45bd4b_delta"
    
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
    
    # Create a temporary table from the valid rows
    create_temp_table_query = f"""
    CREATE TEMPORARY TABLE temp_valid_events (
        row_id STRING,
        account_id STRING,
        event_time BIGINT,
        amount DOUBLE,
        category STRING
    )
    USING CSV
    OPTIONS (
        path 'file:{os.path.abspath('valid_rows.csv')}',
        header 'true',
        inferSchema 'false'
    )
    """
    
    w.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        catalog=catalog_name,
        schema=schema_name,
        statement=create_temp_table_query
    ).result()
    
    # Create a Delta table from the temporary table
    create_delta_table_query = f"""
    CREATE TABLE {full_table_name} USING DELTA LOCATION 'dbfs:{delta_path}' AS
    SELECT * FROM temp_valid_events
    """
    
    w.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        catalog=catalog_name,
        schema=schema_name,
        statement=create_delta_table_query
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