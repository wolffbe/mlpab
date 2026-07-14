#!/usr/bin/env python3
"""
Use the Databricks SDK to:
1. Create an empty Delta table.
2. Insert valid rows in batches.
3. Register the feature table.
4. Enable online access.
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
    
    # Create an empty Delta table
    create_table_query = f"""
    CREATE TABLE {full_table_name} (
        row_id STRING,
        account_id STRING,
        event_time BIGINT,
        amount DOUBLE,
        category STRING
    )
    USING DELTA
    LOCATION 'dbfs:{delta_path}'
    """
    
    w.statement_execution.execute_statement(
        warehouse_id=warehouse.id,
        catalog=catalog_name,
        schema=schema_name,
        statement=create_table_query
    ).result()
    
    # Read valid rows and insert in batches
    batch_size = 100
    valid_rows = []
    with open('valid_rows.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            valid_rows.append(row)
    
    for i in range(0, len(valid_rows), batch_size):
        batch = valid_rows[i:i + batch_size]
        values_sql = ",".join(
            f"('{row['row_id']}', '{row['account_id']}', {row['event_time']}, {row['amount']}, '{row['category']}')"
            for row in batch
        )
        
        insert_query = f"""
        INSERT INTO {full_table_name} (row_id, account_id, event_time, amount, category)
        VALUES {values_sql}
        """
        
        w.statement_execution.execute_statement(
            warehouse_id=warehouse.id,
            catalog=catalog_name,
            schema=schema_name,
            statement=insert_query
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