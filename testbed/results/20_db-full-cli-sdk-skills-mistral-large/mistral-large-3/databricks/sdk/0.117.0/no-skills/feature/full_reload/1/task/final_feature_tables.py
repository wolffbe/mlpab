#!/usr/bin/env python3
"""
Register and reload feature tables for customersc23945.

Steps:
1. Create a Delta table for initial_export.csv using SQL and register as version 1.
2. Drop the table and create a new Delta table for new_export.csv using SQL, register as version 2.
3. Enable online access for version 2.
"""

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode
from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec

def main():
    # Initialize WorkspaceClient
    w = WorkspaceClient()
    
    # Per-run namespace
    catalog_schema = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # e.g., "workspace.mlpab054f9d"
    prefix = os.getenv("MLPAB_DATABRICKS_PREFIX")  # e.g., "mlpab054f9d"
    
    if not catalog_schema or not prefix:
        raise ValueError("MLPAB_DATABRICKS_SCHEMA and MLPAB_DATABRICKS_PREFIX must be set.")
    
    catalog = catalog_schema.split(".")[0]
    schema = catalog_schema.split(".")[1]
    
    # Feature table name
    feature_table_name = f"{catalog_schema}.customersc23945"
    
    # Step 1: Create Delta table for initial_export.csv and register as version 1
    print("Creating Delta table for version 1...")
    initial_export_path = os.path.abspath("data/initial_export.csv")
    
    # Create Delta table using SQL
    create_sql_v1 = f"""
    CREATE TABLE {feature_table_name} (
        row_id STRING,
        name STRING,
        balance_eur DOUBLE,
        updated_at LONG
    )
    USING DELTA
    LOCATION 'dbfs:/FileStore/{prefix}_customersc23945_v1'
    """
    w.statement_execution.execute_statement(
        warehouse_id="4dfab06c923fe3cc",  # mlpab-grader
        catalog=catalog,
        schema=schema,
        statement=create_sql_v1,
    )
    
    # Load initial data
    load_sql_v1 = f"""
    COPY INTO {feature_table_name}
    FROM '{initial_export_path}'
    FILEFORMAT = CSV
    FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'true')
    """
    w.statement_execution.execute_statement(
        warehouse_id="4dfab06c923fe3cc",  # mlpab-grader
        catalog=catalog,
        schema=schema,
        statement=load_sql_v1,
    )
    
    # Register as feature table (version 1)
    print("Registering version 1 as feature table...")
    w.feature_store.publish_table(
        source_table_name=feature_table_name,
        publish_spec=PublishSpec(
            online_store=f"{prefix}_online_store",
            online_table_name=f"{prefix}_customersc23945_v1_online",
            publish_mode=PublishSpecPublishMode.SNAPSHOT,
        ),
    )
    
    print(f"Version 1 registered: {feature_table_name}")
    
    # Step 2: Drop the table and create a new Delta table for new_export.csv, register as version 2
    print("Dropping version 1 and creating Delta table for version 2...")
    drop_sql = f"DROP TABLE IF EXISTS {feature_table_name}"
    w.statement_execution.execute_statement(
        warehouse_id="4dfab06c923fe3cc",  # mlpab-grader
        catalog=catalog,
        schema=schema,
        statement=drop_sql,
    )
    
    new_export_path = os.path.abspath("data/reload/new_export.csv")
    
    # Create Delta table for new schema using SQL
    create_sql_v2 = f"""
    CREATE TABLE {feature_table_name} (
        row_id STRING,
        full_name STRING,
        balance DOUBLE,
        currency STRING,
        updated_at LONG
    )
    USING DELTA
    LOCATION 'dbfs:/FileStore/{prefix}_customersc23945_v2'
    """
    w.statement_execution.execute_statement(
        warehouse_id="4dfab06c923fe3cc",  # mlpab-grader
        catalog=catalog,
        schema=schema,
        statement=create_sql_v2,
    )
    
    # Load new data
    load_sql_v2 = f"""
    COPY INTO {feature_table_name}
    FROM '{new_export_path}'
    FILEFORMAT = CSV
    FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'true')
    """
    w.statement_execution.execute_statement(
        warehouse_id="4dfab06c923fe3cc",  # mlpab-grader
        catalog=catalog,
        schema=schema,
        statement=load_sql_v2,
    )
    
    # Register as feature table (version 2)
    print("Registering version 2 as feature table...")
    w.feature_store.publish_table(
        source_table_name=feature_table_name,
        publish_spec=PublishSpec(
            online_store=f"{prefix}_online_store",
            online_table_name=f"{prefix}_customersc23945_v2_online",
            publish_mode=PublishSpecPublishMode.SNAPSHOT,
        ),
    )
    
    print(f"Version 2 registered: {feature_table_name}")
    
    # Step 3: Enable online access for version 2
    print("Enabling online access for version 2...")
    
    # Create online table for version 2
    online_table = w.online_tables.create(
        table=OnlineTable(
            name=f"{prefix}_customersc23945_online",
            spec=OnlineTableSpec(
                source_table_full_name=feature_table_name,
                primary_key_columns=["row_id"],
                timeseries_key="updated_at",
            ),
        )
    )
    
    print(f"Online access enabled for version 2: {online_table.name}")
    
    print("Task completed successfully.")


if __name__ == "__main__":
    main()