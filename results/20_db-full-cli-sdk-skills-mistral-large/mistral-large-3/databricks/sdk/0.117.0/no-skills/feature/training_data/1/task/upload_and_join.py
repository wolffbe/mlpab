#!/usr/bin/env python3
"""
Uploads CSV files to Unity Catalog tables and joins them to create a training dataset.
"""

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog

# Environment variables
SCHEMA_NAME = os.environ["MLPAB_DATABRICKS_SCHEMA"]
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
DATA_DIR = "./data"

# Initialize WorkspaceClient
w = WorkspaceClient()

# Create schema if not exists
def create_schema():
    try:
        w.schemas.create(name=SCHEMA_NAME, catalog_name="workspace")
        print(f"Created schema: {SCHEMA_NAME}")
    except Exception as e:
        print(f"Schema may already exist or error occurred: {e}")

# Upload CSV files as tables
def upload_csv_as_table(table_name, csv_path):
    try:
        # Use Databricks SDK to create a table from CSV
        with open(csv_path, "rb") as f:
            # Upload file to DBFS (temporary)
            dbfs_path = f"/tmp/{PREFIX}_{table_name}.csv"
            w.dbfs.upload(dbfs_path, f, overwrite=True)
            
            # Create table from CSV
            w.tables.create(
                name=table_name,
                catalog_name="workspace",
                schema_name=SCHEMA_NAME,
                table_type=catalog.TableType.MANAGED,
                data_source_format=catalog.DataSourceFormat.CSV,
                columns=[
                    catalog.ColumnTypeName(name="account_id", type_name=catalog.ColumnTypeName.STRING),
                    catalog.ColumnTypeName(name="event_time", type_name=catalog.ColumnTypeName.LONG),
                    *([catalog.ColumnTypeName(name="amount", type_name=catalog.ColumnTypeName.DOUBLE)] if "transactions" in table_name else []),
                    *([catalog.ColumnTypeName(name="balance", type_name=catalog.ColumnTypeName.DOUBLE)] if "transactions" in table_name else []),
                    *([catalog.ColumnTypeName(name="credit_score", type_name=catalog.ColumnTypeName.INT)] if "profiles" in table_name else []),
                    *([catalog.ColumnTypeName(name="tier", type_name=catalog.ColumnTypeName.STRING)] if "profiles" in table_name else []),
                    *([catalog.ColumnTypeName(name="sessions_7d", type_name=catalog.ColumnTypeName.INT)] if "activity" in table_name else []),
                    *([catalog.ColumnTypeName(name="health_score", type_name=catalog.ColumnTypeName.DOUBLE)] if "account_health" in table_name else []),
                    *([catalog.ColumnTypeName(name="label_time", type_name=catalog.ColumnTypeName.LONG), 
                       catalog.ColumnTypeName(name="churned", type_name=catalog.ColumnTypeName.INT)] if "labels" in table_name else []),
                ],
                storage_location=f"dbfs:{dbfs_path}",
                format_options={
                    "header": "true",
                    "inferSchema": "true"
                }
            )
        print(f"Created table: {table_name}")
    except Exception as e:
        print(f"Error creating table {table_name}: {e}")

# Create the training dataset by joining tables
def create_training_dataset():
    try:
        # SQL query to join tables and get the most recent feature values at or before label_time
        query = f"""
        CREATE OR REPLACE TABLE {SCHEMA_NAME}.churntrainingaf8b21_v1 AS
        WITH latest_transactions AS (
            SELECT 
                t.account_id,
                t.amount,
                t.balance,
                l.label_time
            FROM (
                SELECT 
                    account_id,
                    label_time
                FROM {SCHEMA_NAME}.labels
            ) l
            LEFT JOIN (
                SELECT 
                    account_id,
                    amount,
                    balance,
                    event_time
                FROM (
                    SELECT 
                        account_id,
                        amount,
                        balance,
                        event_time,
                        ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY event_time DESC) as rn
                    FROM (
                        SELECT * FROM {SCHEMA_NAME}.transactions
                        UNION ALL
                        SELECT * FROM {SCHEMA_NAME}.transactions_late
                    )
                )
                WHERE rn = 1
            ) t
            ON l.account_id = t.account_id AND t.event_time <= l.label_time
        ),
        latest_profiles AS (
            SELECT 
                p.account_id,
                p.credit_score,
                p.tier,
                l.label_time
            FROM (
                SELECT 
                    account_id,
                    label_time
                FROM {SCHEMA_NAME}.labels
            ) l
            LEFT JOIN (
                SELECT 
                    account_id,
                    credit_score,
                    tier,
                    event_time
                FROM (
                    SELECT 
                        account_id,
                        credit_score,
                        tier,
                        event_time,
                        ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY event_time DESC) as rn
                    FROM {SCHEMA_NAME}.profiles
                )
                WHERE rn = 1
            ) p
            ON l.account_id = p.account_id AND p.event_time <= l.label_time
        ),
        latest_activity AS (
            SELECT 
                a.account_id,
                a.sessions_7d,
                l.label_time
            FROM (
                SELECT 
                    account_id,
                    label_time
                FROM {SCHEMA_NAME}.labels
            ) l
            LEFT JOIN (
                SELECT 
                    account_id,
                    sessions_7d,
                    event_time
                FROM (
                    SELECT 
                        account_id,
                        sessions_7d,
                        event_time,
                        ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY event_time DESC) as rn
                    FROM {SCHEMA_NAME}.activity
                )
                WHERE rn = 1
            ) a
            ON l.account_id = a.account_id AND a.event_time <= l.label_time
        ),
        latest_health AS (
            SELECT 
                h.account_id,
                h.health_score,
                l.label_time
            FROM (
                SELECT 
                    account_id,
                    label_time
                FROM {SCHEMA_NAME}.labels
            ) l
            LEFT JOIN (
                SELECT 
                    account_id,
                    health_score,
                    event_time
                FROM (
                    SELECT 
                        account_id,
                        health_score,
                        event_time,
                        ROW_NUMBER() OVER (PARTITION BY account_id ORDER BY event_time DESC) as rn
                    FROM {SCHEMA_NAME}.account_health
                )
                WHERE rn = 1
            ) h
            ON l.account_id = h.account_id AND h.event_time <= l.label_time
        )
        SELECT 
            l.account_id,
            l.label_time,
            t.amount,
            t.balance,
            p.credit_score,
            p.tier,
            a.sessions_7d,
            h.health_score,
            l.churned
        FROM {SCHEMA_NAME}.labels l
        LEFT JOIN latest_transactions t ON l.account_id = t.account_id AND l.label_time = t.label_time
        LEFT JOIN latest_profiles p ON l.account_id = p.account_id AND l.label_time = p.label_time
        LEFT JOIN latest_activity a ON l.account_id = a.account_id AND l.label_time = a.label_time
        LEFT JOIN latest_health h ON l.account_id = h.account_id AND l.label_time = h.label_time
        """
        
        # Execute the query
        w.statement_execution.execute_statement(
            catalog="workspace",
            schema=SCHEMA_NAME,
            statement=query,
            warehouse_id=w.warehouses.list()[0].id  # Use the first available warehouse
        )
        print("Training dataset created: churntrainingaf8b21_v1")
        
        # Register the dataset as a versioned model
        w.model_versions.create(
            name=f"{SCHEMA_NAME}.churntrainingaf8b21",
            source=f"{SCHEMA_NAME}.churntrainingaf8b21_v1",
            version="1"
        )
        print("Registered model version: churntrainingaf8b21, version 1")
        
    except Exception as e:
        print(f"Error creating training dataset: {e}")

if __name__ == "__main__":
    create_schema()
    
    # Upload all CSV files
    csv_files = {
        "transactions": f"{DATA_DIR}/transactions.csv",
        "transactions_late": f"{DATA_DIR}/transactions_late.csv",
        "profiles": f"{DATA_DIR}/profiles.csv",
        "activity": f"{DATA_DIR}/activity.csv",
        "account_health": f"{DATA_DIR}/account_health.csv",
        "labels": f"{DATA_DIR}/labels.csv"
    }
    
    for table_name, csv_path in csv_files.items():
        upload_csv_as_table(table_name, csv_path)
    
    # Create the training dataset
    create_training_dataset()