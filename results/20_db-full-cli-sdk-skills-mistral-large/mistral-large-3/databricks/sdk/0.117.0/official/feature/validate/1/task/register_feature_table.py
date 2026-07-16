#!/usr/bin/env python3
"""
Filter events.csv to keep only rows that satisfy the data contract.
Write submission/answers.json with rejected row IDs.
Register a feature table named `events45bd4b` (version 1) in the schema `workspace.mlpab7c6cbb`
with record key `row_id` and event-time column `event_time`.
Enable online/real-time access for the table.
"""

import csv
import json
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import *

# Data contract rules
VALID_CATEGORIES = {"grocery", "travel", "salary", "rent", "other"}


def is_valid_row(row):
    """Check if a row satisfies all contract rules."""
    try:
        amount = row['amount']
        if amount == '':
            return False
        amount = float(amount)
        if not (0 <= amount <= 10000):
            return False
        if row['category'] not in VALID_CATEGORIES:
            return False
        return True
    except (ValueError, KeyError):
        return False


def main():
    # Read and filter events
    valid_rows = []
    rejected_row_ids = []
    
    with open('data/events.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            if is_valid_row(row):
                valid_rows.append(row)
            else:
                rejected_row_ids.append(row['row_id'])
    
    # Write rejected row IDs to submission/answers.json
    os.makedirs('submission', exist_ok=True)
    with open('submission/answers.json', 'w') as f:
        json.dump({"rejected": rejected_row_ids}, f, indent=2)
    
    # Initialize Databricks WorkspaceClient
    w = WorkspaceClient()
    
    # Schema details
    catalog_name = "workspace"
    schema_name = os.environ["MLPAB_DATABRICKS_SCHEMA"]
    table_name = "events45bd4b"
    full_table_name = f"{catalog_name}.{schema_name}.{table_name}"
    
    # Create a temporary Delta table in DBFS
    dbfs_temp_dir = f"/tmp/{os.environ['MLPAB_DATABRICKS_PREFIX']}_events45bd4b"
    
    # Create a DataFrame from valid rows and write to Delta
    from databricks.sdk.runtime import spark
    
    if valid_rows:
        valid_rows_rdd = spark.sparkContext.parallelize([
            (row['row_id'], row['account_id'], int(row['event_time']), float(row['amount']), row['category'])
            for row in valid_rows
        ])
        valid_df = spark.createDataFrame(
            valid_rows_rdd,
            schema=["row_id", "account_id", "event_time", "amount", "category"]
        )
        
        # Write to Delta
        valid_df.write.format("delta").mode("overwrite").save(f"dbfs:{dbfs_temp_dir}")
        
        # Register the feature table
        w.tables.create(
            name=table_name,
            catalog_name=catalog_name,
            schema_name=schema_name,
            table_type=TableType.MANAGED,
            data_source_format=DataSourceFormat.DELTA,
            storage_location=f"dbfs:{dbfs_temp_dir}",
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