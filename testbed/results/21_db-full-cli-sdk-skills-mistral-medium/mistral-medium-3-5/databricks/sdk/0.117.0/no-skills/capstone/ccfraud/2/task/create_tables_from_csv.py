#!/usr/bin/env python3
"""
Create tables from CSV files using SQL INSERT statements.
"""
import csv
import os
from databricks.sdk import WorkspaceClient
import time

# Configuration
SCHEMA = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpabf21a49')
PREFIX = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpabf21a49')

FEATURE_GROUP = 'cctxn0802aa'
TRAINING_DATASET = 'cctd0802aa'
MODEL_NAME = 'ccmodel0802aa'
PREDICTIONS_TABLE = 'ccpred0802aa'

wc = WorkspaceClient()
WAREHOUSE_ID = 'a832b544eb7dc3fe'

def execute_sql(statement, warehouse_id=WAREHOUSE_ID):
    """Execute a SQL statement and return the result."""
    print(f"Executing SQL: {statement[:100]}...")
    result = wc.statement_execution.execute_statement(
        statement=statement,
        warehouse_id=warehouse_id,
        wait_timeout="10s"
    )
    
    # Wait for completion
    while True:
        status = wc.statement_execution.get_statement(result.statement_id)
        if status.status.state in ['SUCCEEDED', 'FAILED', 'CANCELED']:
            if status.status.state == 'FAILED':
                error_msg = getattr(status.status, 'error', 'Unknown error')
                print(f"SQL execution failed: {error_msg}")
                raise Exception(f"SQL execution failed: {error_msg}")
            return status
        time.sleep(1)

def create_table_from_csv(table_name, csv_file, schema=SCHEMA):
    """Create a table from a CSV file by reading it locally and generating INSERT statements."""
    print(f"Creating table {schema}.{table_name} from {csv_file}")
    
    # Read CSV file
    with open(csv_file, 'r') as f:
        reader = csv.DictReader(f)
        rows = list(reader)
    
    if not rows:
        print(f"No data in {csv_file}")
        return
    
    # Get column names and types
    columns = list(rows[0].keys())
    print(f"Columns: {columns}")
    
    # Create table
    column_defs = []
    for col in columns:
        # Try to infer type from first few rows
        sample_values = [row[col] for row in rows[:10] if row[col].strip()]
        if not sample_values:
            column_defs.append(f"{col} STRING")
            continue
        
        # Check if numeric
        try:
            float(sample_values[0].replace(',', ''))
            column_defs.append(f"{col} DOUBLE")
        except ValueError:
            # Check if integer
            try:
                int(sample_values[0])
                column_defs.append(f"{col} BIGINT")
            except ValueError:
                # Check if timestamp
                if any('T' in val and 'Z' in val for val in sample_values):
                    column_defs.append(f"{col} TIMESTAMP")
                else:
                    column_defs.append(f"{col} STRING")
    
    create_sql = f"CREATE OR REPLACE TABLE {schema}.{table_name} ({', '.join(column_defs)})"
    execute_sql(create_sql)
    print(f"Table {schema}.{table_name} created")
    
    # Insert data in batches
    batch_size = 1000
    for i in range(0, len(rows), batch_size):
        batch = rows[i:i+batch_size]
        
        # Generate INSERT statement
        cols = ', '.join(columns)
        values_list = []
        for row in batch:
            values = []
            for col in columns:
                val = row.get(col, '')
                if val == '' or val is None:
                    values.append('NULL')
                elif column_defs[columns.index(col)].endswith('STRING'):
                    # Escape single quotes
                    val = val.replace("'", "''")
                    values.append(f"'{val}'")
                elif column_defs[columns.index(col)].endswith('TIMESTAMP'):
                    values.append(f"TIMESTAMP '{val}'")
                else:
                    values.append(str(val))
            values_list.append(f"({', '.join(values)})")
        
        insert_sql = f"INSERT INTO {schema}.{table_name} ({cols}) VALUES {', '.join(values_list)}"
        execute_sql(insert_sql)
        print(f"Inserted batch {i//batch_size + 1}: {len(batch)} rows")
    
    print(f"Total rows inserted: {len(rows)}")

def main():
    print("Starting table creation from CSV files...")
    
    # Create raw tables
    print("\n=== Creating raw tables ===")
    create_table_from_csv('raw_transactions', 'data/transactions.csv')
    create_table_from_csv('raw_score_transactions', 'data/score_transactions.csv')
    
    print("\n=== Raw tables created ===")

if __name__ == "__main__":
    main()
