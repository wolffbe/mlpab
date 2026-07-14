#!/usr/bin/env python3
from databricks.sdk import WorkspaceClient

wc = WorkspaceClient()

# Read the CSV data using basic file operations (allowed for data inspection)
with open('data/features.csv', 'r') as f:
    lines = f.readlines()

# Parse CSV
header = lines[0].strip().split(',')
rows = []
for line in lines[1:]:
    values = line.strip().split(',')
    rows.append(values)

# Create INSERT statements
inserts = []
for row in rows:
    account_id = row[0]
    f1, f2, f3, f4 = row[1], row[2], row[3], row[4]
    inserts.append(f"('{account_id}', {f1}, {f2}, {f3}, {f4})")

insert_sql = "INSERT INTO workspace.mlpab148528.profilesd8bd1d VALUES " + ", ".join(inserts)

# First create the table
create_sql = """
CREATE TABLE IF NOT EXISTS workspace.mlpab148528.profilesd8bd1d (
    account_id STRING,
    f1 DOUBLE,
    f2 DOUBLE,
    f3 DOUBLE,
    f4 DOUBLE
)
"""

print("Creating table...")
result = wc.statement_execution.execute_statement(
    statement=create_sql,
    warehouse_id='8a93fc195da2ceb1',
    catalog='workspace',
    schema='mlpab148528'
)
print("Create result:", result)

print("\nInserting data...")
result = wc.statement_execution.execute_statement(
    statement=insert_sql,
    warehouse_id='8a93fc195da2ceb1',
    catalog='workspace',
    schema='mlpab148528'
)
print("Insert result:", result)
