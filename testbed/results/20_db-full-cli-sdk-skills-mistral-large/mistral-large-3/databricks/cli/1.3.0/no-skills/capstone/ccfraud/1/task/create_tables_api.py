import os
import requests
import json

# Databricks configuration
host = os.getenv("DATABRICKS_HOST")
token = os.getenv("DATABRICKS_TOKEN")
warehouse_id = "4dfab06c923fe3cc"
schema = os.getenv("MLPAB_DATABRICKS_SCHEMA")
catalog = "workspace"

# API endpoints
statements_url = f"{host}/api/2.0/sql/statements"
headers = {
    "Authorization": f"Bearer {token}",
    "Content-Type": "application/json"
}

# Create transactions table
transactions_stmt = {
    "warehouse_id": warehouse_id,
    "catalog": catalog,
    "schema": schema,
    "statement": f"CREATE TABLE transactions USING CSV OPTIONS (path '/dbfs{os.getcwd()}/data/transactions.csv', header 'true', inferSchema 'true')"
}

response = requests.post(statements_url, headers=headers, data=json.dumps(transactions_stmt))
print("Transactions table creation:", response.json())

# Create score_transactions table
score_stmt = {
    "warehouse_id": warehouse_id,
    "catalog": catalog,
    "schema": schema,
    "statement": f"CREATE TABLE score_transactions USING CSV OPTIONS (path '/dbfs{os.getcwd()}/data/score_transactions.csv', header 'true', inferSchema 'true')"
}

response = requests.post(statements_url, headers=headers, data=json.dumps(score_stmt))
print("Score transactions table creation:", response.json())

# Verify tables
verify_stmt = {
    "warehouse_id": warehouse_id,
    "catalog": catalog,
    "schema": schema,
    "statement": "SHOW TABLES"
}

response = requests.post(statements_url, headers=headers, data=json.dumps(verify_stmt))
print("Tables in schema:", response.json())