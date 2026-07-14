#!/usr/bin/env python3
"""
Creates the training dataset using a Databricks SQL Warehouse.
"""

import os
import requests
import json
import time

# Environment variables
SCHEMA_NAME = os.environ["MLPAB_DATABRICKS_SCHEMA"].replace(".", "_")
DATABRICKS_HOST = os.environ["DATABRICKS_HOST"]
DATABRICKS_TOKEN = os.environ["DATABRICKS_TOKEN"]
DATA_DIR = os.path.abspath("./data")

# Ensure DATABRICKS_HOST has the correct scheme
if not DATABRICKS_HOST.startswith("https://"):
    DATABRICKS_HOST = f"https://{DATABRICKS_HOST}"

# Headers for API requests
headers = {
    "Authorization": f"Bearer {DATABRICKS_TOKEN}",
    "Content-Type": "application/json"
}

# Get the first available SQL warehouse
def get_sql_warehouse_id():
    try:
        url = f"{DATABRICKS_HOST}/api/2.0/sql/warehouses"
        response = requests.get(url, headers=headers)
        if response.status_code != 200:
            raise Exception(f"Failed to list SQL warehouses: {response.text}")
        
        warehouses = response.json().get("warehouses", [])
        if not warehouses:
            raise Exception("No SQL warehouses available")
        
        return warehouses[0]["id"]
    except Exception as e:
        print(f"Error getting SQL warehouse: {e}")
        return None

# Execute a SQL query using the SQL Execution API
def execute_sql(query, warehouse_id):
    try:
        url = f"{DATABRICKS_HOST}/api/2.0/sql/statements/"
        payload = {
            "warehouse_id": warehouse_id,
            "catalog": "workspace",
            "schema": SCHEMA_NAME,
            "statement": query,
            "wait_timeout": "50s"
        }
        
        response = requests.post(url, headers=headers, data=json.dumps(payload))
        if response.status_code != 200:
            raise Exception(f"Failed to execute query: {response.text}")
        
        statement_id = response.json()["statement_id"]
        
        # Wait for completion
        while True:
            url = f"{DATABRICKS_HOST}/api/2.0/sql/statements/{statement_id}"
            response = requests.get(url, headers=headers)
            if response.status_code != 200:
                raise Exception(f"Failed to get statement status: {response.text}")
            
            status = response.json().get("status", {}).get("state", "")
            if status in ["SUCCEEDED", "FAILED", "CANCELED"]:
                break
            
            time.sleep(5)
        
        if status == "SUCCEEDED":
            print(f"Query succeeded: {query[:50]}...")
            return True
        else:
            print(f"Query failed: {response.json()}")
            return False
    except Exception as e:
        print(f"Error executing query: {e}")
        return False

# Create schema
warehouse_id = get_sql_warehouse_id()
if not warehouse_id:
    print("No SQL warehouse available")
    exit(1)

if not execute_sql(f"CREATE SCHEMA IF NOT EXISTS workspace.{SCHEMA_NAME}", warehouse_id):
    print("Failed to create schema")
    exit(1)

# Create tables from CSV files
csv_files = {
    "transactions": f"{DATA_DIR}/transactions.csv",
    "transactions_late": f"{DATA_DIR}/transactions_late.csv",
    "profiles": f"{DATA_DIR}/profiles.csv",
    "activity": f"{DATA_DIR}/activity.csv",
    "account_health": f"{DATA_DIR}/account_health.csv",
    "labels": f"{DATA_DIR}/labels.csv"
}

for table_name, csv_path in csv_files.items():
    # Drop table if it exists
    execute_sql(f"DROP TABLE IF EXISTS workspace.{SCHEMA_NAME}.{table_name}", warehouse_id)
    
    # Create table
    query = f"""
    CREATE TABLE workspace.{SCHEMA_NAME}.{table_name} USING CSV
    OPTIONS (path "{csv_path}", header "true", inferSchema "true")
    """
    if not execute_sql(query, warehouse_id):
        print(f"Failed to create table: {table_name}")
        exit(1)

# Create the training dataset
# Drop the table if it exists
execute_sql(f"DROP TABLE IF EXISTS workspace.{SCHEMA_NAME}.churntrainingaf8b21_v1", warehouse_id)

query = f"""
CREATE TABLE workspace.{SCHEMA_NAME}.churntrainingaf8b21_v1 AS
WITH latest_transactions AS (
    SELECT 
        t.account_id,
        t.amount,
        t.balance,
        l.label_time
    FROM workspace.{SCHEMA_NAME}.labels l
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
                SELECT * FROM workspace.{SCHEMA_NAME}.transactions
                UNION ALL
                SELECT * FROM workspace.{SCHEMA_NAME}.transactions_late
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
    FROM workspace.{SCHEMA_NAME}.labels l
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
            FROM workspace.{SCHEMA_NAME}.profiles
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
    FROM workspace.{SCHEMA_NAME}.labels l
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
            FROM workspace.{SCHEMA_NAME}.activity
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
    FROM workspace.{SCHEMA_NAME}.labels l
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
            FROM workspace.{SCHEMA_NAME}.account_health
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
FROM workspace.{SCHEMA_NAME}.labels l
LEFT JOIN latest_transactions t ON l.account_id = t.account_id AND l.label_time = t.label_time
LEFT JOIN latest_profiles p ON l.account_id = p.account_id AND l.label_time = p.label_time
LEFT JOIN latest_activity a ON l.account_id = a.account_id AND l.label_time = a.label_time
LEFT JOIN latest_health h ON l.account_id = h.account_id AND l.label_time = h.label_time
"""

if not execute_sql(query, warehouse_id):
    print("Failed to create training dataset")
    exit(1)

# Register the dataset as a versioned model
try:
    url = f"{DATABRICKS_HOST}/api/2.0/unity-catalog/model-versions"
    payload = {
        "name": f"workspace.{SCHEMA_NAME}.churntrainingaf8b21",
        "source": f"workspace.{SCHEMA_NAME}.churntrainingaf8b21_v1",
        "version": "1"
    }
    
    response = requests.post(url, headers=headers, data=json.dumps(payload))
    if response.status_code != 200:
        raise Exception(f"Failed to register model: {response.text}")
    
    print("Registered model version: churntrainingaf8b21, version 1")
except Exception as e:
    print(f"Error registering model: {e}")
    exit(1)

print("Training dataset created and registered successfully.")