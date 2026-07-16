#!/usr/bin/env python3
"""
Create recommendations table on Databricks.
All computation happens on the platform through SQL.
"""
import os
import csv
from databricks.sdk import WorkspaceClient

# Environment variables
SCHEMA = os.environ['MLPAB_DATABRICKS_SCHEMA']  # workspace.mlpab3a8240
PREFIX = os.environ['MLPAB_DATABRICKS_PREFIX']  # mlpab3a8240

# Parse schema
catalog_schema = SCHEMA.split('.')
CATALOG = catalog_schema[0]
SCHEMA_NAME = catalog_schema[1]

print(f"Catalog: {CATALOG}, Schema: {SCHEMA_NAME}")

# Initialize client
wc = WorkspaceClient()

# Warehouse ID (use the running one)
warehouse_id = '8a93fc195da2ceb1'  # mlpab-grader

# Step 1: Create empty tables
print("Creating empty tables...")

wc.statement_execution.execute_statement(
    statement=f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA_NAME}.interactions (
        user_id STRING,
        item_id STRING
    )
    """,
    warehouse_id=warehouse_id,
    wait_timeout="10s"
)

wc.statement_execution.execute_statement(
    statement=f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA_NAME}.user_embeddings (
        user_id STRING,
        e1 FLOAT, e2 FLOAT, e3 FLOAT, e4 FLOAT,
        e5 FLOAT, e6 FLOAT, e7 FLOAT, e8 FLOAT
    )
    """,
    warehouse_id=warehouse_id,
    wait_timeout="10s"
)

wc.statement_execution.execute_statement(
    statement=f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA_NAME}.item_embeddings (
        item_id STRING,
        e1 FLOAT, e2 FLOAT, e3 FLOAT, e4 FLOAT,
        e5 FLOAT, e6 FLOAT, e7 FLOAT, e8 FLOAT
    )
    """,
    warehouse_id=warehouse_id,
    wait_timeout="10s"
)

print("Empty tables created.")

# Step 2: Load data using INSERT statements
print("Loading data into tables...")

# Load interactions
with open('data/interactions.csv', 'r') as f:
    reader = csv.DictReader(f)
    batch = []
    for i, row in enumerate(reader):
        batch.append(f"('{row['user_id']}', '{row['item_id']}')")
        if len(batch) >= 50 or i == 282:  # 283 lines - 1 header
            insert_sql = f"""
            INSERT INTO {CATALOG}.{SCHEMA_NAME}.interactions 
            VALUES {', '.join(batch)}
            """
            wc.statement_execution.execute_statement(
                statement=insert_sql,
                warehouse_id=warehouse_id,
                wait_timeout="10s"
            )
            batch = []
            print(f"  Inserted {i+1} interactions")

# Load user embeddings
with open('data/user_embeddings.csv', 'r') as f:
    reader = csv.DictReader(f)
    batch = []
    for i, row in enumerate(reader):
        vals = f"('{row['user_id']}', {row['e1']}, {row['e2']}, {row['e3']}, {row['e4']}, {row['e5']}, {row['e6']}, {row['e7']}, {row['e8']})"
        batch.append(vals)
        if len(batch) >= 20 or i == 39:  # 40 users
            insert_sql = f"""
            INSERT INTO {CATALOG}.{SCHEMA_NAME}.user_embeddings 
            VALUES {', '.join(batch)}
            """
            wc.statement_execution.execute_statement(
                statement=insert_sql,
                warehouse_id=warehouse_id,
                wait_timeout="10s"
            )
            batch = []
            print(f"  Inserted {i+1} user embeddings")

# Load item embeddings
with open('data/item_embeddings.csv', 'r') as f:
    reader = csv.DictReader(f)
    batch = []
    for i, row in enumerate(reader):
        vals = f"('{row['item_id']}', {row['e1']}, {row['e2']}, {row['e3']}, {row['e4']}, {row['e5']}, {row['e6']}, {row['e7']}, {row['e8']})"
        batch.append(vals)
        if len(batch) >= 20 or i == 59:  # 60 items
            insert_sql = f"""
            INSERT INTO {CATALOG}.{SCHEMA_NAME}.item_embeddings 
            VALUES {', '.join(batch)}
            """
            wc.statement_execution.execute_statement(
                statement=insert_sql,
                warehouse_id=warehouse_id,
                wait_timeout="10s"
            )
            batch = []
            print(f"  Inserted {i+1} item embeddings")

print("Data loaded.")

# Step 3: Compute recommendations using SQL
print("Computing recommendations using SQL...")

create_recs_sql = f"""
WITH interacted AS (
    SELECT DISTINCT user_id, item_id
    FROM {CATALOG}.{SCHEMA_NAME}.interactions
),
all_pairs AS (
    SELECT 
        u.user_id,
        i.item_id,
        u.e1 * i.e1 + u.e2 * i.e2 + u.e3 * i.e3 + u.e4 * i.e4 + 
        u.e5 * i.e5 + u.e6 * i.e6 + u.e7 * i.e7 + u.e8 * i.e8 as dot_product
    FROM {CATALOG}.{SCHEMA_NAME}.user_embeddings u
    CROSS JOIN {CATALOG}.{SCHEMA_NAME}.item_embeddings i
),
filtered_pairs AS (
    SELECT 
        ap.user_id,
        ap.item_id,
        ap.dot_product
    FROM all_pairs ap
    LEFT JOIN interacted i ON ap.user_id = i.user_id AND ap.item_id = i.item_id
    WHERE i.item_id IS NULL
),
ranked AS (
    SELECT 
        user_id,
        item_id,
        dot_product,
        ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY dot_product DESC, item_id ASC) as rank
    FROM filtered_pairs
)
SELECT 
    CONCAT(user_id, '#', CAST(rank AS STRING)) as rec_id,
    user_id,
    CAST(rank AS INT) as rank,
    item_id
FROM ranked
WHERE rank <= 5
ORDER BY user_id, rank
"""

# Execute the query and create the feature table
wc.statement_execution.execute_statement(
    statement=f"""
    CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA_NAME}.recs02400f 
    COMMENT 'Recommendation table version 1'
    AS {create_recs_sql}
    """,
    warehouse_id=warehouse_id,
    wait_timeout="50s"
)

print("Feature table created: recs02400f")

# Add version property
wc.statement_execution.execute_statement(
    statement=f"""
    ALTER TABLE {CATALOG}.{SCHEMA_NAME}.recs02400f 
    SET TBLPROPERTIES ('version' = '1')
    """,
    warehouse_id=warehouse_id,
    wait_timeout="10s"
)

print("Version property added.")

# Step 4: Create online table for low-latency access
print("Creating online table for low-latency access...")

from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode, OnlineStore

# Create an online store if it doesn't exist
# Use a DNS-compliant name (alphanumeric and hyphens only)
online_store_name = f"{PREFIX}-online-store"

try:
    online_store = wc.feature_store.get_online_store(online_store_name)
    print(f"Online store {online_store_name} already exists")
except Exception as e:
    print(f"Creating online store {online_store_name}...")
    # Create online store with required parameters
    online_store_obj = OnlineStore(
        name=online_store_name,
        capacity="CU_1"
    )
    wc.feature_store.create_online_store(online_store_obj)
    print(f"Online store {online_store_name} created")

# Publish the table to online store for low-latency lookup
online_table_name = "recs02400f_v1"

try:
    wc.feature_store.publish_table(
        source_table_name=f"{CATALOG}.{SCHEMA_NAME}.recs02400f",
        publish_spec=PublishSpec(
            online_store=online_store_name,
            online_table_name=online_table_name,
            publish_mode=PublishSpecPublishMode.SNAPSHOT
        )
    )
    print(f"Online table {online_table_name} published to store {online_store_name}")
except Exception as e:
    print(f"Warning: Could not publish to online store: {e}")
    print("Feature table is still available for batch access through SQL warehouse")

print("\nDone! Feature table created successfully.")
