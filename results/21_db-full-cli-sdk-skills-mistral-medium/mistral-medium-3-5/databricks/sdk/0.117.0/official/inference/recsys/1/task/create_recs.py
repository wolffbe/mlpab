#!/usr/bin/env python3
"""
Create recommendations table and online table on Databricks.
"""
import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat

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

# Step 1: Upload CSV files to workspace FileStore
print("Uploading CSV files to workspace FileStore...")
workspace_path = f'/FileStore/tmp/{PREFIX}_recs_data/'
wc.workspace.mkdirs(workspace_path)

# Upload interactions.csv
with open('data/interactions.csv', 'r') as f:
    wc.workspace.upload(f"{workspace_path}interactions.csv", f.read(), format=ImportFormat.AUTO, overwrite=True)

# Upload user_embeddings.csv
with open('data/user_embeddings.csv', 'r') as f:
    wc.workspace.upload(f"{workspace_path}user_embeddings.csv", f.read(), format=ImportFormat.AUTO, overwrite=True)

# Upload item_embeddings.csv
with open('data/item_embeddings.csv', 'r') as f:
    wc.workspace.upload(f"{workspace_path}item_embeddings.csv", f.read(), format=ImportFormat.AUTO, overwrite=True)

print("Files uploaded successfully.")

# Step 2: Create temporary tables from CSV files
print("Creating temporary tables...")

# Create interactions table
wc.statement_execution.execute_statement(
    statement=f"""
    CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA_NAME}.interactions_temp 
    USING CSV
    OPTIONS (
        path "/FileStore/tmp/{PREFIX}_recs_data/interactions.csv",
        header "true",
        inferSchema "true"
    )
    """,
    warehouse_id=warehouse_id,
    wait_timeout="10s"
)

# Create user embeddings table
wc.statement_execution.execute_statement(
    statement=f"""
    CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA_NAME}.user_embeddings_temp 
    USING CSV
    OPTIONS (
        path "/FileStore/tmp/{PREFIX}_recs_data/user_embeddings.csv",
        header "true",
        inferSchema "true"
    )
    """,
    warehouse_id=warehouse_id,
    wait_timeout="10s"
)

# Create item embeddings table
wc.statement_execution.execute_statement(
    statement=f"""
    CREATE OR REPLACE TABLE {CATALOG}.{SCHEMA_NAME}.item_embeddings_temp 
    USING CSV
    OPTIONS (
        path "/FileStore/tmp/{PREFIX}_recs_data/item_embeddings.csv",
        header "true",
        inferSchema "true"
    )
    """,
    warehouse_id=warehouse_id,
    wait_timeout="10s"
)

print("Temporary tables created.")

# Step 3: Compute recommendations using SQL
# We need to:
# 1. For each user, compute dot product with all items
# 2. Exclude items the user has already interacted with
# 3. Rank by dot product descending, break ties by item_id ascending
# 4. Take top 5 per user

print("Computing recommendations...")

# Compute dot product manually: e1*e1 + e2*e2 + ... + e8*e8
create_recs_sql = f"""
WITH interacted AS (
    SELECT DISTINCT user_id, item_id
    FROM {CATALOG}.{SCHEMA_NAME}.interactions_temp
),
all_pairs AS (
    SELECT 
        u.user_id,
        i.item_id,
        u.e1 * i.e1 + u.e2 * i.e2 + u.e3 * i.e3 + u.e4 * i.e4 + 
        u.e5 * i.e5 + u.e6 * i.e6 + u.e7 * i.e7 + u.e8 * i.e8 as dot_product
    FROM {CATALOG}.{SCHEMA_NAME}.user_embeddings_temp u
    CROSS JOIN {CATALOG}.{SCHEMA_NAME}.item_embeddings_temp i
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

wc.feature_store.publish_table(
    source_table_name=f"{CATALOG}.{SCHEMA_NAME}.recs02400f",
    publish_spec=PublishSpec(
        online_store=online_store_name,
        online_table_name=online_table_name,
        publish_mode=PublishSpecPublishMode.SNAPSHOT
    )
)

print(f"Online table {online_table_name} published to store {online_store_name}")

print("\nDone! Feature table and online table created successfully.")
