#!/usr/bin/env python3
"""
Create recommendations table on Databricks.
Compute recommendations locally and create the table on the platform.
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

# Step 1: Load data locally
print("Loading data...")

# Load interactions
interactions = {}
with open('data/interactions.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        user_id = row['user_id']
        item_id = row['item_id']
        if user_id not in interactions:
            interactions[user_id] = set()
        interactions[user_id].add(item_id)

# Load user embeddings
user_embeddings = {}
with open('data/user_embeddings.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        user_id = row['user_id']
        emb = [float(row[f'e{i}']) for i in range(1, 9)]
        user_embeddings[user_id] = emb

# Load item embeddings
item_embeddings = {}
with open('data/item_embeddings.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        item_id = row['item_id']
        emb = [float(row[f'e{i}']) for i in range(1, 9)]
        item_embeddings[item_id] = emb

print(f"Loaded {len(interactions)} users, {len(item_embeddings)} items")

# Step 2: Compute recommendations
print("Computing recommendations...")

# For each user, compute dot product with all items
# Exclude items the user has already interacted with
# Rank by dot product descending, break ties by item_id ascending
# Take top 5 per user

recommendations = []

for user_id in sorted(user_embeddings.keys()):
    user_emb = user_embeddings[user_id]
    user_interacted = interactions.get(user_id, set())
    
    # Compute dot products with all items
    item_scores = []
    for item_id in item_embeddings:
        if item_id in user_interacted:
            continue
        item_emb = item_embeddings[item_id]
        dot_product = sum(u * i for u, i in zip(user_emb, item_emb))
        item_scores.append((item_id, dot_product))
    
    # Sort by dot product descending, then by item_id ascending
    item_scores.sort(key=lambda x: (-x[1], x[0]))
    
    # Take top 5
    top_5 = item_scores[:5]
    
    for rank, (item_id, score) in enumerate(top_5, 1):
        rec_id = f"{user_id}#{rank}"
        recommendations.append({
            'rec_id': rec_id,
            'user_id': user_id,
            'rank': rank,
            'item_id': item_id
        })

print(f"Computed {len(recommendations)} recommendations")

# Step 3: Create the feature table on the platform
print("Creating feature table on platform...")

# Create the table with explicit schema
wc.statement_execution.execute_statement(
    statement=f"""
    CREATE TABLE IF NOT EXISTS {CATALOG}.{SCHEMA_NAME}.recs02400f (
        rec_id STRING,
        user_id STRING,
        rank INT,
        item_id STRING
    )
    COMMENT 'Recommendation table version 1'
    """,
    warehouse_id=warehouse_id,
    wait_timeout="10s"
)

# Insert the recommendations
# Build INSERT statements in batches
batch_size = 50
for i in range(0, len(recommendations), batch_size):
    batch = recommendations[i:i+batch_size]
    values = []
    for rec in batch:
        values.append(f"('{rec['rec_id']}', '{rec['user_id']}', {rec['rank']}, '{rec['item_id']}')")
    
    insert_sql = f"""
    INSERT INTO {CATALOG}.{SCHEMA_NAME}.recs02400f 
    VALUES {', '.join(values)}
    """
    
    wc.statement_execution.execute_statement(
        statement=insert_sql,
        warehouse_id=warehouse_id,
        wait_timeout="10s"
    )
    print(f"Inserted batch {i//batch_size + 1}")

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
