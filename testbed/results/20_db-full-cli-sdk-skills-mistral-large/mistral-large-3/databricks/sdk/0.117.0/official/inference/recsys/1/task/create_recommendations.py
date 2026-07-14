#!/usr/bin/env python3
"""
Compute top-5 recommendations for every user using dot product of embeddings,
excluding items they have already interacted with. Create a feature table
`recse3a36e` in the specified Unity Catalog schema and enable online access.
"""

import os
import csv
import numpy as np
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import *
from databricks.sdk.service.sql import *

# Load data
interactions = []
with open('data/interactions.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        interactions.append(row)

user_embeddings = {}
with open('data/user_embeddings.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        user_id = row['user_id']
        embedding = np.array([float(row[f'e{i}']) for i in range(1, 9)])
        user_embeddings[user_id] = embedding

item_embeddings = {}
with open('data/item_embeddings.csv', 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        item_id = row['item_id']
        embedding = np.array([float(row[f'e{i}']) for i in range(1, 9)])
        item_embeddings[item_id] = embedding

# Build user-item interaction set
user_interacted_items = {}
for interaction in interactions:
    user_id = interaction['user_id']
    item_id = interaction['item_id']
    if user_id not in user_interacted_items:
        user_interacted_items[user_id] = set()
    user_interacted_items[user_id].add(item_id)

# Compute recommendations
recommendations = []
for user_id, user_embedding in user_embeddings.items():
    scores = []
    interacted_items = user_interacted_items.get(user_id, set())
    
    for item_id, item_embedding in item_embeddings.items():
        if item_id in interacted_items:
            continue
        score = np.dot(user_embedding, item_embedding)
        scores.append((score, item_id))
    
    # Sort by score descending, then by item_id ascending for ties
    scores.sort(key=lambda x: (-x[0], x[1]))
    
    # Take top 5
    for rank, (score, item_id) in enumerate(scores[:5], start=1):
        rec_id = f"{user_id}#{rank}"
        recommendations.append({
            "rec_id": rec_id,
            "user_id": user_id,
            "rank": rank,
            "item_id": item_id
        })

# Initialize Databricks SDK
w = WorkspaceClient()

# Schema and table names
schema_name = os.environ["MLPAB_DATABRICKS_SCHEMA"].split(".")[-1]
table_name = "recse3a36e"
full_table_name = f"workspace.{schema_name}.{table_name}"

# Create schema if not exists
try:
    w.schemas.create(name=schema_name, catalog_name="workspace")
except Exception as e:
    if "already exists" not in str(e) and "SCHEMA_ALREADY_EXISTS" not in str(e):
        raise

# Create volume for temporary files
volume_name = f"{os.environ['MLPAB_DATABRICKS_PREFIX']}_volume"
try:
    w.volumes.create(
        name=volume_name,
        catalog_name="workspace",
        schema_name=schema_name,
        volume_type=VolumeType.MANAGED
    )
except Exception as e:
    if "already exists" not in str(e) and "VOLUME_ALREADY_EXISTS" not in str(e):
        raise

# Create feature table using SQL
try:
    w.statement_execution.execute_statement(
        warehouse_id=list(w.warehouses.list())[0].id,
        catalog="workspace",
        schema=schema_name,
        statement=f"""
        CREATE TABLE IF NOT EXISTS {full_table_name} (
            rec_id STRING NOT NULL COMMENT 'Record key: <user_id>#<rank>',
            user_id STRING NOT NULL COMMENT 'User ID',
            rank INT NOT NULL COMMENT 'Rank (1-5)',
            item_id STRING NOT NULL COMMENT 'Recommended item ID'
        )
        """
    )
except Exception as e:
    if "already exists" not in str(e) and "TABLE_ALREADY_EXISTS" not in str(e):
        raise

# Write recommendations to a temporary CSV and upload
import tempfile
with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as tmp:
    fieldnames = ["rec_id", "user_id", "rank", "item_id"]
    writer = csv.DictWriter(tmp, fieldnames=fieldnames)
    writer.writeheader()
    writer.writerows(recommendations)
    tmp_path = tmp.name

# Upload to volume
volume_path = f"/Volumes/workspace/{schema_name}/{volume_name}/recommendations.csv"
with open(tmp_path, 'rb') as f:
    w.files.upload(volume_path, f, overwrite=True)

# Load data into the feature table
w.statement_execution.execute_statement(
    warehouse_id=list(w.warehouses.list())[0].id,
    catalog="workspace",
    schema=schema_name,
    statement=f"COPY INTO {full_table_name} FROM '/Volumes/workspace/{schema_name}/{volume_name}/recommendations.csv' FILEFORMAT = CSV FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'false') VALIDATE ALL ROWS"
)

# Enable online table for low-latency access
try:
    w.statement_execution.execute_statement(
        warehouse_id=list(w.warehouses.list())[0].id,
        catalog="workspace",
        schema=schema_name,
        statement=f"""
        ALTER TABLE {full_table_name} SET TBLPROPERTIES (
            'delta.enableChangeDataFeed' = 'true',
            'delta.feature.store.enabled' = 'true'
        )
        """
    )
except Exception as e:
    print(f"Could not enable online access: {e}")

print(f"Feature table {full_table_name} created and online access enabled.")