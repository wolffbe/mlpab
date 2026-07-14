#!/usr/bin/env python3
"""
Compute recommendations locally, upload to a Unity Catalog volume, and create a feature table.
"""

import os
import pandas as pd
import numpy as np
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog

# Load data
interactions = pd.read_csv("data/interactions.csv")
user_embeddings = pd.read_csv("data/user_embeddings.csv")
item_embeddings = pd.read_csv("data/item_embeddings.csv")

# Extract embedding columns (e1..e8)
user_emb_cols = [f"e{i}" for i in range(1, 9)]
item_emb_cols = [f"e{i}" for i in range(1, 9)]

# Convert embeddings to numpy arrays
user_embeddings_array = user_embeddings[user_emb_cols].values
item_embeddings_array = item_embeddings[item_emb_cols].values

# Create a set of interacted items for each user
interacted_items = interactions.groupby('user_id')['item_id'].apply(set).to_dict()

# Compute recommendations for each user
recommendations = []
for idx, user_row in user_embeddings.iterrows():
    user_id = user_row['user_id']
    user_emb = user_embeddings_array[idx]
    
    # Compute dot products for all items
    scores = np.dot(item_embeddings_array, user_emb)
    
    # Create a DataFrame for scores and item_ids
    scores_df = pd.DataFrame({
        'item_id': item_embeddings['item_id'],
        'score': scores
    })
    
    # Exclude interacted items
    interacted = interacted_items.get(user_id, set())
    scores_df = scores_df[~scores_df['item_id'].isin(interacted)]
    
    # Sort by score (descending) and item_id (ascending for ties)
    scores_df = scores_df.sort_values(by=['score', 'item_id'], ascending=[False, True])
    
    # Take top-5
    top_5 = scores_df.head(5)
    
    # Format recommendations
    for rank, (_, row) in enumerate(top_5.iterrows(), start=1):
        rec_id = f"{user_id}#{rank}"
        recommendations.append({
            'rec_id': rec_id,
            'user_id': user_id,
            'rank': rank,
            'item_id': row['item_id']
        })

# Convert recommendations to DataFrame
recs_df = pd.DataFrame(recommendations)

# Save recommendations to a CSV file
output_file = "recommendations.csv"
recs_df.to_csv(output_file, index=False)
print(f"Recommendations saved to {output_file}.")

# Initialize Databricks WorkspaceClient
w = WorkspaceClient()

# Get schema and prefix from environment variables
full_schema_name = os.environ['MLPAB_DATABRICKS_SCHEMA']
schema_name = full_schema_name.split(".")[-1]
table_name = "recse3a36e"
full_table_name = f"{full_schema_name}.{table_name}"
volume_name = f"{os.environ['MLPAB_DATABRICKS_PREFIX']}_volume"
full_volume_name = f"{full_schema_name}.{volume_name}"

# Create schema if it doesn't exist
try:
    w.schemas.create(name=schema_name, catalog_name="workspace")
    print(f"Schema {full_schema_name} created.")
except Exception as e:
    if "already exists" not in str(e):
        raise

# Create volume if it doesn't exist
try:
    w.volumes.create(
        name=volume_name,
        catalog_name="workspace",
        schema_name=schema_name,
        volume_type=catalog.VolumeType.MANAGED
    )
    print(f"Volume {full_volume_name} created.")
except Exception as e:
    if "already exists" not in str(e):
        raise

# Upload recommendations CSV to the volume
volume_path = f"/Volumes/workspace/{schema_name}/{volume_name}/recommendations.csv"
with open(output_file, "rb") as f:
    w.files.upload(volume_path, f, overwrite=True)
print(f"Uploaded recommendations to {volume_path}.")

# Create the feature table from the CSV using Spark SQL
spark_sql = f"""
CREATE TABLE IF NOT EXISTS {full_table_name} 
USING CSV
OPTIONS (path "{volume_path}", header "true", inferSchema "true")
"""

w.statement_execution.execute_statement(
    warehouse_id=os.environ.get("DATABRICKS_SQL_WAREHOUSE_ID"),
    catalog="workspace",
    schema=schema_name,
    statement=spark_sql
).result()
print(f"Feature table {full_table_name} created.")

# Enable online access for low-latency lookup
try:
    w.online_tables.create(
        name=full_table_name,
        spec=catalog.OnlineTableSpec(
            source_table_full_name=full_table_name,
            primary_key_columns=["rec_id"]
        )
    )
    print(f"Online table {full_table_name} created successfully.")
except Exception as e:
    if "already exists" not in str(e):
        raise

print(f"Feature table {full_table_name} is ready for online access.")