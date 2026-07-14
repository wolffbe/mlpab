#!/usr/bin/env python3
"""
Upload input data to a Unity Catalog volume and create a job to compute recommendations.
"""

import os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog, jobs

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

# Upload input data to the volume
for file_name in ["interactions.csv", "user_embeddings.csv", "item_embeddings.csv"]:
    file_path = f"data/{file_name}"
    volume_path = f"/Volumes/workspace/{schema_name}/{volume_name}/{file_name}"
    
    # Upload file
    with open(file_path, "rb") as f:
        w.files.upload(volume_path, f, overwrite=True)
    print(f"Uploaded {file_name} to {volume_path}.")

# Define the job to compute recommendations
job_name = f"{os.environ['MLPAB_DATABRICKS_PREFIX']}_compute_recommendations"

# Define cluster attributes
cluster_attributes = {
    "spark_version": "14.3.x-scala2.12",
    "node_type_id": "i3.xlarge",
    "num_workers": 1
}

# Define job clusters
job_cluster = {
    "job_cluster_key": "shared_cluster",
    "new_cluster": cluster_attributes
}

# Define task
task = {
    "task_key": "compute_recommendations",
    "job_cluster_key": "shared_cluster",
    "spark_python_task": {
        "python_file": f"dbfs:/Volumes/workspace/{schema_name}/{volume_name}/compute_recommendations_job.py"
    }
}

# Define job settings
job_settings = {
    "name": job_name,
    "job_clusters": [job_cluster],
    "tasks": [task]
}

job = w.jobs.create(**job_settings)
print(f"Job {job_name} created with job_id {job.job_id}.")

# Write the job script to the volume
job_script = f"""
import pandas as pd
import numpy as np
from pyspark.sql import SparkSession

# Load data
interactions = pd.read_csv("/dbfs/Volumes/workspace/{schema_name}/{volume_name}/interactions.csv")
user_embeddings = pd.read_csv("/dbfs/Volumes/workspace/{schema_name}/{volume_name}/user_embeddings.csv")
item_embeddings = pd.read_csv("/dbfs/Volumes/workspace/{schema_name}/{volume_name}/item_embeddings.csv")

# Extract embedding columns (e1..e8)
user_emb_cols = [f"e{{i}}" for i in range(1, 9)]
item_emb_cols = [f"e{{i}}" for i in range(1, 9)]

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
    scores_df = pd.DataFrame({{
        'item_id': item_embeddings['item_id'],
        'score': scores
    }})
    
    # Exclude interacted items
    interacted = interacted_items.get(user_id, set())
    scores_df = scores_df[~scores_df['item_id'].isin(interacted)]
    
    # Sort by score (descending) and item_id (ascending for ties)
    scores_df = scores_df.sort_values(by=['score', 'item_id'], ascending=[False, True])
    
    # Take top-5
    top_5 = scores_df.head(5)
    
    # Format recommendations
    for rank, (_, row) in enumerate(top_5.iterrows(), start=1):
        rec_id = f"{{user_id}}#{{rank}}"
        recommendations.append({{
            'rec_id': rec_id,
            'user_id': user_id,
            'rank': rank,
            'item_id': row['item_id']
        }})

# Convert recommendations to DataFrame
recs_df = pd.DataFrame(recommendations)

# Initialize SparkSession
spark = SparkSession.builder.getOrCreate()

# Convert Pandas DataFrame to Spark DataFrame
spark_df = spark.createDataFrame(recs_df)

# Write to the table
spark_df.write.saveAsTable("{full_table_name}", mode="overwrite")

# Enable online access for low-latency lookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog

w = WorkspaceClient()
try:
    w.online_tables.create(
        name="{full_table_name}",
        spec=catalog.OnlineTableSpec(
            source_table_full_name="{full_table_name}",
            primary_key_columns=["rec_id"]
        )
    )
    print("Online table created successfully.")
except Exception as e:
    if "already exists" not in str(e):
        raise
"""

# Upload the job script
job_script_path = f"/Volumes/workspace/{schema_name}/{volume_name}/compute_recommendations_job.py"
w.files.upload(job_script_path, job_script, overwrite=True)
print(f"Uploaded job script to {job_script_path}.")

# Run the job
run = w.jobs.run_now(job_id=job.job_id).result()
print(f"Job {job_name} completed with run_id {run.run_id}.")

print(f"Feature table {full_table_name} created successfully with online access enabled.")