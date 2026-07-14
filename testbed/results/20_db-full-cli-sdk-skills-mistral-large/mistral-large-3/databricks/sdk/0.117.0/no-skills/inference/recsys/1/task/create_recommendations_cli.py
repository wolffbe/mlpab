#!/usr/bin/env python3
"""
Upload input data to a Unity Catalog volume and create a job using the Databricks CLI.
"""

import os
import subprocess
import json

# Get schema and prefix from environment variables
full_schema_name = os.environ['MLPAB_DATABRICKS_SCHEMA']
schema_name = full_schema_name.split(".")[-1]
table_name = "recse3a36e"
full_table_name = f"{full_schema_name}.{table_name}"
volume_name = f"{os.environ['MLPAB_DATABRICKS_PREFIX']}_volume"
full_volume_name = f"{full_schema_name}.{volume_name}"

# Upload input data to the volume
for file_name in ["interactions.csv", "user_embeddings.csv", "item_embeddings.csv"]:
    file_path = f"data/{file_name}"
    volume_path = f"/Volumes/workspace/{schema_name}/{volume_name}/{file_name}"
    
    # Upload file using Databricks CLI
    cmd = [
        "databricks", "fs", "cp",
        file_path,
        f"dbfs:{volume_path}",
        "--overwrite"
    ]
    subprocess.run(cmd, check=True)
    print(f"Uploaded {file_name} to {volume_path}.")

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

job_script_path = f"/Volumes/workspace/{schema_name}/{volume_name}/compute_recommendations_job.py"
cmd = [
    "databricks", "fs", "cp",
    "-",
    f"dbfs:{job_script_path}",
    "--overwrite"
]
subprocess.run(cmd, input=job_script.encode(), check=True)
print(f"Uploaded job script to {job_script_path}.")

# Define the job using Databricks CLI
job_name = f"{os.environ['MLPAB_DATABRICKS_PREFIX']}_compute_recommendations"
job_definition = {
    "name": job_name,
    "tasks": [
        {
            "task_key": "compute_recommendations",
            "new_cluster": {
                "spark_version": "14.3.x-scala2.12",
                "node_type_id": "i3.xlarge",
                "num_workers": 1
            },
            "spark_python_task": {
                "python_file": f"dbfs:{job_script_path}"
            }
        }
    ]
}

# Create the job
cmd = ["databricks", "jobs", "create", "--json", json.dumps(job_definition)]
result = subprocess.run(cmd, capture_output=True, text=True, check=True)
job_id = json.loads(result.stdout)["job_id"]
print(f"Job {job_name} created with job_id {job_id}.")

# Run the job
cmd = ["databricks", "jobs", "run-now", "--job-id", str(job_id)]
result = subprocess.run(cmd, capture_output=True, text=True, check=True)
run_id = json.loads(result.stdout)["run_id"]
print(f"Job {job_name} started with run_id {run_id}.")

print(f"Feature table {full_table_name} will be created by the job.")