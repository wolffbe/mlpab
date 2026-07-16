#!/usr/bin/env python3
"""
Create the feature table and online table on Databricks.
All computation happens on the platform via a notebook.
"""
import os
from databricks.sdk import WorkspaceClient

def main():
    w = WorkspaceClient()
    
    # Get environment variables
    schema_name = os.environ['MLPAB_DATABRICKS_SCHEMA']  # workspace.mlpabcb1003
    prefix = os.environ['MLPAB_DATABRICKS_PREFIX']  # mlpabcb1003
    
    # Parse schema to get catalog and schema
    parts = schema_name.split('.')
    catalog_name = parts[0]
    schema_part = parts[1]
    
    # Get current user
    current_user = w.current_user.me().user_name
    
    # Upload data files to DBFS
    print("Uploading data files to DBFS...")
    dbfs_path = f"dbfs:/FileStore/mlpab/{prefix}/data"
    
    # Upload feature_history.csv
    with open('data/feature_history.csv', 'rb') as f:
        w.dbfs.upload(f.name, f"{dbfs_path}/feature_history.csv", overwrite=True)
    
    # Upload model.json
    with open('data/model.json', 'rb') as f:
        w.dbfs.upload(f.name, f"{dbfs_path}/model.json", overwrite=True)
    
    # Upload scoring_request.md
    with open('data/scoring_request.md', 'rb') as f:
        w.dbfs.upload(f.name, f"{dbfs_path}/scoring_request.md", overwrite=True)
    
    print("Data files uploaded")
    
    # Create a notebook that does all the computation
    notebook_path = f"/Users/{current_user}/{prefix}/compute_scores"
    
    notebook_content = f"""# Databricks notebook source
# This notebook computes scores and creates the feature table

# Read the scoring request to get T
T = 1773568800000  # From data/scoring_request.md

# Read model weights
import json
with open('/dbfs/FileStore/mlpab/{prefix}/data/model.json', 'r') as f:
    model = json.load(f)

weights = model['weights']
bias = model['bias']
w_f1 = weights['f1']
w_f2 = weights['f2']
w_f3 = weights['f3']

# Read feature history
import pandas as pd
feature_history = pd.read_csv('/dbfs/FileStore/mlpab/{prefix}/data/feature_history.csv')

# Filter to get most recent revision at or before T for each account
filtered = feature_history[feature_history['event_time'] <= T]
# Sort by account_id and event_time descending
filtered = filtered.sort_values(['account_id', 'event_time'], ascending=[True, False])
# Keep first row per account (most recent at or before T)
latest_features = filtered.drop_duplicates('account_id', keep='first')

# Compute scores
def sigmoid(x):
    import math
    return 1.0 / (1.0 + math.exp(-x))

latest_features['linear'] = latest_features.apply(
    lambda row: w_f1 * row['f1'] + w_f2 * row['f2'] + w_f3 * row['f3'] + bias,
    axis=1
)
latest_features['score'] = latest_features['linear'].apply(sigmoid)
latest_features['score'] = latest_features['score'].round(6)

# Select only account_id and score
result = latest_features[['account_id', 'score']].sort_values('account_id')

# Create the feature table
spark_df = spark.createDataFrame(result)

# Write to Delta table with feature table properties
spark_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(
    f"{catalog_name}.{schema_part}.scores076684"
)

# Set feature table property
spark.sql(f"""
ALTER TABLE {catalog_name}.{schema_part}.scores076684 
SET TBLPROPERTIES ('feature_table' = 'true')
""")

# Create online table for low-latency lookup
spark.sql(f"""
CREATE ONLINE TABLE IF NOT EXISTS {prefix}_scores076684_online
FROM {catalog_name}.{schema_part}.scores076684
PRIMARY KEY (account_id)
""")

print("Done! Feature table and online table created successfully.")
"""
    
    # Write the notebook
    w.workspace.upload(
        path=f"{notebook_path}.py",
        content=notebook_content.encode('utf-8'),
        overwrite=True,
        language="PYTHON"
    )
    
    print(f"Notebook created at {notebook_path}.py")
    
    # Now run the notebook
    print("Running notebook...")
    
    # Get a cluster to run on
    clusters = w.clusters.list()
    if clusters:
        cluster_id = clusters[0].cluster_id
        print(f"Using cluster: {cluster_id}")
        
        # Run the notebook on the cluster
        run = w.jobs.run_now(
            notebook_task={
                "notebook_path": f"{notebook_path}.py"
            },
            existing_cluster_id=cluster_id
        )
        
        run_id = run.run_id
        print(f"Started run: {run_id}")
        
        # Wait for completion
        final_state = w.jobs.wait_get_run_finished(run_id=run_id)
        print(f"Run finished with state: {final_state.life_cycle_state}")
        
        if final_state.life_cycle_state == "TERMINATED":
            print("Success!")
        else:
            print(f"Failed with result state: {final_state.result_state}")
            
    else:
        print("No clusters available, trying to create one...")
        # Create a small cluster
        cluster = w.clusters.create(
            cluster_name=f"{prefix}_compute_cluster",
            spark_version="14.3.x-scala2.12",
            node_type_id="Standard_DS3_v2",
            num_workers=0,
            autoscale={"min_workers": 0, "max_workers": 2},
            spark_conf={"spark.databricks.delta.preview.enabled": "true"}
        )
        cluster_id = cluster.cluster_id
        print(f"Created cluster: {cluster_id}")
        
        # Wait for cluster to be ready
        w.clusters.wait_get_cluster_running(cluster_id=cluster_id)
        
        # Run the notebook
        run = w.jobs.run_now(
            notebook_task={
                "notebook_path": f"{notebook_path}.py"
            },
            existing_cluster_id=cluster_id
        )
        
        run_id = run.run_id
        print(f"Started run: {run_id}")
        
        # Wait for completion
        final_state = w.jobs.wait_get_run_finished(run_id=run_id)
        print(f"Run finished with state: {final_state.life_cycle_state}")
        
        # Clean up cluster
        w.clusters.delete(cluster_id)
        print(f"Deleted cluster: {cluster_id}")

if __name__ == "__main__":
    main()
