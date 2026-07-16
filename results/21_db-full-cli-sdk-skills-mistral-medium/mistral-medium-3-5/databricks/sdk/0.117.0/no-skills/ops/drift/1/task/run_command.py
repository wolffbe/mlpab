#!/usr/bin/env python3
import os
import json
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import compute

# Initialize workspace client
w = WorkspaceClient()

# Get environment variables
prefix = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpabc6cfe1')
username = "benedict@logicalclocks.com"
workspace_dir = f"/Users/{username}/{prefix}"

# Create a cluster to run commands on
print("Creating cluster...")
try:
    cluster = w.clusters.create(
        cluster_name=f"{prefix}_cluster",
        spark_version="14.3.x-scala2.12",
        node_type_id="Standard_DS3_v2",
        num_workers=0,
        autotermination_minutes=10
    )
    cluster_id = cluster.cluster_id
    print(f"Cluster created: {cluster_id}")
    
    # Wait for cluster to be running
    print("Waiting for cluster to start...")
    for i in range(30):  # Wait up to 5 minutes
        time.sleep(10)
        cluster_info = w.clusters.get(cluster_id)
        state = cluster_info.state
        print(f"  Cluster state: {state}")
        if state == 'RUNNING':
            break
    
    if state != 'RUNNING':
        print(f"Cluster failed to start: {state}")
        exit(1)
    
    # Create a context
    print("Creating context...")
    context = w.command_execution.create(
        cluster_id=cluster_id,
        language='python',
        command="print('Context created')"
    )
    context_id = context.id
    print(f"Context created: {context_id}")
    
    # Wait for context to be ready
    print("Waiting for context to be ready...")
    for i in range(10):
        time.sleep(5)
        context_status = w.command_execution.context_status(context_id)
        status = context_status.status
        print(f"  Context status: {status}")
        if status == 'Running':
            break
    
    # Now run the analysis command
    analysis_code = f"""
import csv
import json
from collections import defaultdict

# Read the features.csv file
features_file = '{workspace_dir}/features.csv'

features_data = []
with open(features_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        features_data.append(row)

print(f"Loaded {{len(features_data)}} rows")

# Group by date
daily_data = defaultdict(list)
for row in features_data:
    date_str = row['event_time'].split('T')[0]
    daily_data[date_str].append(row)

# Calculate daily averages
features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
daily_stats = []
for date in sorted(daily_data.keys()):
    rows = daily_data[date]
    stats = {'date': date}
    for feature in features:
        values = [float(row[feature]) for row in rows]
        avg = sum(values) / len(values)
        stats[f'avg_{feature}'] = avg
    daily_stats.append(stats)

# Find the feature with the largest absolute change
max_abs_change = 0
max_feature = None
max_date = None

for feature in features:
    for i in range(1, len(daily_stats)):
        prev_avg = daily_stats[i-1][f'avg_{feature}']
        curr_avg = daily_stats[i][f'avg_{feature}']
        abs_change = abs(curr_avg - prev_avg)
        if abs_change > max_abs_change:
            max_abs_change = abs_change
            max_feature = feature
            max_date = daily_stats[i]['date']

print(f"Max absolute change: {{max_feature}} on {{max_date}} with change {{max_abs_change:.4f}}")

# Write the answer
answer = {{
    "feature": max_feature,
    "onset": max_date
}}

# Write to workspace
answer_file = '{workspace_dir}/answers.json'
with open(answer_file, 'w') as f:
    json.dump(answer, f)

print(f"Answer: {{answer}}")
"""
    
    print("Running analysis...")
    result = w.command_execution.execute_and_wait(
        cluster_id=cluster_id,
        context_id=context_id,
        language='python',
        command=analysis_code
    )
    
    print(f"Command result: {result}")
    
    # Download the answer
    print("\nDownloading answer...")
    try:
        answer_file = f"{workspace_dir}/answers.json"
        w.workspace.download(workspace_path=answer_file, local_path='submission/answers.json', overwrite=True)
        print("Answer downloaded")
        
        # Verify
        with open('submission/answers.json', 'r') as f:
            answer = json.load(f)
        print(f"Answer: {answer}")
    except Exception as e:
        print(f"Failed to download answer: {e}")
    
    # Clean up - terminate cluster
    print("\nCleaning up...")
    try:
        w.clusters.delete(cluster_id)
        print(f"Cluster {cluster_id} deleted")
    except Exception as e:
        print(f"Failed to delete cluster: {e}")
    
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")
