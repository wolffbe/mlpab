#!/usr/bin/env python3
import os
import json
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat
import base64

# Initialize workspace client
w = WorkspaceClient()

# Get environment variables
prefix = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpabc6cfe1')
username = "benedict@logicalclocks.com"
workspace_dir = f"/Users/{username}/{prefix}"

# Create the notebook content
notebook_content = """# Databricks notebook source
# MAGIC %python
import csv
import json
import os
from collections import defaultdict

# Find and read the features.csv file
features_file = None

# Try in current directory
if os.path.exists('features.csv'):
    features_file = 'features.csv'

# Try in workspace directory
if not features_file:
    for root, dirs, files in os.walk('.'):
        if 'features.csv' in files:
            features_file = os.path.join(root, 'features.csv')
            break

# Try absolute path in workspace
if not features_file:
    features_file = f'/Workspace{workspace_dir}/features.csv'

print(f"Reading from: {features_file}")

# Read the CSV
features_data = []
with open(features_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        features_data.append(row)

print(f"Loaded {len(features_data)} rows")

# Group by date
daily_data = defaultdict(list)
for row in features_data:
    date_str = row['event_time'].split('T')[0]
    daily_data[date_str].append(row)

# Calculate daily statistics
features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']
daily_stats = []
for date in sorted(daily_data.keys()):
    rows = daily_data[date]
    stats = {'date': date}
    for feature in features:
        values = [float(row[feature]) for row in rows]
        avg = sum(values) / len(values)
        variance = sum((x - avg) ** 2 for x in values) / len(values)
        std = variance ** 0.5
        stats[f'avg_{feature}'] = avg
        stats[f'std_{feature}'] = std
    stats['count'] = len(rows)
    daily_stats.append(stats)

print("Daily statistics calculated")

# Analyze for drift
feature_drift = {}
for feature in features:
    changes = []
    for i in range(1, len(daily_stats)):
        prev_avg = daily_stats[i-1][f'avg_{feature}']
        curr_avg = daily_stats[i][f'avg_{feature}']
        change = abs(curr_avg - prev_avg)
        relative_change = change / abs(prev_avg) if prev_avg != 0 else 0
        changes.append((daily_stats[i]['date'], change, relative_change))
    
    # Find the maximum change
    if changes:
        max_change = max(changes, key=lambda x: x[2])
        feature_drift[feature] = max_change
        print(f"{feature}: max relative change = {max_change[2]:.4f} on {max_change[0]}")

# Find the feature with the most significant drift
drifted_feature = max(feature_drift.items(), key=lambda x: x[1][2])
onset_date = drifted_feature[1][0]

print(f"\\nDrift detected: feature={drifted_feature[0]}, onset={onset_date}")

# Write the answer
answer = {
    "feature": drifted_feature[0],
    "onset": onset_date
}

# Write to workspace
answer_file = f'{workspace_dir}/answers.json'
with open(answer_file, 'w') as f:
    json.dump(answer, f)

print(f"Answer written: {answer}")
print(f"Answer file: {answer_file}")
"""

# Upload the notebook
notebook_path = f"{workspace_dir}/drift_analysis"

# Encode notebook content
notebook_content_b64 = base64.b64encode(notebook_content.encode('utf-8')).decode('utf-8')

try:
    w.workspace.import_(
        path=f"{notebook_path}.py",
        content=notebook_content_b64,
        format=ImportFormat.SOURCE,
        overwrite=True
    )
    print(f"Notebook uploaded to {notebook_path}.py")
except Exception as e:
    print(f"Notebook upload failed: {e}")
    import traceback
    traceback.print_exc()

# Now run the notebook using jobs API
print("\nCreating and running job...")
try:
    from databricks.sdk.service import jobs
    
    # Create a new job
    new_job = w.jobs.create(
        name=f"{prefix}_drift_analysis",
        tasks=[
            jobs.Task(
                task_key="drift_analysis",
                notebook_task=jobs.NotebookTask(
                    notebook_path=f"/Users/{username}/{prefix}/drift_analysis",
                ),
                existing_cluster_id="",
                new_cluster=jobs.NewCluster(
                    spark_version="14.3.x-scala2.12",
                    node_type_id="Standard_DS3_v2",
                    num_workers=0  # Single node
                )
            )
        ]
    )
    job_id = new_job.job_id
    print(f"Job created: {job_id}")
    
    # Run the job
    run = w.jobs.run_now(job_id)
    run_id = run.run_id
    print(f"Job run started: {run_id}")
    
    # Wait for completion
    print("Waiting for job to complete...")
    max_wait = 60  # 1 minute
    for i in range(max_wait):
        time.sleep(10)
        run_info = w.jobs.get_run(run_id)
        state = run_info.state.life_cycle_state
        result_state = run_info.state.result_state if hasattr(run_info.state, 'result_state') else None
        
        print(f"  Status: {state}, Result: {result_state}")
        
        if state in ['TERMINATED', 'SKIPPED']:
            print(f"Job completed with state: {state}")
            if result_state:
                print(f"Result state: {result_state}")
            break
        
        if i % 6 == 0:  # Print every minute
            print(f"Waiting... ({i*10} seconds)")
    
    # Get the results
    if result_state == 'SUCCESS':
        print("Job succeeded!")
        # The notebook should have written the answer to workspace
        # Try to read it back
        try:
            answer_file = f"{workspace_dir}/answers.json"
            w.workspace.download(workspace_path=answer_file, local_path='submission/answers.json', overwrite=True)
            print("Answer downloaded to submission/answers.json")
            
            # Verify the answer
            with open('submission/answers.json', 'r') as f:
                answer = json.load(f)
            print(f"Answer: {answer}")
        except Exception as e:
            print(f"Failed to download answer: {e}")
    else:
        print(f"Job failed with state: {result_state}")
        
    # Clean up - delete the job
    try:
        w.jobs.delete(job_id)
        print(f"Job {job_id} deleted")
    except Exception as e:
        print(f"Failed to delete job: {e}")
        
except Exception as e:
    print(f"Job execution failed: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")
