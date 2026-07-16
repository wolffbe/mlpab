#!/usr/bin/env python3
import os
import json
import time
from databricks.sdk import WorkspaceClient

# Initialize workspace client
w = WorkspaceClient()

# Get environment variables
schema_name = os.environ.get('MLPABRICKS_SCHEMA', 'workspace.mlpabc6cfe1')
prefix = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpabc6cfe1')

# Get current user
try:
    user = w.current_user.me()
    username = user.user_name
    print(f"Username: {username}")
except Exception as e:
    print(f"Failed to get username: {e}")
    username = "unknown"

# Upload the CSV file to workspace
workspace_dir = f"/Users/{username}/{prefix}"
workspace_csv_path = f"{workspace_dir}/features.csv"

print(f"Uploading data/features.csv to {workspace_csv_path}")
try:
    w.workspace.upload(
        local_path='data/features.csv',
        workspace_path=workspace_csv_path,
        overwrite=True
    )
    print("Upload successful")
except Exception as e:
    print(f"Upload failed: {e}")
    import traceback
    traceback.print_exc()

# Create a notebook for analysis
notebook_content = """# Databricks notebook source
# MAGIC %python
import csv
import json
import os
from collections import defaultdict

# Find the features.csv file
print("Looking for features.csv...")
features_file = None

# Try in current directory
if os.path.exists('features.csv'):
    features_file = 'features.csv'
else:
    # Try in parent directories
    for root, dirs, files in os.walk('.'):
        if 'features.csv' in files:
            features_file = os.path.join(root, 'features.csv')
            break

if not features_file:
    # Try absolute path
    features_file = '/Workspace/Users/""" + username + f"""/{prefix}/features.csv"

print(f"Using file: {features_file}")

# Read the CSV
features_data = []
with open(features_file, 'r') as f:
    reader = csv.DictReader(f)
    for row in reader:
        features_data.append(row)

print(f"Loaded {len(features_data)} rows")

# Group by date and calculate statistics
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
answer_file = '/Workspace/Users/""" + username + f"""/{prefix}/answers.json"
with open(answer_file, 'w') as f:
    json.dump(answer, f)

print(f"Answer written: {answer}")
print(f"Answer file: {answer_file}")

# Also write to dbfs for easier retrieval
dbfs_answer_file = f'/dbfs/FileStore/tables/{prefix}/answers.json'
with open(dbfs_answer_file, 'w') as f:
    json.dump(answer, f)

print(f"Answer also written to: {dbfs_answer_file}")
"""

notebook_path = f"{workspace_dir}/drift_analysis"
print(f"Creating notebook at: {notebook_path}")

try:
    w.workspace.upload(
        local_path='drift_analysis_notebook.py',
        workspace_path=f"{notebook_path}.py",
        overwrite=True,
        format='SOURCE'  # Upload as notebook source
    )
    print("Notebook uploaded")
except Exception as e:
    print(f"Notebook upload failed: {e}")
    # Try creating it directly
    try:
        w.workspace.mkdirs(workspace_dir)
        # Write the notebook content directly
        with open('temp_notebook.py', 'w') as f:
            f.write(notebook_content)
        w.workspace.upload(
            local_path='temp_notebook.py',
            workspace_path=f"{notebook_path}.py",
            overwrite=True
        )
        print("Notebook created via temp file")
        os.unlink('temp_notebook.py')
    except Exception as e2:
        print(f"Second attempt failed: {e2}")

# Now run the notebook
print(f"\nRunning notebook...")
try:
    # Create a job to run the notebook
    job_name = f"{prefix}_drift_analysis"
    
    # First, check if we can run it directly
    # Use the jobs API
    from databricks.sdk.service import jobs
    
    # Create a new job
    new_job = w.jobs.create(
        name=job_name,
        tasks=[
            jobs.Task(
                task_key="drift_analysis",
                notebook_task=jobs.NotebookTask(
                    notebook_path=f"/Users/{username}/{prefix}/drift_analysis",
                ),
                existing_cluster_id="",  # Use a new cluster
                new_cluster=jobs.NewCluster(
                    spark_version="14.3.x-scala2.12",
                    node_type_id="Standard_DS3_v2",
                    num_workers=1
                )
            )
        ]
    )
    print(f"Job created: {new_job.job_id}")
    
    # Run the job
    run = w.jobs.run_now(new_job.job_id)
    print(f"Job run started: {run.run_id}")
    
    # Wait for completion
    print("Waiting for job to complete...")
    max_wait = 300  # 5 minutes
    for i in range(max_wait):
        time.sleep(10)
        run_info = w.jobs.get_run(run.run_id)
        if run_info.state.life_cycle_state in ['TERMINATED', 'SKIPPED']:
            print(f"Job completed with state: {run_info.state.life_cycle_state}")
            print(f"Result state: {run_info.state.result_state}")
            break
        if i % 6 == 0:  # Print every minute
            print(f"Waiting... ({i*10} seconds)")
    
    # Get the results
    if run_info.state.result_state == 'SUCCESS':
        print("Job succeeded!")
        # The notebook should have written the answer to workspace
        # Try to read it back
        try:
            answer_file = f"{workspace_dir}/answers.json"
            w.workspace.download(workspace_path=answer_file, local_path='submission/answers.json', overwrite=True)
            print("Answer downloaded to submission/answers.json")
        except Exception as e:
            print(f"Failed to download answer: {e}")
    else:
        print(f"Job failed with state: {run_info.state.result_state}")
        
except Exception as e:
    print(f"Job execution failed: {e}")
    import traceback
    traceback.print_exc()

print("\nDone!")
