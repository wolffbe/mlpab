#!/usr/bin/env python3
import os
import json
import time
import base64
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat

# Initialize workspace client
w = WorkspaceClient()

# Get environment variables
prefix = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpabc6cfe1')
username = "benedict@logicalclocks.com"
workspace_dir = f"/Users/{username}/{prefix}"

# Create a simple notebook that does the analysis
notebook_content = """# Databricks notebook source
# MAGIC %python
import csv
import json
import os
from collections import defaultdict

# Read the features.csv file from workspace
features_file = '/Workspace/Users/benedict@logicalclocks.com/mlpabc6cfe1/features.csv'

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

print(f"Max absolute change: {max_feature} on {max_date} with change {max_abs_change:.4f}")

# Also check relative change
max_rel_change = 0
max_rel_feature = None
max_rel_date = None

for feature in features:
    for i in range(1, len(daily_stats)):
        prev_avg = daily_stats[i-1][f'avg_{feature}']
        curr_avg = daily_stats[i][f'avg_{feature}']
        abs_change = abs(curr_avg - prev_avg)
        rel_change = abs_change / abs(prev_avg) if prev_avg != 0 else 0
        if rel_change > max_rel_change:
            max_rel_change = rel_change
            max_rel_feature = feature
            max_rel_date = daily_stats[i]['date']

print(f"Max relative change: {max_rel_feature} on {max_rel_date} with change {max_rel_change:.4f}")

# Use absolute change to determine drift (more reliable for this case)
answer = {
    "feature": max_feature,
    "onset": max_date
}

# Write to workspace
answer_file = '/Workspace/Users/benedict@logicalclocks.com/mlpabc6cfe1/answers.json'
with open(answer_file, 'w') as f:
    json.dump(answer, f)

print(f"Answer: {answer}")
"""

# Upload the notebook
notebook_path = f"{workspace_dir}/drift_analysis"
notebook_content_b64 = base64.b64encode(notebook_content.encode('utf-8')).decode('utf-8')

try:
    w.workspace.import_(
        path=f"{notebook_path}.py",
        content=notebook_content_b64,
        format=ImportFormat.SOURCE,
        overwrite=True
    )
    print(f"Notebook uploaded")
except Exception as e:
    print(f"Notebook upload failed: {e}")

# Run the notebook using jobs API
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
                new_cluster=jobs.NewCluster(
                    spark_version="14.3.x-scala2.12",
                    node_type_id="Standard_DS3_v2",
                    num_workers=0
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
    for i in range(12):  # Wait up to 2 minutes
        time.sleep(10)
        run_info = w.jobs.get_run(run_id)
        state = run_info.state.life_cycle_state
        result_state = run_info.state.result_state if hasattr(run_info.state, 'result_state') else None
        
        if state in ['TERMINATED', 'SKIPPED']:
            print(f"Job completed: state={state}, result={result_state}")
            break
        print(f"  Waiting... ({i*10}s)")
    
    # Get the results
    if result_state == 'SUCCESS':
        print("Job succeeded!")
        # Download the answer
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
    else:
        print(f"Job failed: {result_state}")
    
    # Clean up
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
