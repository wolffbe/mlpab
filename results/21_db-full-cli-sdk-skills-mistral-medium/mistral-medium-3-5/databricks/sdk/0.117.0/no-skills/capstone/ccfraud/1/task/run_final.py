#!/usr/bin/env python3
"""
Upload and run the fraud detection notebook.
"""

import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import compute, workspace

# Environment variables
SCHEMA = os.environ.get("MLPAB_DATABRICKS_SCHEMA", "workspace.mlpab958e4d")
PREFIX = os.environ.get("MLPAB_DATABRICKS_PREFIX", "mlpab958e4d")

print(f"Schema: {SCHEMA}")
print(f"Prefix: {PREFIX}")

# Initialize workspace client
w = WorkspaceClient()
user = w.current_user.me()
current_user = user.user_name or "benedict@logicalclocks.com"
print(f"Current user: {current_user}")

# Step 1: Upload the notebook
print("\n=== Uploading notebook ===")
with open("fraud_notebook.py", "r") as f:
    notebook_content = f.read()

notebook_path = f"/Users/{current_user}/{PREFIX}/fraud_pipeline"
w.workspace.upload(path=notebook_path, content=notebook_content, language=workspace.Language.PYTHON, overwrite=True)
print(f"✓ Uploaded notebook: {notebook_path}")

# Step 2: Create cluster
print("\n=== Creating cluster ===")
cluster_name = f"{PREFIX}_cluster"

try:
    clusters = w.clusters.list()
    cluster = next((c for c in clusters if c.cluster_name == cluster_name), None)
    
    if cluster:
        cluster_id = cluster.cluster_id
        print(f"✓ Using existing cluster: {cluster_name}")
    else:
        cluster = w.clusters.create(
            cluster_name=cluster_name,
            node_type_id="Standard_DS3_v2",
            spark_version="14.3.x-scala2.12",
            num_workers=2
        )
        cluster_id = cluster.cluster_id
        print(f"✓ Created cluster: {cluster_name}")
        
        # Wait for cluster
        while True:
            info = w.clusters.get(cluster_id=cluster_id)
            if info.state == "RUNNING":
                break
            elif info.state == "ERROR":
                print(f"✗ Cluster error: {info.state_message}")
                exit(1)
            time.sleep(10)
        print("✓ Cluster running")
        
except Exception as e:
    print(f"✗ Cluster error: {e}")
    exit(1)

# Step 3: Run notebook as job
print("\n=== Running job ===")
job_name = f"{PREFIX}_fraud_job"

try:
    job = w.jobs.create(
        name=job_name,
        tasks=[{
            "task_key": "run_pipeline",
            "notebook_task": {"notebook_path": notebook_path},
            "existing_cluster_id": cluster_id
        }]
    )
    print(f"✓ Created job: {job.job_id}")
    
    # Run job
    run = w.jobs.run_now(job_id=job.job_id)
    print(f"✓ Started run: {run.run_id}")
    
    # Wait for completion
    max_wait = 3600
    start = time.time()
    
    while time.time() - start < max_wait:
        time.sleep(30)
        run_info = w.jobs.get_run(run_id=run.run_id)
        state = run_info.state.life_cycle_state
        result = getattr(run_info.state, 'result_state', None)
        
        if state == "TERMINATED":
            if result == "SUCCESS":
                print("✓ Job succeeded!")
                break
            else:
                print(f"✗ Job failed: {result}")
                if hasattr(run_info.state, 'state_message'):
                    print(f"Error: {run_info.state.state_message}")
                break
        elif state in ["SKIPPED", "INTERNAL_ERROR"]:
            print(f"✗ Job {state}")
            break
        else:
            print(f"  Status: {state}")
    
    if time.time() - start >= max_wait:
        print("✗ Job timed out")

except Exception as e:
    print(f"✗ Job error: {e}")

print("\n=== Done ===")
