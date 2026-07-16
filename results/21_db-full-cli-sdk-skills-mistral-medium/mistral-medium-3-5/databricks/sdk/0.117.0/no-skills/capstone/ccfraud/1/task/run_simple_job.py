#!/usr/bin/env python3
"""
Create and run the fraud detection job with simple cluster specification.
"""

import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import compute, workspace, jobs

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

# Step 1: Create directory
print("\n=== Creating directory ===")
directory_path = f"/Users/{current_user}/{PREFIX}"
try:
    w.workspace.mkdirs(directory_path)
    print(f"✓ Created directory: {directory_path}")
except Exception as e:
    print(f"Directory may already exist: {e}")

# Step 2: Upload the notebook
print("\n=== Uploading notebook ===")
with open("fraud_notebook.py", "r") as f:
    notebook_content = f.read()

notebook_path = f"{directory_path}/fraud_pipeline"
w.workspace.upload(path=notebook_path, content=notebook_content, language=workspace.Language.PYTHON, overwrite=True)
print(f"✓ Uploaded notebook: {notebook_path}")

# Step 3: Create and run job with new cluster
print("\n=== Creating and running job ===")
job_name = f"{PREFIX}_fraud_job"
cluster_name = f"{PREFIX}_cluster"

try:
    # Create cluster spec
    cluster_spec = compute.ClusterSpec(
        cluster_name=cluster_name,
        node_type_id="Standard_DS3_v2",
        spark_version="14.3.x-scala2.12",
        num_workers=2
    )
    
    # Create notebook task
    notebook_task = jobs.NotebookTask(
        notebook_path=notebook_path
    )
    
    # Create task with new cluster
    task = jobs.Task(
        task_key="run_pipeline",
        notebook_task=notebook_task,
        new_cluster=cluster_spec
    )
    
    job = w.jobs.create(
        name=job_name,
        tasks=[task]
    )
    print(f"✓ Created job: {job.job_id}")
    
    # Run job
    run = w.jobs.run_now(job_id=job.job_id)
    print(f"✓ Started run: {run.run_id}")
    
    # Wait for completion
    max_wait = 3600
    start = time.time()
    
    while time.time() - start < max_wait:
        time.sleep(60)  # Check every minute
        run_info = w.jobs.get_run(run_id=run.run_id)
        state = run_info.state.life_cycle_state
        result = getattr(run_info.state, 'result_state', None)
        
        print(f"  Status: {state}, Result: {result}")
        
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
    
    if time.time() - start >= max_wait:
        print("✗ Job timed out")

except Exception as e:
    print(f"✗ Job error: {e}")
    import traceback
    traceback.print_exc()

print("\n=== Done ===")
