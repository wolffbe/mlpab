#!/usr/bin/env python3
"""Script to run fine-tuning on Databricks platform and register the model."""

import os
import json
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace, jobs, compute

def main():
    # Initialize workspace client
    wc = WorkspaceClient()
    
    # Get current user
    current_user = wc.current_user.me()
    user_email = current_user.emails[0].value
    
    # Get environment variables
    prefix = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpab3f7490')
    schema = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpab3f7490')
    
    job_name = "ftjob65e929"
    model_name = "ftmodel65e929"
    
    # Full job name with prefix
    full_job_name = f"{prefix}_{job_name}"
    full_model_name = f"{prefix}_{model_name}"
    
    # Path in workspace for files - use the current user's email
    workspace_path = f"/Users/{user_email}/{prefix}"
    
    print(f"User: {user_email}")
    print(f"Job name: {full_job_name}")
    print(f"Model name: {full_model_name}")
    print(f"Workspace path: {workspace_path}")
    print(f"Schema: {schema}")
    
    # Step 1: Upload data files to workspace
    print("\n=== Uploading files to workspace ===")
    
    # Create workspace directory
    wc.workspace.mkdirs(workspace_path)
    
    # Upload all data files
    data_files = ['base_model.npz', 'finetune.txt', 'eval.txt', 'finetune_model.py']
    for file in data_files:
        local_path = f"data/{file}"
        remote_path = f"{workspace_path}/{file}"
        print(f"Uploading {local_path} to {remote_path}")
        with open(local_path, 'rb') as f:
            content = f.read()
            # Use RAW format for binary files
            wc.workspace.upload(remote_path, content, format=workspace.ImportFormat.RAW, overwrite=True)
    
    print("Files uploaded successfully")
    
    # Step 2: Create and run the job using SparkPythonTask
    print("\n=== Creating job ===")
    
    # Create the task using SparkPythonTask
    task = jobs.Task(
        task_key="finetune_task",
        spark_python_task=jobs.SparkPythonTask(
            python_file=f"workspace:{workspace_path}/finetune_model.py"
        ),
        # Don't specify libraries here for serverless
    )
    
    # Create the job
    job = wc.jobs.create(
        name=full_job_name,
        tasks=[task],
        max_concurrent_runs=1,
    )
    
    job_id = job.job_id
    print(f"Job created with ID: {job_id}")
    
    # Step 3: Run the job and wait for completion
    print("\n=== Running job ===")
    
    run = wc.jobs.run_now(job_id)
    run_id = run.run_id
    print(f"Job run started with ID: {run_id}")
    
    # Wait for job to complete
    print("Waiting for job to complete...")
    while True:
        run_info = wc.jobs.get_run(run_id)
        state = run_info.state
        result_state = state.result_state if state else None
        
        if result_state in [jobs.RunResultState.SUCCESS, jobs.RunResultState.FAILED, jobs.RunResultState.TIMEDOUT, jobs.RunResultState.CANCELED]:
            print(f"Job completed with state: {result_state}")
            break
        
        print(f"Current state: {state.life_cycle_state if state else 'UNKNOWN'}")
        time.sleep(10)
    
    if result_state != jobs.RunResultState.SUCCESS:
        print(f"Job failed with state: {result_state}")
        # Get run output for debugging
        try:
            output = wc.jobs.get_run_output(run_id)
            print("Job output:")
            print(output)
        except Exception as e:
            print(f"Could not get job output: {e}")
        raise Exception(f"Job failed with state: {result_state}")
    
    print("Job completed successfully!")
    
    # Step 4: Download the results
    print("\n=== Downloading results ===")
    
    # The job runs in the workspace, so outputs should be there
    try:
        # Try to download metrics.json from workspace
        local_metrics_path = "metrics.json"
        remote_metrics_path = f"{workspace_path}/metrics.json"
        if wc.workspace.get_status(remote_metrics_path):
            with open(local_metrics_path, "wb") as f:
                f.write(wc.workspace.download(remote_metrics_path))
            print(f"Downloaded metrics.json from workspace")
        else:
            print(f"metrics.json not found at {remote_metrics_path}")
        
        # Try to download finetuned_model.npz from workspace
        local_model_path = "finetuned_model.npz"
        remote_model_path = f"{workspace_path}/finetuned_model.npz"
        if wc.workspace.get_status(remote_model_path):
            with open(local_model_path, "wb") as f:
                f.write(wc.workspace.download(remote_model_path))
            print(f"Downloaded finetuned_model.npz from workspace")
        else:
            print(f"finetuned_model.npz not found at {remote_model_path}")
            
    except Exception as e:
        print(f"Could not download files: {e}")
        raise
    
    # Read metrics
    with open("metrics.json", "r") as f:
        metrics = json.load(f)
    
    print(f"Metrics: {metrics}")
    eval_loss = metrics["eval_loss"]
    base_eval_loss = metrics["base_eval_loss"]
    
    # Step 5: Register the model in Unity Catalog
    print("\n=== Registering model ===")
    
    # First, upload the model file to a location accessible by Unity Catalog
    # Use DBFS under /tmp which should be accessible
    model_dbfs_path = f"/tmp/{prefix}/finetuned_model.npz"
    
    # Create DBFS directory
    try:
        wc.dbfs.mkdirs(f"/tmp/{prefix}")
    except Exception as e:
        print(f"Could not create DBFS directory: {e}")
    
    # Upload model to DBFS
    with open("finetuned_model.npz", 'rb') as f:
        wc.dbfs.upload(f, model_dbfs_path)
    print(f"Uploaded model to DBFS: {model_dbfs_path}")
    
    # Create the registered model
    from databricks.sdk.service import catalog
    
    # Split schema into catalog and schema name
    catalog_name, schema_name = schema.split('.')
    
    # Create the registered model
    model = wc.registered_models.create(
        name=full_model_name,
        catalog_name=catalog_name,
        schema_name=schema_name,
        description="Fine-tuned character-level language model",
    )
    
    model_id = model.name  # This should be the full name
    print(f"Registered model: {model_id}")
    
    # Create model version
    from datetime import datetime
    
    version = wc.model_versions.update(
        name=full_model_name,
        version=1,
        catalog_name=catalog_name,
        schema_name=schema_name,
        description="Version 1 of fine-tuned model",
        storage_location=f"dbfs:{model_dbfs_path}",
        tags={
            "eval_loss": str(eval_loss),
            "base_eval_loss": str(base_eval_loss),
        },
        custom_headers={
            "eval_loss": str(eval_loss),
            "base_eval_loss": str(base_eval_loss),
        },
    )
    
    print(f"Created model version: {version.version}")
    
    # Step 6: Write answers.json
    print("\n=== Writing answers.json ===")
    
    answers = {
        "job_name": job_name,
        "model_name": model_name,
        "eval_loss": eval_loss,
        "base_eval_loss": base_eval_loss,
    }
    
    # Create submission directory if it doesn't exist
    os.makedirs("submission", exist_ok=True)
    
    with open("submission/answers.json", "w") as f:
        json.dump(answers, f, indent=2)
    
    print(f"Answers written to submission/answers.json")
    print(f"Content: {answers}")
    
    print("\n=== All steps completed successfully! ===")

if __name__ == "__main__":
    main()
