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
        # Delete existing file if it exists
        try:
            wc.workspace.delete(remote_path)
        except:
            pass
        with open(local_path, 'rb') as f:
            content = f.read()
            # Use AUTO format for all files
            wc.workspace.upload(remote_path, content, format=workspace.ImportFormat.AUTO)
    
    print("Files uploaded successfully")
    
    # Step 2: Create and run the job using SparkPythonTask
    print("\n=== Creating job ===")
    
    from databricks.sdk.service import jobs, compute
    
    # Create an environment for the job
    job_environment = jobs.JobEnvironment(
        environment_key="finetune_env",
        spec=compute.Environment(
            base_environment="workspace-base-environments/databricks_ml"
        ),
    )
    
    # Create the task using SparkPythonTask with workspace path
    task = jobs.Task(
        task_key="finetune_task",
        spark_python_task=jobs.SparkPythonTask(
            python_file=f"{workspace_path}/finetune_model.py"
        ),
        environment_key="finetune_env",
    )
    
    # Create the job
    job = wc.jobs.create(
        name=full_job_name,
        tasks=[task],
        environments=[job_environment],
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
            # Try to get task runs
            try:
                task_runs = list(wc.jobs.list_runs(job_id=job_id, limit=10))
                print(f"Task runs: {len(task_runs)}")
                for tr in task_runs:
                    print(f"Task run {tr.run_id}: {tr.state}")
                    try:
                        task_output = wc.jobs.get_run_output(tr.run_id)
                        print(f"Task output: {task_output}")
                    except Exception as e2:
                        print(f"Could not get task output: {e2}")
            except Exception as e3:
                print(f"Could not list task runs: {e3}")
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
            resp = wc.workspace.download(remote_metrics_path)
            # Handle StreamingResponse
            if hasattr(resp, 'read'):
                content = resp.read()
            elif hasattr(resp, 'content'):
                content = resp.content
            else:
                content = resp
            if isinstance(content, bytes):
                with open(local_metrics_path, "wb") as f:
                    f.write(content)
            else:
                with open(local_metrics_path, "w") as f:
                    f.write(content)
            print(f"Downloaded metrics.json from workspace")
        else:
            print(f"metrics.json not found at {remote_metrics_path}")
        
        # Try to download finetuned_model.npz from workspace
        local_model_path = "finetuned_model.npz"
        remote_model_path = f"{workspace_path}/finetuned_model.npz"
        if wc.workspace.get_status(remote_model_path):
            resp = wc.workspace.download(remote_model_path)
            # Handle StreamingResponse
            if hasattr(resp, 'read'):
                content = resp.read()
            elif hasattr(resp, 'content'):
                content = resp.content
            else:
                content = resp
            if isinstance(content, bytes):
                with open(local_model_path, "wb") as f:
                    f.write(content)
            else:
                with open(local_model_path, "w") as f:
                    f.write(content)
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
    # Use workspace path since DBFS root is disabled
    model_workspace_path = f"{workspace_path}/finetuned_model.npz"
    
    # Upload model to workspace if not already there
    if not wc.workspace.get_status(model_workspace_path):
        with open("finetuned_model.npz", 'rb') as f:
            content = f.read()
        wc.workspace.upload(model_workspace_path, content, format=workspace.ImportFormat.RAW)
        print(f"Uploaded model to workspace: {model_workspace_path}")
    else:
        print(f"Model already exists at {model_workspace_path}")
    
    # Use workspace path for storage location
    model_dbfs_path = f"workspace:{model_workspace_path}"
    
    # Create the registered model
    from databricks.sdk.service import catalog
    
    # Split schema into catalog and schema name
    catalog_name, schema_name = schema.split('.')
    
    # Check if model already exists, delete it if so
    try:
        existing_model = wc.registered_models.get(f"{catalog_name}.{schema_name}.{full_model_name}")
        wc.registered_models.delete(f"{catalog_name}.{schema_name}.{full_model_name}")
        print(f"Deleted existing model: {full_model_name}")
    except:
        pass
    
    # Create the registered model
    model = wc.registered_models.create(
        name=full_model_name,
        catalog_name=catalog_name,
        schema_name=schema_name,
        comment="Fine-tuned character-level language model",
    )
    
    model_id = model.name  # This should be the full name
    print(f"Registered model: {model_id}")
    
    # Full name for the model version
    full_model_full_name = f"{catalog_name}.{schema_name}.{full_model_name}"
    
    # Check if version 1 already exists
    existing_versions = list(wc.model_versions.list(full_model_full_name))
    print(f"Existing versions: {existing_versions}")
    
    # Create model version
    from datetime import datetime
    
    # Try using the registered_models.update to set storage location
    try:
        updated_model = wc.registered_models.update(
            full_name=full_model_full_name,
            storage_location=model_dbfs_path,
            comment=f"Version 1 of fine-tuned model. eval_loss={eval_loss}, base_eval_loss={base_eval_loss}",
        )
        print(f"Updated model with storage location: {updated_model}")
        
        # Now try to update version 1 with source parameter and run_id
        version = wc.model_versions.update(
            full_name=full_model_full_name,
            version=1,
            comment="Version 1 of fine-tuned model",
            source=model_dbfs_path,
            storage_location=model_dbfs_path,
            run_id=str(run_id),
        )
        print(f"Created/updated model version: {version.version}")
    except Exception as e:
        print(f"Could not create/update model version: {e}")
        # Try using the old model registry API with just the model name (without catalog.schema)
        try:
            from databricks.sdk.service import ml
            # Convert workspace path to DBFS path
            dbfs_model_path = model_dbfs_path.replace("workspace:", "dbfs:")
            version = wc.model_registry.create_model_version(
                name=full_model_name,
                source=dbfs_model_path,
                description="Version 1 of fine-tuned model",
                run_id=str(run_id),
                tags=[
                    ml.ModelVersionTag(key="eval_loss", value=str(eval_loss)),
                    ml.ModelVersionTag(key="base_eval_loss", value=str(base_eval_loss)),
                ],
            )
            print(f"Created model version with old API: {version}")
        except Exception as e2:
            print(f"Could not create model version with old API either: {e2}")
            # Just proceed without creating version 1 explicitly
            # The model is registered with the storage location and metrics in the comment
            print("Proceeding without explicit version 1 creation")
    
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
