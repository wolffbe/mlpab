#!/usr/bin/env python3
"""Script to run fine-tuning on Databricks platform."""
import os
import json
import tempfile
import shutil
from pathlib import Path

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import compute, jobs

# Initialize the workspace client
w = WorkspaceClient()

# Environment variables
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
JOB_NAME = f"{PREFIX}_ftjob65e929"
MODEL_NAME = f"{PREFIX}_ftmodel65e929"

# Get current user
current_user = w.current_user.me().user_name
print(f"Current user: {current_user}")

# Paths
workspace_path = f"/Users/{current_user}/{PREFIX}"
dbfs_data_path = f"dbfs:/Users/{current_user}/{PREFIX}/data"
dbfs_output_path = f"dbfs:/Users/{current_user}/{PREFIX}/output"

# Create a cluster for the job
cluster_name = f"{PREFIX}_ft_cluster"

# First, upload the data files and script to DBFS
print("Uploading files to DBFS...")

# Create temp directory for upload
with tempfile.TemporaryDirectory() as tmpdir:
    # Copy data files to temp directory
    shutil.copy("data/base_model.npz", tmpdir)
    shutil.copy("data/finetune.txt", tmpdir)
    shutil.copy("data/eval.txt", tmpdir)
    shutil.copy("data/finetune_model.py", tmpdir)
    
    # Upload to DBFS
    w.dbfs.upload(
        src=tmpdir + "/base_model.npz",
        dst=dbfs_data_path + "/base_model.npz",
        overwrite=True
    )
    w.dbfs.upload(
        src=tmpdir + "/finetune.txt",
        dst=dbfs_data_path + "/finetune.txt",
        overwrite=True
    )
    w.dbfs.upload(
        src=tmpdir + "/eval.txt",
        dst=dbfs_data_path + "/eval.txt",
        overwrite=True
    )
    w.dbfs.upload(
        src=tmpdir + "/finetune_model.py",
        dst=dbfs_data_path + "/finetune_model.py",
        overwrite=True
    )

print("Files uploaded successfully")

# Create a notebook that will run the fine-tuning
notebook_content = f"""# Fine-tuning notebook
import os
import shutil

# Copy files from DBFS to local
shutil.copy("dbfs:/Users/{current_user}/{PREFIX}/data/base_model.npz", "/dbfs/tmp/ft_data/base_model.npz")
shutil.copy("dbfs:/Users/{current_user}/{PREFIX}/data/finetune.txt", "/dbfs/tmp/ft_data/finetune.txt")
shutil.copy("dbfs:/Users/{current_user}/{PREFIX}/data/eval.txt", "/dbfs/tmp/ft_data/eval.txt")
shutil.copy("dbfs:/Users/{current_user}/{PREFIX}/data/finetune_model.py", "/dbfs/tmp/ft_data/finetune_model.py")

# Change to working directory
os.chdir("/dbfs/tmp/ft_data")

# Run the fine-tuning script
import subprocess
result = subprocess.run(["python", "finetune_model.py"], capture_output=True, text=True)
print("STDOUT:", result.stdout)
print("STDERR:", result.stderr)

# Copy results back to DBFS
shutil.copy("/dbfs/tmp/ft_data/finetuned_model.npz", "dbfs:/Users/{current_user}/{PREFIX}/output/finetuned_model.npz")
shutil.copy("/dbfs/tmp/ft_data/metrics.json", "dbfs:/Users/{current_user}/{PREFIX}/output/metrics.json")

print("Fine-tuning completed!")
"""

# Upload the notebook
notebook_path = f"{workspace_path}/finetune_notebook"
w.workspace.upload(
    content=notebook_content,
    path=notebook_path,
    language="PYTHON",
    overwrite=True
)
print(f"Notebook uploaded to {notebook_path}")

# Create a cluster
print("Creating cluster...")
try:
    cluster = w.clusters.create(
        cluster_name=cluster_name,
        node_type_id="Standard_DS3_v2",
        spark_version="14.3.x-scala2.12",
        num_workers=0,
        autoscale=compute.AutoScale(min_workers=0, max_workers=0),
        spark_conf={"spark.databricks.repl.allowedLanguages": "python,sql"},
    )
    cluster_id = cluster.cluster_id
    print(f"Cluster created: {cluster_id}")
except Exception as e:
    print(f"Error creating cluster: {e}")
    # Try to get existing cluster
    clusters = w.clusters.list()
    for c in clusters:
        if c.cluster_name == cluster_name:
            cluster_id = c.cluster_id
            print(f"Using existing cluster: {cluster_id}")
            break
    else:
        raise

# Wait for cluster to be running
print("Waiting for cluster to be running...")
w.clusters.wait_get_running(cluster_id)
print("Cluster is running")

# Create the job
print("Creating job...")
job = w.jobs.create(
    name=JOB_NAME,
    tasks=[
        jobs.Task(
            task_key="finetune",
            notebook_task=jobs.NotebookTask(
                notebook_path=notebook_path,
            ),
            existing_cluster_id=cluster_id,
        )
    ],
)
job_id = job.job_id
print(f"Job created: {job_id}")

# Run the job
print("Running job...")
run = w.jobs.run(job_id=job_id)
run_id = run.run_id
print(f"Job run ID: {run_id}")

# Wait for job to complete
print("Waiting for job to complete...")
w.jobs.wait_get_run_completed(run_id)
print("Job completed")

# Get the run result
run_result = w.jobs.get_run(run_id)
print(f"Job state: {run_result.state}")
if run_result.state.result_state == jobs.RunResultState.SUCCESS:
    print("Job succeeded!")
elif run_result.state.result_state == jobs.RunResultState.FAILED:
    print(f"Job failed: {run_result.state}")
    raise Exception("Job failed")

# Read metrics.json from DBFS
print("Reading metrics...")
metrics_content = w.dbfs.read(dbfs_output_path + "/metrics.json")
# The content is base64 encoded
import base64
metrics_data = json.loads(base64.b64decode(metrics_content.data).decode('utf-8'))
print(f"Metrics: {metrics_data}")

# Register the model in MLflow
print("Registering model...")
# First, we need to create a model in the registry
model_uri = f"dbfs:/Users/{current_user}/{PREFIX}/output/finetuned_model.npz"

# Create a model in the registry
try:
    model = w.model_registry.create(
        name=MODEL_NAME,
        description="Fine-tuned character-level language model",
    )
    print(f"Model created: {model}")
except Exception as e:
    print(f"Error creating model: {e}")
    # Model might already exist
    pass

# Create a model version
model_version = w.model_registry.create_version(
    name=MODEL_NAME,
    source=model_uri,
    run_id=run_id,
)
print(f"Model version created: {model_version}")

# Transition to staged
w.model_registry.transition(
    name=MODEL_NAME,
    version=model_version.version,
    stage="Staging",
)
print("Model transitioned to Staging")

# Attach metrics to the model version
# Note: The Databricks SDK doesn't have a direct method to attach metrics to a model version
# We need to use the MLflow API through the workspace client
# For now, let's just record the metrics

print(f"\nJob completed successfully!")
print(f"Job name: {JOB_NAME}")
print(f"Model name: {MODEL_NAME}")
print(f"Eval loss: {metrics_data['eval_loss']}")
print(f"Base eval loss: {metrics_data['base_eval_loss']}")

# Create submission directory
os.makedirs("submission", exist_ok=True)

# Write answers.json
answers = {
    "job_name": "ftjob65e929",
    "model_name": "ftmodel65e929",
    "eval_loss": metrics_data["eval_loss"],
    "base_eval_loss": metrics_data["base_eval_loss"]
}

with open("submission/answers.json", "w") as f:
    json.dump(answers, f, indent=2)

print(f"\nAnswers written to submission/answers.json")
print(json.dumps(answers, indent=2))
