#!/usr/bin/env python3
"""Script to run fine-tuning on Databricks platform."""
import os
import json

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
from databricks.sdk.service.workspace import ImportFormat, Language

# Initialize the workspace client
w = WorkspaceClient()

# Environment variables
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
JOB_NAME = f"{PREFIX}_ftjob65e929"
MODEL_NAME = f"{PREFIX}_ftmodel65e929"

# Get current user
current_user = w.current_user.me().user_name
print(f"Current user: {current_user}")

# Paths - use workspace paths
workspace_path = f"/Users/{current_user}/{PREFIX}"

# Get catalog and schema for Unity Catalog
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
catalog_name, schema_name = SCHEMA.split(".")
full_model_name = f"{catalog_name}.{schema_name}.{MODEL_NAME}"

# Create directories first
print("Creating directories...")
w.workspace.mkdirs(workspace_path + "/data")
w.workspace.mkdirs(workspace_path + "/output")

# First, upload the data files and script to workspace
print("Uploading files to workspace...")

# Read and upload base_model.npz as bytes with RAW format
with open("data/base_model.npz", "rb") as f:
    base_model_content = f.read()
w.workspace.upload(
    path=workspace_path + "/data/base_model.npz",
    content=base_model_content,
    format=ImportFormat.RAW,
    overwrite=True
)

# Read and upload finetune.txt
with open("data/finetune.txt", "rb") as f:
    finetune_content = f.read()
w.workspace.upload(
    path=workspace_path + "/data/finetune.txt",
    content=finetune_content,
    format=ImportFormat.RAW,
    overwrite=True
)

# Read and upload eval.txt
with open("data/eval.txt", "rb") as f:
    eval_content = f.read()
w.workspace.upload(
    path=workspace_path + "/data/eval.txt",
    content=eval_content,
    format=ImportFormat.RAW,
    overwrite=True
)

# Read and upload finetune_model.py
with open("data/finetune_model.py", "rb") as f:
    script_content = f.read()
w.workspace.upload(
    path=workspace_path + "/data/finetune_model.py",
    content=script_content,
    format=ImportFormat.RAW,
    overwrite=True
)

print("Files uploaded successfully")

# Create a notebook that will run the fine-tuning and register the model
notebook_content = f"""# Fine-tuning notebook
import os
import sys
import json
import shutil
import subprocess

# Setup paths
working_dir = "/tmp/ft_data_65e929"
os.makedirs(working_dir, exist_ok=True)

# Copy files from workspace to local
# In Databricks, workspace files are at /Workspace/Users/...
shutil.copy("/Workspace{workspace_path}/data/base_model.npz", working_dir + "/base_model.npz")
shutil.copy("/Workspace{workspace_path}/data/finetune.txt", working_dir + "/finetune.txt")
shutil.copy("/Workspace{workspace_path}/data/eval.txt", working_dir + "/eval.txt")
shutil.copy("/Workspace{workspace_path}/data/finetune_model.py", working_dir + "/finetune_model.py")

# Change to working directory
os.chdir(working_dir)

# Run the fine-tuning script
result = subprocess.run([sys.executable, "finetune_model.py"], capture_output=True, text=True)
print("STDOUT:", result.stdout)
if result.stderr:
    print("STDERR:", result.stderr)

# Read metrics
with open(working_dir + "/metrics.json", "r") as f:
    metrics = json.load(f)

print(f"Metrics: {{metrics}}")

# Log metrics to MLflow and register model in Unity Catalog
import mlflow
mlflow.set_experiment("/Users/{current_user}/{PREFIX}/ft_experiment")

with mlflow.start_run() as run:
    # Log metrics
    mlflow.log_metric("eval_loss", metrics["eval_loss"])
    mlflow.log_metric("base_eval_loss", metrics["base_eval_loss"])
    
    # Log the model file
    mlflow.log_artifact(working_dir + "/finetuned_model.npz")
    
    # Get run ID
    run_id = run.info.run_id
    print(f"MLflow run ID: {{run_id}}")
    
    # Register the model with version 1 in Unity Catalog
    model_uri = f"runs:/{{run_id}}/finetuned_model.npz"
    mv = mlflow.register_model(
        model_uri=model_uri,
        name="{full_model_name}"
    )
    print(f"Model version: {{mv}}")

# Copy results back to workspace for verification
shutil.copy(working_dir + "/finetuned_model.npz", "/Workspace{workspace_path}/output/finetuned_model.npz")
shutil.copy(working_dir + "/metrics.json", "/Workspace{workspace_path}/output/metrics.json")

print("Fine-tuning and model registration completed!")
"""

# Upload the notebook
notebook_path = f"{workspace_path}/finetune_notebook_65e929"
w.workspace.upload(
    content=notebook_content,
    path=notebook_path,
    language=Language.PYTHON,
    overwrite=True
)
print(f"Notebook uploaded to {notebook_path}")

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
        )
    ],
)
job_id = job.job_id
print(f"Job created: {job_id}")

# Run the job
print("Running job...")
run = w.jobs.run_now(job_id=job_id)
run_id = run.run_id
print(f"Job run ID: {run_id}")

# Wait for job to complete
print("Waiting for job to complete...")
w.jobs.wait_get_run_job_terminated_or_skipped(run_id)
print("Job completed")

# Get the run result
run_result = w.jobs.get_run(run_id)
print(f"Job state: {run_result.state}")
if run_result.state.result_state == jobs.RunResultState.SUCCESS:
    print("Job succeeded!")
elif run_result.state.result_state in [jobs.RunResultState.FAILED, jobs.RunResultState.INTERNAL_ERROR]:
    print(f"Job failed: {run_result.state}")
    raise Exception("Job failed")

# Read metrics.json from workspace
print("Reading metrics...")
metrics_response = w.workspace.download(workspace_path + "/output/metrics.json")
metrics_content = metrics_response.read()
if isinstance(metrics_content, bytes):
    metrics_content = metrics_content.decode('utf-8')
metrics_data = json.loads(metrics_content)
print(f"Metrics: {metrics_data}")

# Verify the model was registered in Unity Catalog
print("Verifying model registration...")
try:
    model = w.registered_models.get(full_name=full_model_name)
    print(f"Model found: {model.full_name}")
    
    # List model versions
    model_versions = list(w.model_versions.list(name=full_model_name))
    print(f"Model versions: {len(model_versions)}")
    if model_versions:
        print(f"Latest version: {model_versions[0].version}")
except Exception as e:
    print(f"Error verifying model: {e}")

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
print("\nDone!")
