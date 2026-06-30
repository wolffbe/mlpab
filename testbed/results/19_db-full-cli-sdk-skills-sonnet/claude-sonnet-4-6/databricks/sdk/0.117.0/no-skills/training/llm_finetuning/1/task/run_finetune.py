"""Orchestrates fine-tuning job on Databricks and registers the model."""
import io
import json
import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog as cat_svc
from databricks.sdk.service import jobs as jobs_svc
from databricks.sdk.service import compute as compute_svc
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()

PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]       # mlpab99a3d4
SCHEMA_FULL = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # workspace.mlpab99a3d4
CATALOG_NAME, SCHEMA_NAME = SCHEMA_FULL.split(".")

USER = w.current_user.me().user_name
WORKSPACE_DIR = f"/Users/{USER}/{PREFIX}"
VOLUME_NAME = "ftvol"
VOLUME_PATH = f"/Volumes/{CATALOG_NAME}/{SCHEMA_NAME}/{VOLUME_NAME}"

JOB_DISPLAY_NAME = f"{PREFIX}_ftjob0b3133"
MODEL_FULL_NAME = f"{SCHEMA_FULL}.ftmodel0b3133"
EXPERIMENT_PATH = f"{WORKSPACE_DIR}/finetune_experiment"

print(f"User: {USER}")
print(f"Job display name: {JOB_DISPLAY_NAME}")
print(f"Model: {MODEL_FULL_NAME}")
print(f"Volume path: {VOLUME_PATH}")

# ---------------------------------------------------------------------------
# 1. Ensure volume exists
# ---------------------------------------------------------------------------
print("\n[1] Creating/checking UC volume...")
try:
    vol = w.volumes.create(
        catalog_name=CATALOG_NAME,
        schema_name=SCHEMA_NAME,
        name=VOLUME_NAME,
        volume_type=cat_svc.VolumeType.MANAGED,
    )
    print(f"    Created volume: {vol.full_name}")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"    Volume already exists, continuing")
    else:
        raise

# ---------------------------------------------------------------------------
# 2. Upload data files to volume via Files API
# ---------------------------------------------------------------------------
print("\n[2] Uploading data files to volume...")
data_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")
for fname in ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]:
    fpath = os.path.join(data_dir, fname)
    dest = f"{VOLUME_PATH}/{fname}"
    with open(fpath, "rb") as f:
        content = f.read()
    w.files.upload(file_path=dest, contents=io.BytesIO(content), overwrite=True)
    print(f"    Uploaded: {fname} ({len(content)} bytes)")

# ---------------------------------------------------------------------------
# 3. Create workspace directory and notebook
# ---------------------------------------------------------------------------
print("\n[3] Creating workspace directory and notebook...")
try:
    w.workspace.mkdirs(path=WORKSPACE_DIR)
    print(f"    Created dir: {WORKSPACE_DIR}")
except Exception as e:
    print(f"    Dir: {e}")

notebook_path = f"{WORKSPACE_DIR}/finetune_notebook"

notebook_source = f"""# Databricks notebook source
# COMMAND ----------
import os, shutil, json
import numpy as np

VOL_PATH = "{VOLUME_PATH}"
WORK_DIR = "/tmp/ft_work_{PREFIX}"
MODEL_FULL_NAME = "{MODEL_FULL_NAME}"
EXPERIMENT_PATH = "{EXPERIMENT_PATH}"

os.makedirs(WORK_DIR, exist_ok=True)

# Copy input files from volume to local work dir
for fname in ["base_model.npz", "finetune.txt", "eval.txt", "finetune_model.py"]:
    src = f"{{VOL_PATH}}/{{fname}}"
    dst = f"{{WORK_DIR}}/{{fname}}"
    shutil.copy(src, dst)
    print(f"Copied {{fname}}")

# Change to working directory so the script reads/writes from .
os.chdir(WORK_DIR)
print(f"Working dir: {{os.getcwd()}}")

# COMMAND ----------

# Execute fine-tuning script (reads from cwd, writes finetuned_model.npz and metrics.json)
exec(open(f"{{WORK_DIR}}/finetune_model.py").read())

# COMMAND ----------

# Copy results back to volume
for fname in ["finetuned_model.npz", "metrics.json"]:
    src = f"{{WORK_DIR}}/{{fname}}"
    dst = f"{{VOL_PATH}}/{{fname}}"
    shutil.copy(src, dst)
    print(f"Saved {{fname}} to volume")

metrics = json.load(open(f"{{WORK_DIR}}/metrics.json"))
print(f"Metrics: {{metrics}}")

# COMMAND ----------

# Log to MLflow and register model in UC model registry
import mlflow
import mlflow.pyfunc
mlflow.set_registry_uri("databricks-uc")

# Create or get experiment in our workspace dir
try:
    exp = mlflow.get_experiment_by_name(EXPERIMENT_PATH)
    if exp is None:
        mlflow.create_experiment(EXPERIMENT_PATH)
    mlflow.set_experiment(EXPERIMENT_PATH)
    print(f"Using experiment: {{EXPERIMENT_PATH}}")
except Exception as ex:
    print(f"Experiment setup note: {{ex}}")

# Define a pyfunc wrapper so MLflow can create a valid model directory
class BigramModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        import numpy as np
        data = np.load(context.artifacts["model_file"])
        self.logits = data["logits"].astype(float)
        self.vocab = [str(v) for v in data["vocab"]]
    def predict(self, context, model_input):
        return self.logits

# Build a signature for the model (required for UC model registry)
from mlflow.types.schema import Schema, ColSpec, TensorSpec
from mlflow.models.signature import ModelSignature
import numpy as np
sig = ModelSignature(
    inputs=Schema([ColSpec(name="input_char", type="string")]),
    outputs=Schema([TensorSpec(np.dtype("float64"), (-1,), "logits")]),
)

# Run and log model as a proper MLflow pyfunc model
with mlflow.start_run(run_name="ftjob0b3133") as run:
    mlflow.log_metric("eval_loss", metrics["eval_loss"])
    mlflow.log_metric("base_eval_loss", metrics["base_eval_loss"])
    mlflow.log_param("job_name", "ftjob0b3133")
    mlflow.pyfunc.log_model(
        artifact_path="model",
        python_model=BigramModel(),
        artifacts={{"model_file": f"{{WORK_DIR}}/finetuned_model.npz"}},
        signature=sig,
    )
    run_id = run.info.run_id
    print(f"MLflow run_id: {{run_id}}")

# Create registered model in UC (ignore if already exists)
from databricks.sdk import WorkspaceClient
w2 = WorkspaceClient()
try:
    w2.registered_models.create(
        name=MODEL_FULL_NAME,
        comment=f"Fine-tuned bigram LM. eval_loss={{metrics['eval_loss']}}, base_eval_loss={{metrics['base_eval_loss']}}",
    )
    print(f"Created registered model: {{MODEL_FULL_NAME}}")
except Exception as ex:
    print(f"Registered model note: {{ex}}")

# Register model version
mv = mlflow.register_model(
    model_uri=f"runs:/{{run_id}}/model",
    name=MODEL_FULL_NAME,
    await_registration_for=300,
)
print(f"Model version registered: {{mv.version}}")

# Add metrics as tags on the model version
client = mlflow.tracking.MlflowClient()
try:
    client.set_model_version_tag(MODEL_FULL_NAME, str(mv.version), "eval_loss", str(metrics["eval_loss"]))
    client.set_model_version_tag(MODEL_FULL_NAME, str(mv.version), "base_eval_loss", str(metrics["base_eval_loss"]))
    client.set_model_version_tag(MODEL_FULL_NAME, str(mv.version), "job_name", "ftjob0b3133")
    print("Tags added to model version")
except Exception as ex:
    print(f"Tags note: {{ex}}")

# Save results to volume for pickup
result = {{
    "run_id": run_id,
    "model_version": str(mv.version),
    "eval_loss": metrics["eval_loss"],
    "base_eval_loss": metrics["base_eval_loss"],
}}
with open(f"{{VOL_PATH}}/job_result.json", "w") as f:
    json.dump(result, f, indent=2)
print(f"Saved job_result.json: {{result}}")
print("ALL DONE")
"""

import base64
encoded = base64.b64encode(notebook_source.encode()).decode()
w.workspace.import_(
    path=notebook_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=encoded,
    overwrite=True,
)
print(f"    Notebook created: {notebook_path}")

# ---------------------------------------------------------------------------
# 4. Configure serverless compute environment
# ---------------------------------------------------------------------------
print("\n[4] Configuring serverless job environment...")

# Workspace only supports serverless compute - use environment-based approach
job_environment = jobs_svc.JobEnvironment(
    environment_key="default",
    spec=compute_svc.Environment(
        environment_version="2",
    ),
)

job_task = jobs_svc.Task(
    task_key="finetune",
    notebook_task=jobs_svc.NotebookTask(
        notebook_path=notebook_path,
    ),
    environment_key="default",
    # No new_cluster/existing_cluster_id - uses serverless compute
)
print("    Configured serverless compute (environment_version=2)")

# ---------------------------------------------------------------------------
# 5. Create and run the job
# ---------------------------------------------------------------------------
print("\n[5] Creating Databricks job...")

job = w.jobs.create(
    name=JOB_DISPLAY_NAME,
    tasks=[job_task],
    environments=[job_environment],
)
job_id = job.job_id
print(f"    Created job: {JOB_DISPLAY_NAME} (id={job_id})")

print("\n[6] Running the job...")
run = w.jobs.run_now(job_id=job_id)
run_id_job = run.run_id
print(f"    Run started: run_id={run_id_job}")

print("    Waiting for job to complete (checking every 30s)...")
max_wait = 1800
start_time = time.time()
while time.time() - start_time < max_wait:
    run_status = w.jobs.get_run(run_id=run_id_job)
    life_cycle = run_status.state.life_cycle_state
    result_state = run_status.state.result_state
    elapsed = int(time.time() - start_time)
    print(f"    [{elapsed}s] {life_cycle} / {result_state}")

    if str(life_cycle) in ["RunLifeCycleState.TERMINATED", "TERMINATED"]:
        if str(result_state) in ["ResultState.SUCCESS", "SUCCESS"]:
            print("    Job SUCCEEDED!")
        else:
            # Try to get error message
            state_msg = run_status.state.state_message or ""
            print(f"    Job FAILED: {result_state} - {state_msg}")
            # Get task-level info
            if run_status.tasks:
                for t in run_status.tasks:
                    print(f"    Task '{t.task_key}': {t.state}")
            raise RuntimeError(f"Job failed with state: {result_state}")
        break
    time.sleep(30)
else:
    raise TimeoutError("Job did not complete within 30 minutes")

# ---------------------------------------------------------------------------
# 7. Read results from volume
# ---------------------------------------------------------------------------
print("\n[7] Reading job results from volume...")
result_content = w.files.download(file_path=f"{VOLUME_PATH}/job_result.json").contents.read()
result = json.loads(result_content)
print(f"    Result: {result}")

metrics = {
    "eval_loss": result["eval_loss"],
    "base_eval_loss": result["base_eval_loss"],
}

# ---------------------------------------------------------------------------
# 8. Write submission/answers.json
# ---------------------------------------------------------------------------
print("\n[8] Writing submission/answers.json...")
os.makedirs("submission", exist_ok=True)
answers = {
    "job_name": "ftjob0b3133",
    "model_name": "ftmodel0b3133",
    "eval_loss": metrics["eval_loss"],
    "base_eval_loss": metrics["base_eval_loss"],
}
with open("submission/answers.json", "w") as f:
    json.dump(answers, f, indent=2)
print(f"    Written: {answers}")
print("\nDone!")
