"""
Air Quality PM2.5 FTI Pipeline - Orchestration using Databricks SDK
Notebook code (nb_src.py) runs on Databricks serverless.
"""
import os
import io
import time
import json
import base64

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import (
    SubmitTask, NotebookTask, Source,
    RunLifeCycleState, RunResultState,
    JobEnvironment
)
from databricks.sdk.service.compute import Environment
from databricks.sdk.service.workspace import ImportFormat, Language
from databricks.sdk.service.catalog import VolumeType

w = WorkspaceClient()
me = w.current_user.me()
user = me.user_name
print(f"Connected as: {user}")

schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]   # workspace.mlpab5e4845
prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]    # mlpab5e4845
catalog = schema.split(".")[0]                    # workspace
db_schema = schema.split(".")[1]                  # mlpab5e4845

print(f"Schema: {schema}, Prefix: {prefix}")

# ── 1. Upload CSVs to Unity Catalog Volume ─────────────────────────────────
vol_name = f"{prefix}_data"
print(f"Creating volume: {catalog}.{db_schema}.{vol_name}")
try:
    w.volumes.create(
        catalog_name=catalog,
        schema_name=db_schema,
        name=vol_name,
        volume_type=VolumeType.MANAGED
    )
    print("Volume created")
except Exception as e:
    print(f"Volume exists or error: {e}")

vol_path = f"/Volumes/{catalog}/{db_schema}/{vol_name}"

with open("data/airquality_history.csv", "rb") as f:
    w.files.upload(f"{vol_path}/airquality_history.csv", f, overwrite=True)
print("Uploaded history CSV")

with open("data/forecast_days.csv", "rb") as f:
    w.files.upload(f"{vol_path}/forecast_days.csv", f, overwrite=True)
print("Uploaded forecast CSV")

# ── 2. Upload Notebook ─────────────────────────────────────────────────────
notebook_dir = f"/Users/{user}/{prefix}"
try:
    w.workspace.mkdirs(notebook_dir)
except Exception as e:
    print(f"mkdirs: {e}")

notebook_path = f"{notebook_dir}/airquality_pipeline"

with open("nb_src.py", "rb") as f:
    nb_bytes = f.read()

w.workspace.import_(
    path=notebook_path,
    content=base64.b64encode(nb_bytes).decode(),
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    overwrite=True
)
print(f"Notebook uploaded: {notebook_path}")

# ── 3. Submit Job (serverless with mlflow + sklearn) ─────────────────────
run = w.jobs.submit(
    run_name=f"{prefix}_airquality_pipeline",
    environments=[
        JobEnvironment(
            environment_key="ml_env",
            spec=Environment(
                client="1",
                dependencies=["mlflow", "scikit-learn"]
            )
        )
    ],
    tasks=[
        SubmitTask(
            task_key="pipeline",
            environment_key="ml_env",
            notebook_task=NotebookTask(
                notebook_path=notebook_path,
                source=Source.WORKSPACE,
                base_parameters={
                    "catalog":   catalog,
                    "db_schema": db_schema,
                    "vol_path":  vol_path,
                    "user":      user,
                    "prefix":    prefix,
                }
            ),
        )
    ]
)

run_id = run.run_id
print(f"Job submitted, run_id={run_id}")

# ── 4. Wait for Completion ─────────────────────────────────────────────────
print("Waiting for job (up to 50 min)...")
for i in range(100):
    time.sleep(30)
    state = w.jobs.get_run(run_id=run_id)
    lc = state.state.life_cycle_state
    elapsed = (i + 1) * 30
    print(f"  [{elapsed}s] {lc}")
    if lc in (RunLifeCycleState.TERMINATED, RunLifeCycleState.SKIPPED,
              RunLifeCycleState.INTERNAL_ERROR):
        result = state.state.result_state
        print(f"  Result: {result}")
        if result != RunResultState.SUCCESS:
            for task in (state.tasks or []):
                print(f"  Task {task.task_key}: life={task.state.life_cycle_state} result={task.state.result_state}")
        else:
            # Try to get notebook output
            for task in (state.tasks or []):
                try:
                    out = w.jobs.get_run_output(run_id=task.run_id)
                    if out.notebook_output:
                        print(f"Notebook output: {out.notebook_output.result}")
                except Exception as e:
                    print(f"Could not get output: {e}")
        break

# ── 5. Create Online Table ─────────────────────────────────────────────────
pred_table_full = f"{schema}.airqpredfdfb59"
print(f"\nCreating online table for: {pred_table_full}")
try:
    from databricks.sdk.service.catalog import (
        OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
    )

    ot = w.online_tables.create(
        table=OnlineTable(
            name=f"{schema}.airqpredfdfb59_online",
            spec=OnlineTableSpec(
                source_table_full_name=pred_table_full,
                primary_key_columns=["date"],
                run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
            )
        )
    )
    print(f"Online table created: {ot.name}")
except Exception as e:
    print(f"Online table note: {e}")

print("\nAll done!")
