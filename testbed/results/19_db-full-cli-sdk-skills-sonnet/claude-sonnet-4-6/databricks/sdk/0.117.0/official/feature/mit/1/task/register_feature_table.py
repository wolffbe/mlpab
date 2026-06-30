"""
Register featuresb1ea93 as a Feature Engineering table by running
on an ML cluster notebook.
"""
import io, os, time, base64

from databricks.sdk import WorkspaceClient
from databricks.sdk.service import workspace as ws_svc
from databricks.sdk.service.jobs import SubmitTask, NotebookTask as NBTask
from databricks.sdk.service.compute import ClusterSpec, DataSecurityMode, RuntimeEngine

w = WorkspaceClient()
schema = os.environ['MLPAB_DATABRICKS_SCHEMA']
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
user = w.current_user.me().user_name

catalog, schema_name = schema.split('.')
notebook_dir = f"/Users/{user}/{prefix}"

print(f"Schema: {schema}")
print(f"User: {user}")

# Create registration notebook
notebook_path = f"{notebook_dir}/register_featuresb1ea93_fe"

notebook_content = f'''# Databricks notebook source
# MAGIC %md
# MAGIC # Register featuresb1ea93 as Feature Engineering table

# COMMAND ----------

from databricks.feature_engineering import FeatureEngineeringClient

catalog = "{catalog}"
schema_name = "{schema_name}"
table_name = f"{{catalog}}.{{schema_name}}.featuresb1ea93"

fe = FeatureEngineeringClient()

# Read the existing Delta table
df = spark.table(table_name)
print(f"Table has {{df.count()}} rows")
df.show(3)

# COMMAND ----------

# Drop and re-create as Feature Engineering table
# First check if it's already registered
try:
    ft = fe.get_table(name=table_name)
    print(f"Feature table already registered: {{ft}}")
except Exception as e:
    print(f"Not yet registered: {{e}}")

    # Since the Delta table already exists, we need to either:
    # 1. Drop it and re-create via fe.create_table
    # 2. Register the existing table
    # Try registering the existing table
    try:
        ft = fe.register_table(
            delta_table=table_name,
            primary_keys=["row_id"],
            timestamp_keys=["event_time"],
        )
        print(f"Table registered: {{ft}}")
    except Exception as e2:
        print(f"register_table error: {{e2}}")

        # If register_table doesn't work, drop and re-create
        print("Dropping and re-creating as feature table...")
        # We saved the data in the table already, so re-read it
        spark.sql(f"DROP TABLE IF EXISTS {{table_name}}")

        fe.create_table(
            name=table_name,
            primary_keys=["row_id"],
            timestamp_keys=["event_time"],
            df=df,
            description="Transaction features: amount_usd, is_weekend, amount_7d",
        )
        print("Feature table re-created!")

# COMMAND ----------

# Verify
ft = fe.get_table(name=table_name)
print(f"Feature table: {{ft}}")
print("SUCCESS!")
'''

content_b64 = base64.b64encode(notebook_content.encode()).decode()

try:
    w.workspace.mkdirs(notebook_dir)
except Exception:
    pass

w.workspace.import_(
    path=notebook_path,
    format=ws_svc.ExportFormat.SOURCE,
    language=ws_svc.Language.PYTHON,
    content=content_b64,
    overwrite=True,
)
print(f"Notebook created: {notebook_path}")

# Submit job on ML cluster
print("Submitting job on ML cluster...")
run_resp = w.jobs.submit(
    run_name=f"{prefix}_register_featuresb1ea93_fe",
    tasks=[
        SubmitTask(
            task_key="register_fe",
            notebook_task=NBTask(
                notebook_path=notebook_path,
            ),
            new_cluster=ClusterSpec(
                spark_version="15.4.x-cpu-ml-scala2.12",
                node_type_id="r4.xlarge",
                num_workers=0,
                spark_conf={
                    "spark.master": "local[*]",
                    "spark.databricks.cluster.profile": "singleNode",
                },
                custom_tags={"ResourceClass": "SingleNode"},
                data_security_mode=DataSecurityMode.SINGLE_USER,
            ),
        )
    ],
)

run_id = run_resp.run_id
print(f"Run submitted: run_id={run_id}")

print("Waiting for job completion...")
max_wait = 1800
start = time.time()
prev_state = None
while time.time() - start < max_wait:
    run_state = w.jobs.get_run(run_id=run_id)
    state = run_state.state
    life_cycle = state.life_cycle_state.value if state and state.life_cycle_state else "UNKNOWN"
    result = state.result_state.value if state and state.result_state else ""
    current = f"{life_cycle}/{result}"
    if current != prev_state:
        elapsed = int(time.time() - start)
        print(f"[{elapsed}s] {current}")
        prev_state = current
    if life_cycle in ["TERMINATED", "SKIPPED", "INTERNAL_ERROR"]:
        break
    time.sleep(15)

state = w.jobs.get_run(run_id=run_id).state
result = state.result_state.value if state and state.result_state else "UNKNOWN"
print(f"\nFinal result: {result}")

if result != "SUCCESS":
    for task_run in w.jobs.get_run(run_id=run_id).tasks or []:
        task_run_id = task_run.run_id
        if task_run_id:
            try:
                out = w.jobs.get_run_output(run_id=task_run_id)
                if out.error:
                    print(f"Error: {out.error}")
                if out.error_trace:
                    print(f"Trace: {out.error_trace[:2000]}")
            except Exception as e:
                print(f"Output error: {e}")
