#!/usr/bin/env python3
"""
Creates the scored50223c feature table on Databricks and enables online access.
Uses serverless compute via notebook task.
"""
import os
import io
import base64
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs
from databricks.sdk.service.workspace import ImportFormat, Language
from databricks.sdk.service.catalog import (
    VolumeType, OnlineTable, OnlineTableSpec,
    OnlineTableSpecTriggeredSchedulingPolicy
)

w = WorkspaceClient()

schema_full = os.environ['MLPAB_DATABRICKS_SCHEMA']   # workspace.mlpabdaecb1
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']         # mlpabdaecb1

catalog_name, schema_name = schema_full.split('.')
volume_name = "task_data"
table_name = "scored50223c"
full_table_name = f"{catalog_name}.{schema_name}.{table_name}"
volume_path = f"/Volumes/{catalog_name}/{schema_name}/{volume_name}"

current_user = w.current_user.me()
user_name = current_user.user_name
notebook_dir = f"/Users/{user_name}/{prefix}"
notebook_path = f"{notebook_dir}/create_feature_table"

print(f"User: {user_name}")
print(f"Schema: {schema_full}")
print(f"Table: {full_table_name}")
print(f"Notebook: {notebook_path}")

# ── 1. Create volume ──────────────────────────────────────────────────────────
try:
    w.volumes.create(
        catalog_name=catalog_name,
        schema_name=schema_name,
        name=volume_name,
        volume_type=VolumeType.MANAGED
    )
    print(f"Created volume: {volume_name}")
except Exception as e:
    print(f"Volume: {e}")

# ── 2. Upload data files ──────────────────────────────────────────────────────
with open("data/requests.csv", "rb") as f:
    w.files.upload(f"{volume_path}/requests.csv", f, overwrite=True)
print("Uploaded requests.csv")

with open("data/profiles.csv", "rb") as f:
    w.files.upload(f"{volume_path}/profiles.csv", f, overwrite=True)
print("Uploaded profiles.csv")

# ── 3. Create notebook in workspace ──────────────────────────────────────────
notebook_source = f"""# Databricks notebook source
# Create scored50223c feature table

import math
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, udf
from pyspark.sql.types import DoubleType

catalog_name = "{catalog_name}"
schema_name = "{schema_name}"
table_name = "{table_name}"
volume_path = "{volume_path}"
full_table_name = f"{{catalog_name}}.{{schema_name}}.{{table_name}}"

# Read CSV files from volume
requests = spark.read.csv(f"{{volume_path}}/requests.csv", header=True, inferSchema=True)
profiles = spark.read.csv(f"{{volume_path}}/profiles.csv", header=True, inferSchema=True)

print(f"Requests: {{requests.count()}} rows")
print(f"Profiles: {{profiles.count()}} rows")

# Join on account_id
joined = requests.join(profiles, on="account_id", how="inner")

# On-demand transformation UDFs
def compute_distance(req_lat, req_lon, home_lat, home_lon):
    dist = math.sqrt((req_lat - home_lat)**2 + (req_lon - home_lon)**2)
    return round(dist, 6)

def compute_score(base_score, distance_deg):
    return round(base_score - 0.1 * distance_deg, 6)

distance_udf = udf(compute_distance, DoubleType())
score_udf = udf(compute_score, DoubleType())

result = joined.withColumn("distance_deg", distance_udf(
    col("request_lat"), col("request_lon"), col("home_lat"), col("home_lon")
)).withColumn("score", score_udf(
    col("base_score"), col("distance_deg")
)).select("request_id", "account_id", "distance_deg", "score")

print(f"Result count: {{result.count()}}")
result.show(5)

# Create feature table using Feature Engineering client
try:
    from databricks.feature_engineering import FeatureEngineeringClient
    fe = FeatureEngineeringClient()
    try:
        spark.sql(f"DROP TABLE IF EXISTS {{full_table_name}}")
        print("Dropped existing table")
    except Exception as drop_err:
        print(f"Drop attempt: {{drop_err}}")
    fe.create_table(
        name=full_table_name,
        primary_keys=["request_id"],
        df=result,
        description="Scored feature table with on-demand distance and score"
    )
    print(f"Feature table created via FeatureEngineeringClient: {{full_table_name}}")
except Exception as e:
    print(f"FeatureEngineeringClient error: {{e}}")
    # Fallback: write as Delta table with primary key constraint
    try:
        spark.sql(f"DROP TABLE IF EXISTS {{full_table_name}}")
    except Exception:
        pass
    result.write.format("delta").mode("overwrite").saveAsTable(full_table_name)
    try:
        spark.sql(f"ALTER TABLE {{full_table_name}} ADD CONSTRAINT pk_request_id PRIMARY KEY (request_id)")
        print("Added primary key constraint")
    except Exception as pk_err:
        print(f"Primary key constraint error: {{pk_err}}")
    print(f"Delta table created: {{full_table_name}}")

count = spark.table(full_table_name).count()
print(f"Final row count in {{full_table_name}}: {{count}}")
spark.table(full_table_name).show(5)
print("NOTEBOOK COMPLETE")
"""

# Create workspace directory and import notebook
try:
    w.workspace.mkdirs(path=notebook_dir)
    print(f"Created notebook directory: {notebook_dir}")
except Exception as e:
    print(f"Notebook dir: {e}")

content_b64 = base64.b64encode(notebook_source.encode("utf-8")).decode("utf-8")
w.workspace.import_(
    path=notebook_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=content_b64,
    overwrite=True
)
print(f"Uploaded notebook: {notebook_path}")

# ── 4. Submit job with notebook task (serverless) ─────────────────────────────
run = w.jobs.submit(
    run_name=f"{prefix}_create_scored50223c",
    tasks=[
        jobs.SubmitTask(
            task_key="create_feature_table",
            notebook_task=jobs.NotebookTask(
                notebook_path=notebook_path,
                source=jobs.Source.WORKSPACE
            )
            # No new_cluster or existing_cluster_id = serverless
        )
    ]
)

print(f"Submitted job run ID: {run.run_id}")

# ── 5. Wait for completion ────────────────────────────────────────────────────
start = time.time()
while True:
    run_state = w.jobs.get_run(run_id=run.run_id)
    state = run_state.state
    lc = state.life_cycle_state.value if state.life_cycle_state else "UNKNOWN"
    rs = state.result_state.value if state.result_state else ""
    elapsed = int(time.time() - start)
    print(f"[{elapsed}s] State: {lc} {rs}")
    if lc in ["TERMINATED", "SKIPPED", "INTERNAL_ERROR"]:
        break
    time.sleep(15)

# Get task-level output
run_details = w.jobs.get_run(run_id=run.run_id)
for task in (run_details.tasks or []):
    print(f"Task {task.task_key}: {task.state}")
    if task.run_id:
        try:
            out = w.jobs.get_run_output(run_id=task.run_id)
            if out.error:
                print(f"  Error: {out.error}")
            if out.notebook_output:
                print(f"  Output: {out.notebook_output.result}")
        except Exception as oe:
            print(f"  Output error: {oe}")

if state.result_state and state.result_state.value == "SUCCESS":
    print("\nJob succeeded!")
else:
    print(f"\nJob did not succeed: {state.state_message}")

# ── 6. Create online table ────────────────────────────────────────────────────
online_table_name = f"{catalog_name}.{schema_name}.{table_name}_online"
print(f"\nCreating online table: {online_table_name}")

try:
    w.online_tables.create(
        table=OnlineTable(
            name=online_table_name,
            spec=OnlineTableSpec(
                source_table_full_name=full_table_name,
                primary_key_columns=["request_id"],
                run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
                perform_full_copy=True
            )
        )
    )
    print(f"Online table creation initiated: {online_table_name}")

    # Wait for online table to provision
    ot_start = time.time()
    while True:
        ot = w.online_tables.get(name=online_table_name)
        prov_state = ot.unity_catalog_provisioning_state
        ot_elapsed = int(time.time() - ot_start)
        prov_val = prov_state.value if prov_state else "UNKNOWN"
        print(f"[{ot_elapsed}s] Online table provisioning: {prov_val}")
        if ot.status:
            print(f"  Detailed status: {ot.status}")
        if prov_val in ["ACTIVE", "FAILED"]:
            break
        if ot_elapsed > 600:
            print("Timed out waiting for online table")
            break
        time.sleep(20)

except Exception as e:
    print(f"Online table error: {e}")

print("\nImplementation complete.")
print(f"Feature table: {full_table_name}")
print(f"Online table: {online_table_name}")
