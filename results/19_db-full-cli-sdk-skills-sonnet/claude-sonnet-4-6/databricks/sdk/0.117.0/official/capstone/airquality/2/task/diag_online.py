"""Diagnose online table creation and feature store publish options."""
import os
import base64
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.jobs import SubmitTask, NotebookTask, Source
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
CATALOG = SCHEMA.split(".")[0]
DB = SCHEMA.split(".")[1]
me = w.current_user.me()
user = me.user_name
NOTEBOOK_DIR = f"/Users/{user}/{PREFIX}"

nb = f'''# Databricks notebook source
import os, requests, json

CATALOG = "{CATALOG}"
DB = "{DB}"
PRED_NAME = "airqpredf4aae3"
PREFIX = "{PREFIX}"

# Auth
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().get()
host = ctx.apiUrl().get()
headers = {{"Authorization": f"Bearer {{token}}", "Content-Type": "application/json"}}

results = []

# 1. Check PK on predictions table
from pyspark.sql import SparkSession
spark = SparkSession.builder.getOrCreate()
try:
    # Check for pk via DESCRIBE
    desc = spark.sql(f"DESCRIBE EXTENDED {{CATALOG}}.{{DB}}.{{PRED_NAME}}").collect()
    pk_rows = [str(r) for r in desc if "constraint" in str(r).lower() or "primary" in str(r).lower()]
    results.append(f"PK_ROWS: {{pk_rows[:3]}}")
except Exception as e:
    results.append(f"PK_CHECK_ERR: {{e}}")

# 2. Check INFORMATION_SCHEMA for constraints
try:
    constraints = spark.sql(f"""
        SELECT constraint_name, constraint_type, column_name
        FROM {{CATALOG}}.information_schema.constraint_column_usage
        WHERE table_schema = '{{DB}}' AND table_name = '{{PRED_NAME}}'
    """).collect()
    results.append(f"CONSTRAINTS: {{constraints}}")
except Exception as e:
    results.append(f"CONSTRAINTS_ERR: {{e}}")

# 3. Try POST online-tables API
ot_payload = {{
    "name": f"{{CATALOG}}.{{DB}}.{{PRED_NAME}}",
    "spec": {{
        "source_table_full_name": f"{{CATALOG}}.{{DB}}.{{PRED_NAME}}",
        "primary_key_columns": ["date"],
        "run_triggered": {{}}
    }}
}}
r_post = requests.post(
    f"{{host}}/api/2.0/online-tables/tables",
    headers=headers,
    json=ot_payload
)
results.append(f"OT_POST: {{r_post.status_code}} {{r_post.text[:400]}}")

# 4. GET online table
r_get = requests.get(
    f"{{host}}/api/2.0/online-tables/tables/{{CATALOG}}.{{DB}}.{{PRED_NAME}}",
    headers=headers
)
results.append(f"OT_GET: {{r_get.status_code}} {{r_get.text[:300]}}")

# 5. Try listing online tables
r_list = requests.get(
    f"{{host}}/api/2.0/online-tables/tables",
    headers=headers,
    params={{"catalog_name": CATALOG, "schema_name": DB}}
)
results.append(f"OT_LIST: {{r_list.status_code}} {{r_list.text[:300]}}")

# 6. Check feature_store methods
from databricks.sdk import WorkspaceClient as WC
wc = WC(host=host, token=token)
fs_methods = [m for m in dir(wc.feature_store) if not m.startswith("_")]
results.append(f"FS_METHODS: {{fs_methods}}")

# 7. Try feature_store publish_table with debug
try:
    import inspect
    sig = inspect.signature(wc.feature_store.publish_table)
    results.append(f"PUBLISH_SIG: {{sig}}")
except Exception as e:
    results.append(f"PUBLISH_SIG_ERR: {{e}}")

# 8. List existing online stores
try:
    stores = list(wc.feature_store.list_online_stores())
    results.append(f"STORES: {{[getattr(s, 'name', str(s)) for s in stores]}}")
except Exception as e:
    results.append(f"STORES_ERR: {{e}}")

# 9. Try feature-store REST API
r_fs = requests.get(
    f"{{host}}/api/2.0/feature-store/online-stores",
    headers=headers
)
results.append(f"FS_STORES: {{r_fs.status_code}} {{r_fs.text[:300]}}")

# 10. Try publish via REST
r_pub = requests.post(
    f"{{host}}/api/2.0/feature-store/feature-tables/publish",
    headers=headers,
    json={{
        "feature_table_name": f"{{CATALOG}}.{{DB}}.{{PRED_NAME}}",
        "online_store_spec": {{
            "online_store_name": f"{{PREFIX}}-pred-store"
        }}
    }}
)
results.append(f"FS_PUB: {{r_pub.status_code}} {{r_pub.text[:300]}}")

dbutils.notebook.exit("\\n".join(results))
'''

nb_path = f"{NOTEBOOK_DIR}/diag_online"
w.workspace.import_(
    path=nb_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=base64.b64encode(nb.encode()).decode(),
    overwrite=True,
)

run = w.jobs.submit(
    run_name=f"{PREFIX}_diag_online",
    tasks=[SubmitTask(task_key="diag", notebook_task=NotebookTask(notebook_path=nb_path, source=Source.WORKSPACE))]
)
run_id = run.run_id
print(f"Run: {run_id}")

while True:
    info = w.jobs.get_run(run_id=run_id)
    lc = info.state.life_cycle_state.value
    rs = info.state.result_state.value if info.state.result_state else ""
    if lc in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        for t in (info.tasks or []):
            out = w.jobs.get_run_output(run_id=t.run_id)
            if out and out.notebook_output and out.notebook_output.result:
                print("OUTPUT:")
                print(out.notebook_output.result)
            if out and out.error:
                print("ERROR:", out.error[:1000])
            if out and out.error_trace:
                print("TRACE:", out.error_trace[:800])
        print(f"Final: {lc}/{rs}")
        break
    print(f"  {lc}/{rs}")
    time.sleep(10)
