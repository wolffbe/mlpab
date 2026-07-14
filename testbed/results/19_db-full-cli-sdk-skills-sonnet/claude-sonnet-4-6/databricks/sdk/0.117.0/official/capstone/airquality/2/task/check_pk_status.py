"""Check PK status on predictions table and synced table status."""
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

# Check synced table status locally
PRED_NAME = "airqpredf4aae3"
SYNCED_NAME = f"{CATALOG}.{DB}.{PRED_NAME}ot"
print(f"Checking synced table: {SYNCED_NAME}")
try:
    st = w.database.get_synced_database_table(name=SYNCED_NAME)
    print(f"  UC state: {st.unity_catalog_provisioning_state}")
    print(f"  DB instance: {st.effective_database_instance_name}")
    print(f"  Logical DB: {st.effective_logical_database_name}")
    print(f"  Sync: {st.data_synchronization_status}")
except Exception as e:
    print(f"  Error: {e}")

# Check UC table constraints
print()
print("Checking UC table constraints:")
try:
    tbl = w.tables.get(full_name=f"{CATALOG}.{DB}.{PRED_NAME}")
    print(f"  Table: {tbl.name} type={tbl.table_type}")
    if tbl.columns:
        for col in tbl.columns:
            print(f"  Col: {col.name} nullable={col.nullable}")
    if hasattr(tbl, "table_constraints"):
        print(f"  Constraints: {tbl.table_constraints}")
except Exception as e:
    print(f"  Error: {e}")

print()
print("Checking table_constraints API:")
try:
    constraints = list(w.table_constraints.list(table_full_name=f"{CATALOG}.{DB}.{PRED_NAME}"))
    print(f"  Constraints: {constraints}")
except Exception as e:
    print(f"  Error: {e}")

# Run notebook to check and set PK properly
print()
print("=== Running notebook to fix PK and verify ===")

nb = f'''# Databricks notebook source
from pyspark.sql import SparkSession
import json

spark = SparkSession.builder.getOrCreate()

CATALOG = "{CATALOG}"
DB = "{DB}"
PRED_NAME = "airqpredf4aae3"

results = {{}}

# Check existing constraints via SQL
try:
    r = spark.sql(f"""
        SELECT constraint_name, constraint_type, column_name
        FROM {{CATALOG}}.information_schema.table_constraints tc
        JOIN {{CATALOG}}.information_schema.constraint_column_usage ccu
          ON tc.constraint_name = ccu.constraint_name
        WHERE tc.table_schema = '{{DB}}' AND tc.table_name = '{{PRED_NAME}}'
    """).collect()
    results["constraints"] = str(r)
except Exception as e:
    results["constraints_err"] = str(e)

# Check via DESCRIBE EXTENDED
try:
    desc = spark.sql(f"DESCRIBE EXTENDED {{CATALOG}}.{{DB}}.{{PRED_NAME}}").collect()
    pk_rows = [str(r) for r in desc if "primary" in str(r).lower() or "constraint" in str(r).lower()]
    results["pk_rows"] = str(pk_rows[:5])
except Exception as e:
    results["desc_err"] = str(e)

# Try adding PK directly (might fail if exists)
try:
    spark.sql(f"ALTER TABLE {{CATALOG}}.{{DB}}.{{PRED_NAME}} DROP CONSTRAINT IF EXISTS pred_pk")
    spark.sql(f"ALTER TABLE {{CATALOG}}.{{DB}}.{{PRED_NAME}} ADD CONSTRAINT pred_pk PRIMARY KEY (date) NOT ENFORCED")
    results["pk_add"] = "SUCCESS"
except Exception as e:
    results["pk_add_err"] = str(e)

# Verify again
try:
    r2 = spark.sql(f"""
        SELECT constraint_name, constraint_type, column_name
        FROM {{CATALOG}}.information_schema.table_constraints tc
        JOIN {{CATALOG}}.information_schema.constraint_column_usage ccu
          ON tc.constraint_name = ccu.constraint_name
        WHERE tc.table_schema = '{{DB}}' AND tc.table_name = '{{PRED_NAME}}'
    """).collect()
    results["constraints_after"] = str(r2)
except Exception as e:
    results["constraints_after_err"] = str(e)

# Row count
count = spark.sql(f"SELECT COUNT(*) FROM {{CATALOG}}.{{DB}}.{{PRED_NAME}}").first()[0]
results["count"] = count

dbutils.notebook.exit(json.dumps(results))
'''

nb_path = f"{NOTEBOOK_DIR}/check_pk"
w.workspace.import_(
    path=nb_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=base64.b64encode(nb.encode()).decode(),
    overwrite=True,
)

run = w.jobs.submit(
    run_name=f"{PREFIX}_check_pk",
    tasks=[SubmitTask(task_key="check", notebook_task=NotebookTask(notebook_path=nb_path, source=Source.WORKSPACE))]
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
                print("NOTEBOOK OUTPUT:")
                import json
                try:
                    data = json.loads(out.notebook_output.result)
                    for k, v in data.items():
                        print(f"  {k}: {v}")
                except:
                    print(out.notebook_output.result)
            if out and out.error:
                print("ERROR:", out.error[:500])
        print(f"Final: {lc}/{rs}")
        break
    print(f"  {lc}/{rs}")
    time.sleep(10)
