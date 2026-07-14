"""Recreate predictions table with NOT NULL date and primary key, then publish."""
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
import json
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, DateType, DoubleType
from pyspark.sql.functions import col, to_date

spark = SparkSession.builder.getOrCreate()

CATALOG = "{CATALOG}"
DB = "{DB}"
PRED_NAME = "airqpredf4aae3"

results = {{}}

# 1. Read existing predictions
try:
    df_existing = spark.table(f"{{CATALOG}}.{{DB}}.{{PRED_NAME}}")
    count = df_existing.count()
    results["existing_count"] = count
    # Collect data
    rows = [(str(r["date"]), float(r["pm25_pred"])) for r in df_existing.collect()]
    results["sample"] = str(rows[:3])
except Exception as e:
    results["read_err"] = str(e)
    rows = []

# 2. Drop existing table
try:
    spark.sql(f"DROP TABLE IF EXISTS {{CATALOG}}.{{DB}}.{{PRED_NAME}}")
    results["drop"] = "OK"
except Exception as e:
    results["drop_err"] = str(e)

# 3. Recreate with NOT NULL date and primary key
try:
    spark.sql(f"""
    CREATE TABLE {{CATALOG}}.{{DB}}.{{PRED_NAME}} (
        date DATE NOT NULL,
        pm25_pred DOUBLE
    )
    USING DELTA
    TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')
    """)
    results["create"] = "OK"
except Exception as e:
    results["create_err"] = str(e)

# 4. Add primary key constraint
try:
    spark.sql(f"ALTER TABLE {{CATALOG}}.{{DB}}.{{PRED_NAME}} ADD CONSTRAINT pred_pk PRIMARY KEY (date) NOT ENFORCED")
    results["pk"] = "ADDED"
except Exception as e:
    results["pk_err"] = str(e)

# 5. Insert data back
if rows:
    try:
        from pyspark.sql import Row
        from datetime import date as date_type

        # Use SQL INSERT
        vals = ", ".join([f"(DATE '{{r[0]}}', {{r[1]}})" for r in rows])
        spark.sql(f"INSERT INTO {{CATALOG}}.{{DB}}.{{PRED_NAME}} VALUES {{vals}}")
        final_count = spark.sql(f"SELECT COUNT(*) FROM {{CATALOG}}.{{DB}}.{{PRED_NAME}}").first()[0]
        results["inserted"] = final_count
    except Exception as e:
        results["insert_err"] = str(e)

# 6. Verify PK constraint
try:
    r = spark.sql(f"""
        SELECT tc.constraint_name, tc.constraint_type, ccu.column_name
        FROM {{CATALOG}}.information_schema.table_constraints tc
        JOIN {{CATALOG}}.information_schema.constraint_column_usage ccu
          USING (constraint_catalog, constraint_schema, constraint_name)
        WHERE tc.table_schema = '{{DB}}' AND tc.table_name = '{{PRED_NAME}}'
    """).collect()
    results["constraints"] = str(r)
except Exception as e:
    results["constraints_err"] = str(e)

# 7. Check row count and sample
try:
    count2 = spark.sql(f"SELECT COUNT(*) FROM {{CATALOG}}.{{DB}}.{{PRED_NAME}}").first()[0]
    sample = spark.sql(f"SELECT * FROM {{CATALOG}}.{{DB}}.{{PRED_NAME}} ORDER BY date LIMIT 3").collect()
    results["final_count"] = count2
    results["sample_rows"] = str(sample)
except Exception as e:
    results["final_err"] = str(e)

dbutils.notebook.exit(json.dumps(results))
'''

nb_path = f"{NOTEBOOK_DIR}/recreate_pred"
w.workspace.import_(
    path=nb_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=base64.b64encode(nb.encode()).decode(),
    overwrite=True,
)

run = w.jobs.submit(
    run_name=f"{PREFIX}_recreate_pred",
    tasks=[SubmitTask(task_key="recreate", notebook_task=NotebookTask(notebook_path=nb_path, source=Source.WORKSPACE))]
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
                import json
                try:
                    data = json.loads(out.notebook_output.result)
                    for k, v in data.items():
                        print(f"  {k}: {v}")
                except:
                    print(out.notebook_output.result)
            if out and out.error:
                print("ERROR:", out.error[:300])
        print(f"Final: {lc}/{rs}")
        break
    print(f"  {lc}/{rs}")
    time.sleep(10)

# After notebook, try publish_table locally
print()
print("=== Trying publish_table after PK fix ===")
from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode

PRED_NAME = "airqpredf4aae3"
ONLINE_STORE = f"{PREFIX}-pred-store"
source_table = f"{CATALOG}.{DB}.{PRED_NAME}"
# Try with a new online table name that doesn't conflict
for online_name in [
    f"{CATALOG}.{DB}.{PRED_NAME}_ot",
    f"{CATALOG}.{DB}.{PRED_NAME}_pub",
    f"{CATALOG}.{DB}.{PRED_NAME}pub",
]:
    try:
        result = w.feature_store.publish_table(
            source_table_name=source_table,
            publish_spec=PublishSpec(
                online_store=ONLINE_STORE,
                online_table_name=online_name,
                publish_mode=PublishSpecPublishMode.TRIGGERED,
            ),
        )
        print(f"SUCCESS publish online_table_name={online_name}: {result}")
        break
    except Exception as e:
        print(f"FAIL {online_name}: {e}")
