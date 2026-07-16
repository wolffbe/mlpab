"""Add primary key to predictions table and publish to online store."""
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
from pyspark.sql import SparkSession
import os, requests

spark = SparkSession.builder.getOrCreate()

CATALOG = "{CATALOG}"
DB = "{DB}"
PRED_NAME = "airqpredf4aae3"
ONLINE_NAME = f"{{CATALOG}}.{{DB}}.{{PRED_NAME}}"

# COMMAND ----------
# Get auth context
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().get()
host = ctx.apiUrl().get()
os.environ["DATABRICKS_HOST"] = host
os.environ["DATABRICKS_TOKEN"] = token
headers = {{"Authorization": f"Bearer {{token}}", "Content-Type": "application/json"}}
print(f"Auth: {{host[:40]}}")

# COMMAND ----------
# 1. Add primary key constraint to predictions table
try:
    spark.sql(f"ALTER TABLE {{CATALOG}}.{{DB}}.{{PRED_NAME}} ADD CONSTRAINT pred_pk PRIMARY KEY (date) NOT ENFORCED")
    print("Primary key added successfully")
except Exception as e:
    print(f"PK add (may already exist): {{e}}")

# Verify table structure
result = spark.sql(f"SHOW TBLPROPERTIES {{CATALOG}}.{{DB}}.{{PRED_NAME}}")
result.show(10, truncate=False)

count = spark.sql(f"SELECT COUNT(*) FROM {{CATALOG}}.{{DB}}.{{PRED_NAME}}").first()[0]
print(f"Predictions count: {{count}}")

# COMMAND ----------
# 2. Try creating online table via REST API (new UC Online Tables)
import json

online_table_payload = {{
    "name": f"{{CATALOG}}.{{DB}}.{{PRED_NAME}}",
    "spec": {{
        "source_table_full_name": f"{{CATALOG}}.{{DB}}.{{PRED_NAME}}",
        "primary_key_columns": ["date"],
        "run_triggered": {{}}
    }}
}}

print(f"Creating online table: {{online_table_payload['name']}}")

r = requests.post(
    f"{{host}}/api/2.0/online-tables/tables",
    headers=headers,
    json=online_table_payload
)
print(f"Online table create: {{r.status_code}}")
print(r.text[:600])

if r.status_code == 200:
    print("Online table created successfully!")
    ot_data = r.json()
    print(f"Status: {{ot_data.get('status', 'unknown')}}")
elif r.status_code == 409:
    print("Online table already exists")
    # Try to get it
    rg = requests.get(
        f"{{host}}/api/2.0/online-tables/tables/{{CATALOG}}.{{DB}}.{{PRED_NAME}}",
        headers=headers
    )
    print(f"Get online table: {{rg.status_code}} {{rg.text[:300]}}")
else:
    print(f"REST API failed: {{r.status_code}} {{r.text[:500]}}")

# COMMAND ----------
# 3. Also try feature_store publish_table with primary key
try:
    from databricks.sdk import WorkspaceClient as WC
    wc = WC(host=host, token=token)

    # Try publish_table
    try:
        result = wc.feature_store.publish_table(
            name=f"{{CATALOG}}.{{DB}}.{{PRED_NAME}}",
            online_store=wc.feature_store.get_online_store(name="{PREFIX}-pred-store")
        )
        print(f"Published via feature_store: {{result}}")
    except Exception as e2:
        print(f"feature_store.publish_table: {{e2}}")

    # Try listing online stores
    try:
        stores = list(wc.feature_store.list_online_stores())
        print(f"Online stores: {{[s.name for s in stores]}}")
    except Exception as e3:
        print(f"List online stores: {{e3}}")

except Exception as e:
    print(f"Feature store SDK: {{e}}")

# COMMAND ----------
# 4. Try Databricks Feature Engineering Client if available
try:
    from databricks.feature_engineering import FeatureEngineeringClient
    fe = FeatureEngineeringClient()
    print("Feature Engineering client available")

    # Register feature table
    try:
        ft = fe.create_table(
            name=f"{{CATALOG}}.{{DB}}.{{PRED_NAME}}",
            primary_keys=["date"],
            df=spark.table(f"{{CATALOG}}.{{DB}}.{{PRED_NAME}}"),
            description="Air quality PM2.5 predictions"
        )
        print(f"Feature table registered: {{ft}}")
    except Exception as e2:
        print(f"create_table: {{e2}}")

    # Try publish
    try:
        from databricks.feature_engineering import OnlineStoreSpec, AmazonDynamoDBSpec
        pass
    except ImportError:
        pass

except ImportError as e:
    print(f"No FeatureEngineeringClient: {{e}}")

# COMMAND ----------
# Final verification
print("\\n=== Final Status ===")
count = spark.sql(f"SELECT COUNT(*) FROM {{CATALOG}}.{{DB}}.{{PRED_NAME}}").first()[0]
print(f"Predictions table: {{count}} rows")

sample = spark.sql(f"SELECT * FROM {{CATALOG}}.{{DB}}.{{PRED_NAME}} LIMIT 3").collect()
for r in sample:
    print(f"  {{r}}")

# Check online table via REST
rg = requests.get(
    f"{{host}}/api/2.0/online-tables/tables/{{CATALOG}}.{{DB}}.{{PRED_NAME}}",
    headers=headers
)
print(f"Online table status: {{rg.status_code}} {{rg.text[:300]}}")

dbutils.notebook.exit(f"DONE pk_added count={{count}} online={{rg.status_code}}")
'''

nb_path = f"{NOTEBOOK_DIR}/fix_pk"
w.workspace.import_(
    path=nb_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=base64.b64encode(nb.encode()).decode(),
    overwrite=True,
)

run = w.jobs.submit(
    run_name=f"{PREFIX}_fix_pk",
    tasks=[SubmitTask(task_key="fix", notebook_task=NotebookTask(notebook_path=nb_path, source=Source.WORKSPACE))]
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
                print("OUTPUT:", out.notebook_output.result)
            if out and out.error:
                print("ERROR:", out.error[:1000])
            if out and out.error_trace:
                print("TRACE:", out.error_trace[:500])
        print(f"Final: {lc}/{rs}")
        break
    print(f"  {lc}/{rs}")
    time.sleep(10)
