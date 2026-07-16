from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat, Language
import base64, time

w = WorkspaceClient()
user = 'benedict@logicalclocks.com'
prefix = 'mlpab88e583'

nb_content = r"""# Databricks notebook source
# Enable online serving for predictions using FeatureEngineeringClient
SCHEMA = "workspace.mlpab88e583"
PREFIX = "mlpab88e583"
ONLINE_STORE = "mlpab88e583-store"

# COMMAND ----------
print("Checking available libraries...")
import subprocess
result = subprocess.run(["pip", "show", "databricks-feature-engineering"], capture_output=True, text=True)
print(result.stdout)
print(result.stderr)

try:
    from databricks.feature_engineering import FeatureEngineeringClient
    print("FeatureEngineeringClient available")
    fe = FeatureEngineeringClient()

    # Load predictions
    pred_df = spark.table(f"{SCHEMA}.ccpred739ee9")
    print(f"Predictions count: {pred_df.count()}")
    pred_df.printSchema()

    # First check if feature table exists
    try:
        ft_info = fe.get_table(name=f"{SCHEMA}.ccpred739ee9")
        print(f"Feature table already exists: {ft_info}")
    except Exception as e:
        print(f"Feature table doesn't exist yet: {e}")
        # Create it
        print("Creating feature table...")
        ft = fe.create_table(
            name=f"{SCHEMA}.ccpred739ee9",
            primary_keys=["transaction_id"],
            df=pred_df,
            description="Credit card fraud prediction probabilities"
        )
        print(f"Feature table created: {ft.name}")

    # Try to enable online serving
    print("Publishing to online store...")
    try:
        spec = AzureCosmosDBSpec(
            account_uri="placeholder",
            write_secret_prefix="placeholder"
        )
    except Exception:
        pass

    # Try publish using online store
    try:
        fe.publish_table(
            name=f"{SCHEMA}.ccpred739ee9",
            online_store=OnlineStoreSpec(
                online_store_name=ONLINE_STORE
            )
        )
        print("Published to online store!")
    except Exception as e:
        print(f"Publish error: {e}")

except ImportError as e:
    print(f"FeatureEngineeringClient not available: {e}")

    # Use alternative approach: check if Synced Tables API is available
    print("Checking Synced Tables API...")

    # Try using REST API directly via databricks.sdk
    from databricks.sdk import WorkspaceClient
    w2 = WorkspaceClient()

    # Try to create online table via REST
    print("Checking online table support...")
    try:
        import requests
        # Check what APIs are available
        print("No direct Synced Tables SDK API found")
    except:
        pass

    # At minimum, ensure the Delta table is CDF-enabled
    spark.sql(f"ALTER TABLE {SCHEMA}.ccpred739ee9 SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")
    print("CDF enabled on predictions table")

    # Verify table is ready for online serving
    result = spark.sql(f"DESCRIBE EXTENDED {SCHEMA}.ccpred739ee9")
    result.filter(result.col_name.isin(['Type', 'delta.enableChangeDataFeed', 'Table Properties'])).show(truncate=False)

# COMMAND ----------
# Final verification
pred_check = spark.table(f"{SCHEMA}.ccpred739ee9")
print(f"Predictions table: {pred_check.count()} rows")
pred_check.show(5)
dbutils.notebook.exit("DONE")
"""

nb_path = f'/Users/{user}/{prefix}/ccfraud_online_publish'
encoded = base64.b64encode(nb_content.encode()).decode()
w.workspace.import_(
    path=nb_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=encoded,
    overwrite=True
)
print(f'Notebook uploaded: {nb_path}')

from databricks.sdk.service.jobs import NotebookTask, Task, Source
task = Task(
    task_key='online_publish',
    notebook_task=NotebookTask(notebook_path=nb_path, source=Source.WORKSPACE),
    timeout_seconds=600
)
job = w.jobs.create(name=f'{prefix}_online_publish', tasks=[task])
run = w.jobs.run_now(job_id=job.job_id)
print(f'Job: {job.job_id}, Run: {run.run_id}')

start = time.time()
while True:
    r = w.jobs.get_run(run_id=run.run_id)
    lc = r.state.life_cycle_state.value if r.state.life_cycle_state else 'UNKNOWN'
    rs = r.state.result_state.value if r.state.result_state else ''
    elapsed = int(time.time() - start)
    print(f'[{elapsed}s] {lc} {rs}')
    if lc in ('TERMINATED', 'SKIPPED', 'INTERNAL_ERROR'):
        for t in (r.tasks or []):
            if t.run_id:
                out = w.jobs.get_run_output(run_id=t.run_id)
                if out.notebook_output:
                    print('Output:', out.notebook_output.result)
                if out.error:
                    print('Error:', out.error[:2000])
        print(f'FINAL: {lc} {rs}')
        break
    time.sleep(15)
