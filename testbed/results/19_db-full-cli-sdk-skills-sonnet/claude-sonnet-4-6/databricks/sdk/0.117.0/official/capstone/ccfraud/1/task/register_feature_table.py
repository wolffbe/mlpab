from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat, Language
import base64, time

w = WorkspaceClient()
user = 'benedict@logicalclocks.com'
prefix = 'mlpab88e583'

nb_content = r"""# Databricks notebook source
# Register ccpred739ee9 as a Feature Engineering table
SCHEMA = "workspace.mlpab88e583"
ONLINE_STORE = "mlpab88e583-store"

# COMMAND ----------
# Check what's available
import sys
print(f"Python: {sys.version}")
print(f"Spark version: {spark.version}")

# Check databricks.feature_engineering availability
try:
    import databricks.feature_engineering
    print(f"databricks.feature_engineering version: {databricks.feature_engineering.__version__}")
    from databricks.feature_engineering import FeatureEngineeringClient
    fe = FeatureEngineeringClient()
    print("FeatureEngineeringClient created")
except Exception as e:
    print(f"databricks.feature_engineering error: {e}")
    fe = None

# COMMAND ----------
if fe is not None:
    # Load predictions
    pred_df = spark.table(f"{SCHEMA}.ccpred739ee9")
    pred_df.printSchema()

    # Drop existing feature table if exists
    try:
        fe.drop_table(name=f"{SCHEMA}.ccpred739ee9")
        print("Dropped existing feature table")
    except Exception as e:
        print(f"Drop error (may not exist): {e}")

    # Recreate with proper NOT NULL constraint for transaction_id
    # Use the _v2 table which has the NOT NULL pk constraint
    pred_df_v2 = spark.table(f"{SCHEMA}.ccpred739ee9_v2")
    pred_df_v2.printSchema()

    # Create as feature table
    try:
        ft = fe.create_table(
            name=f"{SCHEMA}.ccpred739ee9",
            primary_keys=["transaction_id"],
            df=pred_df_v2,
            description="Credit card fraud prediction probabilities for low-latency serving"
        )
        print(f"Feature table created: {ft}")
    except Exception as e:
        print(f"Create feature table error: {e}")
        # Try to write to existing
        try:
            fe.write_table(
                name=f"{SCHEMA}.ccpred739ee9",
                df=pred_df_v2,
                mode="overwrite"
            )
            print("Wrote to existing feature table")
        except Exception as e2:
            print(f"Write error: {e2}")

    # Verify feature table
    try:
        ft_info = fe.get_table(name=f"{SCHEMA}.ccpred739ee9")
        print(f"Feature table info: {ft_info}")
    except Exception as e:
        print(f"Get table error: {e}")

    # Try to publish to online store
    print(f"Attempting to publish to online store: {ONLINE_STORE}...")
    try:
        # Try different publish approaches
        # Method 1: Using OnlineStoreSpec if available
        try:
            from databricks.feature_engineering.entities.feature_store_online_store_spec import FeatureStoreOnlineStoreSpec
            spec = FeatureStoreOnlineStoreSpec(
                online_store=ONLINE_STORE,
                online_table_name=f"{SCHEMA}.ccpred739ee9"
            )
            fe.publish_table(
                name=f"{SCHEMA}.ccpred739ee9",
                online_store=spec
            )
            print("Published to online store!")
        except ImportError:
            print("FeatureStoreOnlineStoreSpec not available")
            # Method 2: Direct online_store argument
            try:
                fe.publish_table(
                    name=f"{SCHEMA}.ccpred739ee9",
                    online_store=ONLINE_STORE
                )
                print("Published via direct online_store!")
            except Exception as e:
                print(f"Direct publish error: {e}")
    except Exception as e:
        print(f"Publish error: {e}")
else:
    print("FeatureEngineeringClient not available - using REST API approach")
    from databricks.sdk import WorkspaceClient
    from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode
    w2 = WorkspaceClient()

    # Try to register ccpred739ee9_v2 as online feature table via the feature_store API
    # The v2 table has a primary key constraint
    print("Trying feature_store.publish_table on v2...")
    for tbl in [f"{SCHEMA}.ccpred739ee9_v2", f"{SCHEMA}.ccpred739ee9"]:
        try:
            ps = PublishSpec(
                online_store=ONLINE_STORE,
                online_table_name=tbl,
                publish_mode=PublishSpecPublishMode.SNAPSHOT
            )
            result = w2.feature_store.publish_table(
                source_table_name=tbl,
                publish_spec=ps
            )
            print(f"Published {tbl}: {result}")
        except Exception as e:
            print(f"Publish {tbl} error: {e}")

# COMMAND ----------
# Final verification
pred_df_final = spark.table(f"{SCHEMA}.ccpred739ee9")
print(f"Final predictions count: {pred_df_final.count()}")
pred_df_final.show(5)
dbutils.notebook.exit("DONE")
"""

nb_path = f'/Users/{user}/{prefix}/ccfraud_register_ft'
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
    task_key='register_ft',
    notebook_task=NotebookTask(notebook_path=nb_path, source=Source.WORKSPACE),
    timeout_seconds=600
)
job = w.jobs.create(name=f'{prefix}_register_ft', tasks=[task])
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
                    print('Error:', out.error[:3000])
                if out.error_trace:
                    print('Trace:', out.error_trace[:2000])
        print(f'FINAL: {lc} {rs}')
        break
    time.sleep(15)
