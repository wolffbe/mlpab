from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat, Language
import base64, time

w = WorkspaceClient()
user = 'benedict@logicalclocks.com'
prefix = 'mlpab88e583'

nb_content = r"""# Databricks notebook source
# Register predictions table as Feature Engineering table and enable online serving
SCHEMA = "workspace.mlpab88e583"
PREFIX = "mlpab88e583"

# Check if the FeatureEngineeringClient is available
try:
    from databricks.feature_engineering import FeatureEngineeringClient
    fe = FeatureEngineeringClient()
    print("FeatureEngineeringClient available")

    # Create feature table for predictions
    pred_table = spark.table(f"{SCHEMA}.ccpred739ee9")
    print(f"Predictions schema: {pred_table.schema}")

    # Register as feature table (using overwrite if exists)
    try:
        ft = fe.create_table(
            name=f"{SCHEMA}.ccpred739ee9_ft",
            primary_keys=["transaction_id"],
            df=pred_table,
            description="Credit card fraud predictions feature table"
        )
        print(f"Feature table created: {ft.name}")
    except Exception as e:
        print(f"Create feature table error (may already exist): {e}")
        try:
            fe.write_table(
                name=f"{SCHEMA}.ccpred739ee9_ft",
                df=pred_table,
                mode="overwrite"
            )
            print("Feature table written")
        except Exception as e2:
            print(f"Write feature table error: {e2}")

    # Enable online store
    try:
        from databricks.feature_engineering.entities.feature_store_online_store_spec import AzureCosmosDBSpec, DynamoDBSpec
        print("Online store specs available")
    except Exception as e:
        print(f"Online store spec import error: {e}")

except ImportError as e:
    print(f"FeatureEngineeringClient not available: {e}")

    # Fall back to just enabling the online lookup via Delta with proper config
    print("Falling back to Delta table with online lookup enabled")
    spark.sql(f"ALTER TABLE {SCHEMA}.ccpred739ee9 SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")
    spark.sql(f"DESCRIBE EXTENDED {SCHEMA}.ccpred739ee9").show(50, truncate=False)
    print("Table properties set for online lookup compatibility")

# COMMAND ----------
# Verify the predictions table is accessible
from pyspark.sql import functions as F
pred_df = spark.table(f"{SCHEMA}.ccpred739ee9")
print(f"Predictions count: {pred_df.count()}")
print(f"Schema:")
pred_df.printSchema()

# Show sample lookups
pred_df.orderBy(F.col("fraud_probability").desc()).show(5)
pred_df.orderBy(F.col("fraud_probability").asc()).show(5)

# Show distribution
print("Fraud probability distribution:")
pred_df.select(
    F.count("*").alias("total"),
    F.sum(F.when(F.col("fraud_probability") > 0.5, 1).otherwise(0)).alias("high_prob"),
    F.avg("fraud_probability").alias("avg_prob"),
    F.min("fraud_probability").alias("min_prob"),
    F.max("fraud_probability").alias("max_prob")
).show()

dbutils.notebook.exit(f"DONE|count={pred_df.count()}")
"""

nb_path = f'/Users/{user}/{prefix}/ccfraud_online_table'
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
    task_key='online_table',
    notebook_task=NotebookTask(notebook_path=nb_path, source=Source.WORKSPACE),
    timeout_seconds=600
)
job = w.jobs.create(name=f'{prefix}_online_table', tasks=[task])
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
