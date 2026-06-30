# Databricks notebook source
# MAGIC %pip install databricks-sdk --upgrade -q
dbutils.library.restartPython()

# COMMAND ----------
import databricks.sdk.service.database as db_svc
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
results = []

# Create SyncedTableSpec with NewPipelineSpec to provide explicit storage
new_pipeline = db_svc.NewPipelineSpec(
    storage_catalog="workspace",
    storage_schema="mlpab0442b8"
)

spec = db_svc.SyncedTableSpec(
    source_table_full_name="workspace.mlpab0442b8.accountse81ff1",
    primary_key_columns=["row_id", "updated_at"],
    timeseries_key="updated_at",
    create_database_objects_if_missing=True,
    scheduling_policy=db_svc.SyncedTableSchedulingPolicy.TRIGGERED,
    new_pipeline_spec=new_pipeline
)

synced_table = db_svc.SyncedDatabaseTable(
    name="mlpab0442b8db.mlpab0442b8.accountse81ff1",
    spec=spec
)

results.append(f"Spec: {spec.as_dict()}")
results.append(f"SyncedDatabaseTable: {synced_table.as_dict()}")

try:
    result = w.database.create_synced_database_table(synced_table=synced_table)
    results.append(f"Success: {result}")
    results.append(f"State: {result.data_synchronization_status.detailed_state}")
    results.append(f"Pipeline: {result.data_synchronization_status.pipeline_id}")
except Exception as e:
    results.append(f"Error: {type(e).__name__}: {e}")

spark.createDataFrame([(r,) for r in results], ["result"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.synced_v3_output")
