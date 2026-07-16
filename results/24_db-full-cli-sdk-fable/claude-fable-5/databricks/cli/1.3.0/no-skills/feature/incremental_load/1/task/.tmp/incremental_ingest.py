# Databricks notebook source
# Daily incremental ingestion: merge any new increment files from the raw
# volume into the feature table (idempotent on row_id + event_time).
spark.sql("""
MERGE INTO workspace.mlpab679845.incrementalf3c1bf t
USING (
  SELECT row_id,
         account_id,
         CAST(event_time AS BIGINT) AS event_time,
         CAST(amount AS DOUBLE) AS amount,
         category
  FROM read_files(
    '/Volumes/workspace/mlpab679845/raw_increments/*.csv',
    format => 'csv',
    header => true
  )
) s
ON t.row_id = s.row_id AND t.event_time = s.event_time
WHEN NOT MATCHED THEN INSERT *
""")

# COMMAND ----------
# Refresh the online synced table so low-latency lookups see the new rows.
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
tbl = w.database.get_synced_database_table(
    name="workspace.mlpab679845.incrementalf3c1bf_online"
)
pipeline_id = tbl.data_synchronization_status.pipeline_id
try:
    w.pipelines.start_update(pipeline_id=pipeline_id)
except Exception as e:
    print(f"Sync refresh not started (may already be running): {e}")
