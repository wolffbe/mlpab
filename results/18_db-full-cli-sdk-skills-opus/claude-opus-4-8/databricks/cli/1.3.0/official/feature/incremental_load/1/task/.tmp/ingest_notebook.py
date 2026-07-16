# Databricks notebook source
# Daily incremental ingestion for feature table workspace.mlpab77f9d2.incrementalb48074
# COPY INTO is idempotent: it only loads CSV files in the volume that have not
# already been ingested, so re-running daily picks up new increments only.

spark.sql("""
COPY INTO workspace.mlpab77f9d2.incrementalb48074
FROM (
  SELECT
    CAST(row_id AS STRING)     AS row_id,
    CAST(account_id AS STRING) AS account_id,
    CAST(event_time AS BIGINT) AS event_time,
    CAST(amount AS DOUBLE)     AS amount,
    CAST(category AS STRING)   AS category
  FROM '/Volumes/workspace/mlpab77f9d2/increments/'
)
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true')
""")

# COMMAND ----------

# Refresh the Lakebase online store (synced table) so low-latency lookups
# reflect the newly ingested rows.
import requests
ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
host = ctx.apiUrl().get()
token = ctx.apiToken().get()
pipeline_id = "9604daa2-3ec4-4912-965f-4eee7fe408b6"
resp = requests.post(
    f"{host}/api/2.0/pipelines/{pipeline_id}/updates",
    headers={"Authorization": f"Bearer {token}"},
    json={"full_refresh": False},
)
print("synced-table refresh:", resp.status_code, resp.text)
