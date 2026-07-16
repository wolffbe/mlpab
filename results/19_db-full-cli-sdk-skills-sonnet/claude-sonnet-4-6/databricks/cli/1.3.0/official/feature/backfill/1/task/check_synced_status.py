# Databricks notebook source
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.database as db_svc

w = WorkspaceClient()
results = []

# Check the synced table status
try:
    st = w.database.get_synced_database_table(name="mlpab0442b8db.mlpab0442b8.accountse81ff1")
    results.append(f"Status: {st.data_synchronization_status.detailed_state}")
    results.append(f"Message: {st.data_synchronization_status.message}")
    results.append(f"Pipeline ID: {st.data_synchronization_status.pipeline_id}")
    results.append(f"Full: {st}")
except Exception as e:
    results.append(f"Error: {e}")

# Also check the feature table properties
try:
    r = w._api_client.do("GET", "/api/2.0/unity-catalog/tables/workspace.mlpab0442b8.accountse81ff1")
    results.append(f"Feature table constraints: {r.get('table_constraints')}")
except Exception as e:
    results.append(f"Feature table: {e}")

spark.createDataFrame([(r,) for r in results], ["result"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.synced_status")
