# Databricks notebook source
# MAGIC %pip install databricks-sdk --upgrade -q
dbutils.library.restartPython()

# COMMAND ----------
from databricks.sdk import WorkspaceClient
results = []

w = WorkspaceClient()

# Check if database module works
try:
    import databricks.sdk.service.database as db_svc
    results.append(f"database module OK")

    # Check current synced table status
    try:
        st = w.database.get_synced_database_table(name="mlpab0442b8db.mlpab0442b8.accountse81ff1")
        results.append(f"Synced table state: {st.data_synchronization_status.detailed_state}")
        results.append(f"Synced table message: {st.data_synchronization_status.message}")
    except Exception as e:
        results.append(f"Get synced table: {type(e).__name__}: {e}")

    # Try to delete the failing synced table
    try:
        w.database.delete_synced_database_table(name="mlpab0442b8db.mlpab0442b8.accountse81ff1")
        results.append("Deleted synced table OK")
    except Exception as e:
        results.append(f"Delete synced table: {type(e).__name__}: {e}")

except ImportError as e:
    results.append(f"Import error: {e}")

spark.createDataFrame([(r,) for r in results], ["result"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.fix_output_1")
