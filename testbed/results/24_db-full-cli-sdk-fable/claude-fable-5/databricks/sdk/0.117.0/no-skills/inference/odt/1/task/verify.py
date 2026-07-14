import os

from databricks.sdk import WorkspaceClient

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
w = WorkspaceClient()

t = w.tables.get(f"{SCHEMA}.scored288ecf")
print("table:", t.full_name, "| columns:", [c.name for c in t.columns])

resp = w.statement_execution.execute_statement(
    warehouse_id="8a93fc195da2ceb1",
    statement=f"SELECT count(*) FROM {SCHEMA}.scored288ecf",
    wait_timeout="50s",
)
print("offline rows:", resp.result.data_array[0][0])

st = w.database.get_synced_database_table(f"{SCHEMA}.scored288ecf_online")
print("synced table:", st.name, "| instance:", st.effective_database_instance_name)
print("sync state:", st.data_synchronization_status.detailed_state)
prog = getattr(st.data_synchronization_status, "provisioning_status", None)
if prog and prog.initial_pipeline_sync_progress:
    print("synced rows:", prog.initial_pipeline_sync_progress.total_row_count)
