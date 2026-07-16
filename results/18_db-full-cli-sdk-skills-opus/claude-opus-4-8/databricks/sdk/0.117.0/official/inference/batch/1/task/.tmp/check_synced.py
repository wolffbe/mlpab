import databricks.sdk as s
import os, sys
w = s.WorkspaceClient()
catalog, schema = os.environ["MLPAB_DATABRICKS_SCHEMA"].split(".")
synced_name = f"{catalog}.{schema}.scores3380ed_online"
cur = w.database.get_synced_database_table(synced_name)
dss = cur.data_synchronization_status
state = dss.detailed_state.value if (dss and dss.detailed_state) else None
prov = cur.unity_catalog_provisioning_state.value if cur.unity_catalog_provisioning_state else None
print("sync state:", state, "prov:", prov, flush=True)
print("message:", getattr(dss, "message", None), flush=True)
print("instance:", cur.effective_database_instance_name, "logical_db:", cur.effective_logical_database_name, flush=True)
