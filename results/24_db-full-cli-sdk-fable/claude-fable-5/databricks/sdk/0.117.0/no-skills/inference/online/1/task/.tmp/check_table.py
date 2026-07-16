import os

import databricks.sdk

w = databricks.sdk.WorkspaceClient()
schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]

t = w.tables.get(f"{schema}.profiles44b75e_online")
print("table_type:", t.table_type)
print("data_source_format:", t.data_source_format)
print("storage_location:", t.storage_location)
print("properties:", t.properties)

cur = w.database.get_synced_database_table(f"{schema}.profiles44b75e_online")
print("instance:", cur.effective_database_instance_name)
print("logical db:", cur.effective_logical_database_name)
s = cur.data_synchronization_status
print("state:", s.detailed_state)
