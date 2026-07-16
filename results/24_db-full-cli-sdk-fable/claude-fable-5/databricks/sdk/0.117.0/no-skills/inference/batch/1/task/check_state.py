from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

print("-- database instances --")
try:
    for inst in w.database.list_database_instances():
        print(inst.name, inst.state)
except Exception as e:
    print("err:", e)

print("-- tables in schema --")
for t in w.tables.list(catalog_name="workspace", schema_name="mlpab4fd108"):
    print(t.full_name, t.table_type)

print("-- synced table check --")
try:
    cur = w.database.get_synced_database_table("workspace.mlpab4fd108.scoresbedd56_online")
    print(cur.name, cur.data_synchronization_status)
except Exception as e:
    print("err:", e)
