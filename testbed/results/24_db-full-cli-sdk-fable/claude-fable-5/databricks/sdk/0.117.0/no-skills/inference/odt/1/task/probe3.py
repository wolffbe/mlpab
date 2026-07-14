import inspect
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import database

w = WorkspaceClient()
print("database methods:", [a for a in dir(w.database) if not a.startswith('_')])
print()
print("create_synced_database_table:", inspect.signature(w.database.create_synced_database_table))
print("create_database_instance:", inspect.signature(w.database.create_database_instance))
print()
print([a for a in dir(database) if 'Synced' in a or 'Instance' in a][:40])
print()
print("SyncedDatabaseTable:", inspect.signature(database.SyncedDatabaseTable.__init__))
print("SyncedTableSpec:", inspect.signature(database.SyncedTableSpec.__init__))
print("DatabaseInstance:", inspect.signature(database.DatabaseInstance.__init__))
print()
print("existing instances:")
for inst in w.database.list_database_instances():
    print(" ", inst.name, inst.state)
