from databricks.sdk import WorkspaceClient
from databricks.sdk.service import database

w = WorkspaceClient()
for inst in w.database.list_database_instances():
    print("instance:", inst.name, inst.state, inst.capacity)

import inspect
print(inspect.signature(database.SyncedDatabaseTable.__init__))
print(inspect.signature(database.DatabaseInstance.__init__))
print(inspect.signature(database.NewPipelineSpec.__init__))
print([e.value for e in database.SyncedTableSchedulingPolicy])
