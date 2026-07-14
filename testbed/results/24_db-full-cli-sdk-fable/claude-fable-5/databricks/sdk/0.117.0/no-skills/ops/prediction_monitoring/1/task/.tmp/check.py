import os, time, sys
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]

for x in w.warehouses.list():
    print("warehouse:", x.name, x.id, x.state)

for t in w.tables.list(catalog_name=schema.split(".")[0], schema_name=schema.split(".")[1]):
    print("table:", t.full_name)
