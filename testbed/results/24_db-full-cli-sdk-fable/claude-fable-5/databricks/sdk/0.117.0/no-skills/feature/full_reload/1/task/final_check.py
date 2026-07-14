import os

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

FULL_SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
TABLE = f"{FULL_SCHEMA}.customers03eedc"
w = WorkspaceClient()

r = w.statement_execution.execute_statement(
    statement=f"SELECT count(*), count(DISTINCT row_id) FROM {TABLE}",
    warehouse_id="a832b544eb7dc3fe", wait_timeout="50s")
assert r.status.state == StatementState.SUCCEEDED, r.status
print("rows, distinct keys:", r.result.data_array)

t = w.tables.get(TABLE)
print("columns:", [c.name for c in t.columns])
print("pk:", [c.name for c in (t.table_constraints or []) if c.primary_key_constraint])

st = w.database.get_synced_database_table(f"{TABLE}_online")
print("online:", st.name, st.data_synchronization_status.detailed_state)
