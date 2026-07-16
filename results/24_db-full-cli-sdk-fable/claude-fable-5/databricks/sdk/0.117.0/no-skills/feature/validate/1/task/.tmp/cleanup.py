import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
r = w.statement_execution.execute_statement(
    warehouse_id="8a93fc195da2ceb1",
    statement=f"DROP TABLE IF EXISTS {SCHEMA}.events385469_staging",
    wait_timeout="50s",
)
while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
    time.sleep(2)
    r = w.statement_execution.get_statement(r.statement_id)
print(r.status.state)
