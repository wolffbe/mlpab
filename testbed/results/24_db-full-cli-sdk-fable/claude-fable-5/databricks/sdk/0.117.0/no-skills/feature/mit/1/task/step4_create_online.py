from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import (
    OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy)

w = WorkspaceClient()
schema = "workspace.mlpab60c44c"
WH = "8a93fc195da2ceb1"

# CDF is required for online table sync pipelines
r = w.statement_execution.execute_statement(
    statement=f"ALTER TABLE {schema}.features3bde51 "
              "SET TBLPROPERTIES (delta.enableChangeDataFeed = true)",
    warehouse_id=WH, wait_timeout="50s")
print("enable CDF:", r.status.state, r.status.error)

spec = OnlineTableSpec(
    source_table_full_name=f"{schema}.features3bde51",
    primary_key_columns=["row_id"],
    timeseries_key=None,
    run_triggered=OnlineTableSpecTriggeredSchedulingPolicy(),
    perform_full_copy=True,
)
ot = w.online_tables.create_and_wait(
    table=OnlineTable(name=f"{schema}.features3bde51_online", spec=spec))
print("online table:", ot.name)
print("status:", ot.status)
