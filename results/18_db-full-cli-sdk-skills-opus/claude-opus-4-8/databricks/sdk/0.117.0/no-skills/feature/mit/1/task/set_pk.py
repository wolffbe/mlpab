import sys, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
WH = "4dfab06c923fe3cc"
T = "workspace.mlpab4bb10d.features74f1ef"


def run(sql):
    r = w.statement_execution.execute_statement(statement=sql, warehouse_id=WH, wait_timeout="50s")
    sid = r.statement_id
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(2)
        r = w.statement_execution.get_statement(sid)
    if r.status.state != StatementState.SUCCEEDED:
        print("FAILED:", sql.split(chr(10))[0], r.status.state, getattr(r.status, "error", None))
        sys.exit(1)
    return r


# drop the existing simple PK
run(f"ALTER TABLE {T} DROP CONSTRAINT IF EXISTS pk_features74f1ef")
# event_time must be NOT NULL for timeseries PK
run(f"ALTER TABLE {T} ALTER COLUMN event_time SET NOT NULL")
# time-series feature table: record key row_id, timeseries column event_time
run(f"ALTER TABLE {T} ADD CONSTRAINT pk_features74f1ef PRIMARY KEY (row_id, event_time TIMESERIES)")
print("timeseries PK set")

r = run(f"DESCRIBE TABLE EXTENDED {T}")
for row in r.result.data_array:
    print(row)
