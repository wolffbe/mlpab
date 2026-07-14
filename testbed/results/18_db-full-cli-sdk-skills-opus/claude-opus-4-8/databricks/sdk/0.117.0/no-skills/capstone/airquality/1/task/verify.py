from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
import time
w = WorkspaceClient()
wid = "4dfab06c923fe3cc"

def run(sql):
    r = w.statement_execution.execute_statement(warehouse_id=wid, statement=sql, wait_timeout="30s")
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(1)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state is not StatementState.SUCCEEDED:
        return "ERR " + str(r.status.error)
    return r.result.data_array if r.result else None

print("pred count/range:", run("SELECT count(*), min(date), max(date) FROM workspace.mlpab23ab6a.airqpred0ecd46"))
print("pred sample:", run("SELECT date, round(pm25_pred,2) FROM workspace.mlpab23ab6a.airqpred0ecd46 ORDER BY date LIMIT 5"))
print("fg cols:", run('SELECT column_name FROM workspace.information_schema.columns WHERE table_schema="mlpab23ab6a" AND table_name="airq0ecd46" ORDER BY ordinal_position'))
print("td count:", run("SELECT count(*) FROM workspace.mlpab23ab6a.airqtd0ecd46"))
print("pred PK:", run('SELECT constraint_name, constraint_type FROM workspace.information_schema.table_constraints WHERE table_schema="mlpab23ab6a" AND table_name="airqpred0ecd46"'))
print("fg PK:", run('SELECT constraint_type FROM workspace.information_schema.table_constraints WHERE table_schema="mlpab23ab6a" AND table_name="airq0ecd46"'))
