import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import ExecuteStatementRequestOnWaitTimeout, StatementState

w = WorkspaceClient()
for wh in w.warehouses.list():
    print(wh.id, wh.name, wh.state, wh.health)

# quick probe statement on each warehouse
for wh in w.warehouses.list():
    try:
        st = w.statement_execution.execute_statement(
            warehouse_id=wh.id, statement="SELECT 1", wait_timeout="30s",
            on_wait_timeout=ExecuteStatementRequestOnWaitTimeout.CONTINUE)
        for _ in range(20):
            if st.status.state not in (StatementState.PENDING, StatementState.RUNNING):
                break
            time.sleep(5)
            st = w.statement_execution.get_statement(st.statement_id)
        print(wh.name, "->", st.status.state, st.status.error)
    except Exception as e:
        print(wh.name, "-> err:", str(e)[:300])
