import time
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
WH = "a832b544eb7dc3fe"

deadline = time.time() + 900
while time.time() < deadline:
    try:
        r = w.statement_execution.execute_statement(
            warehouse_id=WH, statement="SELECT 1 AS ok", wait_timeout="50s")
        st = str(r.status.state)
        print("state:", st, flush=True)
        if st == "StatementState.SUCCEEDED":
            print("data:", r.result.data_array, flush=True)
            break
        if st == "StatementState.FAILED":
            print("err:", r.status.error, flush=True)
    except Exception as e:
        print("exc:", e, flush=True)
    time.sleep(20)
