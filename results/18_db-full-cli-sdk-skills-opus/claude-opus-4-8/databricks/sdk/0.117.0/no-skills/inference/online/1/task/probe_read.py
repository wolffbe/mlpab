import time
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
WH = "4dfab06c923fe3cc"
ONLINE = "workspace.mlpab6cf45f.profilesf45007_online"

def run(sql):
    r = w.statement_execution.execute_statement(warehouse_id=WH, statement=sql, wait_timeout="50s")
    st = r.status.state.value if r.status else None
    while st in ("PENDING", "RUNNING"):
        time.sleep(2)
        r = w.statement_execution.get_statement(r.statement_id)
        st = r.status.state.value
    return r, st

r, st = run(f"SELECT account_id, f1, f2, f3, f4 FROM {ONLINE} WHERE account_id='A0004'")
print("state:", st)
if st == "SUCCEEDED":
    print("data:", r.result.data_array if r.result else None)
else:
    print("error:", r.status.error)
