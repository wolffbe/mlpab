import os, time, sys
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
WH_ID = "a832b544eb7dc3fe"  # Serverless Starter Warehouse

def run(sql, quiet=False):
    r = w.statement_execution.execute_statement(
        statement=sql, warehouse_id=WH_ID, wait_timeout="50s")
    t0 = time.time()
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        if time.time() - t0 > 600:
            raise RuntimeError("query >10min")
        time.sleep(3)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"{r.status.state}: {r.status.error}")
    rows = r.result.data_array if r.result and r.result.data_array else []
    if not quiet:
        for row in rows:
            print(row)
    return rows

if __name__ == "__main__":
    sql = sys.stdin.read() if len(sys.argv) < 2 else sys.argv[1]
    t0 = time.time()
    run(sql)
    print(f"-- ok in {time.time()-t0:.1f}s", file=sys.stderr)
