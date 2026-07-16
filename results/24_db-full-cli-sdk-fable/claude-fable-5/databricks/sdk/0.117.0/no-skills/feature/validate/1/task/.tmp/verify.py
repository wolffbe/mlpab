import json
import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
WAREHOUSE = "8a93fc195da2ceb1"


def sql(stmt):
    r = w.statement_execution.execute_statement(
        warehouse_id=WAREHOUSE, statement=stmt, wait_timeout="50s"
    )
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(2)
        r = w.statement_execution.get_statement(r.statement_id)
    assert r.status.state == StatementState.SUCCEEDED, r.status
    return r.result.data_array or []


st = w.database.get_synced_database_table(f"{SCHEMA}.events385469_online")
s = st.data_synchronization_status
print("synced table state:", s.detailed_state if s else None)

print("table count:", sql(f"SELECT COUNT(*) FROM {SCHEMA}.events385469")[0][0])
print(
    "violations in table:",
    sql(
        f"SELECT COUNT(*) FROM {SCHEMA}.events385469 "
        "WHERE amount IS NULL OR amount < 0 OR amount > 10000 "
        "OR category NOT IN ('grocery','travel','salary','rent','other')"
    )[0][0],
)
print(
    "online (synced) count:",
    sql(f"SELECT COUNT(*) FROM {SCHEMA}.events385469_online")[0][0],
)
print("sample:", sql(f"SELECT * FROM {SCHEMA}.events385469 LIMIT 2"))

ans = json.load(open("submission/answers.json"))
print("rejected in answers.json:", len(ans["rejected"]), ans["rejected"][:5])
overlap = sql(
    f"SELECT COUNT(*) FROM {SCHEMA}.events385469 WHERE row_id IN ("
    + ",".join(f"'{r}'" for r in ans["rejected"])
    + ")"
)[0][0]
print("rejected ids present in table (should be 0):", overlap)
