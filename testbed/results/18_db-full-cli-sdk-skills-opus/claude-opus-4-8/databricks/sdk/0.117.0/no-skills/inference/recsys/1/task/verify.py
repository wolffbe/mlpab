import os, time
import databricks.sdk
from databricks.sdk.service.sql import StatementState
w = databricks.sdk.WorkspaceClient()
WH = "4dfab06c923fe3cc"
CAT, SCH = os.environ["MLPAB_DATABRICKS_SCHEMA"].split(".")
T = f"`{CAT}`.`{SCH}`.`recs2ead15`"

def run(sql):
    r = w.statement_execution.execute_statement(warehouse_id=WH, statement=sql, wait_timeout="50s")
    sid = r.statement_id
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(2); r = w.statement_execution.get_statement(sid)
    assert r.status.state == StatementState.SUCCEEDED, r.status.error
    return r.result.data_array if r.result else None

print("columns:", run(f"DESCRIBE {T}")[:6])
print("rowcount:", run(f"SELECT count(*) FROM {T}")[0][0])
print("users:", run(f"SELECT count(DISTINCT user_id) FROM {T}")[0][0])
print("rank range:", run(f"SELECT min(rank), max(rank) FROM {T}")[0])
print("rows per user not 5:", run(f"SELECT count(*) FROM (SELECT user_id FROM {T} GROUP BY user_id HAVING count(*)<>5)")[0][0])
print("rec_id format bad:", run(f"SELECT count(*) FROM {T} WHERE rec_id <> concat(user_id,'#',rank)")[0][0])
# leakage: any recommended item already interacted with?
leak = run(f"""SELECT count(*) FROM {T} r JOIN `{CAT}`.`{SCH}`.`interactions` x
  ON r.user_id=x.user_id AND r.item_id=x.item_id""")[0][0]
print("interaction leakage:", leak)
print("PK constraint:", run(f"SELECT constraint_name FROM `{CAT}`.information_schema.table_constraints WHERE table_schema='{SCH}' AND table_name='recs2ead15'"))
print("sample U0000:", run(f"SELECT rec_id,rank,item_id FROM {T} WHERE user_id='U0000' ORDER BY rank"))
