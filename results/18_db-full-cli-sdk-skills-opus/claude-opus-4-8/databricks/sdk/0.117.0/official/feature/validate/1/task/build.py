from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
import time, sys, json
w = WorkspaceClient()
WH = '4dfab06c923fe3cc'

def run(sql, fetch=False):
    r = w.statement_execution.execute_statement(warehouse_id=WH, statement=sql, wait_timeout='50s')
    sid = r.statement_id
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(2); r = w.statement_execution.get_statement(sid)
    if r.status.state != StatementState.SUCCEEDED:
        print('SQL FAILED:', r.status.state, r.status.error); sys.exit(1)
    if fetch and r.result and r.result.data_array is not None:
        return r.result.data_array
    return []

C = 'workspace.mlpab6d0586'
FT = f"{C}.events88b330"
VALID = ("amount IS NOT NULL AND amount >= 0 AND amount <= 10000 "
         "AND category IN ('grocery','travel','salary','rent','other')")

# Feature table: PK on record key + event_time as TIMESERIES; CDF for online sync.
run(f"DROP TABLE IF EXISTS {FT}")
run(f"""CREATE TABLE {FT} (
  row_id STRING NOT NULL,
  account_id STRING,
  event_time BIGINT NOT NULL,
  amount DOUBLE,
  category STRING,
  CONSTRAINT events88b330_pk PRIMARY KEY (row_id, event_time TIMESERIES)
) TBLPROPERTIES (delta.enableChangeDataFeed = true)""")

run(f"""INSERT INTO {FT}
SELECT row_id, account_id, event_time, amount, category
FROM {C}.events_raw WHERE {VALID}""")

loaded = run(f"SELECT count(*) FROM {FT}", fetch=True)
print('loaded rows:', loaded)

rej = run(f"SELECT row_id FROM {C}.events_raw WHERE NOT ({VALID}) OR {VALID.replace('amount IS NOT NULL AND ','')} IS NULL ORDER BY row_id", fetch=True)
# simpler, robust rejected query:
rej = run(f"SELECT row_id FROM {C}.events_raw WHERE NOT (amount IS NOT NULL AND amount >= 0 AND amount <= 10000 AND category IN ('grocery','travel','salary','rent','other')) ORDER BY row_id", fetch=True)
rejected = [r[0] for r in rej]
print('rejected count:', len(rejected))
with open('submission/answers.json', 'w') as f:
    json.dump({"rejected": rejected}, f)
print('wrote answers.json')
