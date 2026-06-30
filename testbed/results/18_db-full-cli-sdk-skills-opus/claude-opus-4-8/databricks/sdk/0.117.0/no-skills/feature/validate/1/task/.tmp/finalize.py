import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
WH = '4dfab06c923fe3cc'
cat, sch = 'workspace', 'mlpabbaf12e'
tbl = f'{cat}.{sch}.events88b330'
path = f'/Volumes/{cat}/{sch}/ingest/events.csv'


def run(sql):
    r = w.statement_execution.execute_statement(warehouse_id=WH, statement=sql, wait_timeout='50s')
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f'{r.status.state}: {r.status.error}')
    return r


# Rejected = rows in CSV not present in the loaded valid table
r = run(f"""
SELECT raw.row_id FROM
read_files('{path}', format=>'csv', header=>true,
  schema=>'row_id string, account_id string, event_time string, amount string, category string') raw
LEFT ANTI JOIN {tbl} t ON raw.row_id = t.row_id
ORDER BY raw.row_id
""")
rows = r.result.data_array or []
rejected = [row[0] for row in rows]
print('rejected count:', len(rejected))

import os
os.makedirs('submission', exist_ok=True)
with open('submission/answers.json', 'w') as f:
    json.dump({"rejected": rejected}, f, indent=2)
print('wrote submission/answers.json')
