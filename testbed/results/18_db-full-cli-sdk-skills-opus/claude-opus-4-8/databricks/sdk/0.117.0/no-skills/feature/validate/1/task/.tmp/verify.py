import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
WH = '4dfab06c923fe3cc'
cat, sch = 'workspace', 'mlpabbaf12e'
tbl = f'{cat}.{sch}.events88b330'


def run(sql):
    r = w.statement_execution.execute_statement(warehouse_id=WH, statement=sql, wait_timeout='50s')
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f'{r.status.state}: {r.status.error}')
    return r


# Table info
t = w.tables.get(tbl)
print('feature table:', t.full_name)
print('columns:', [c.name for c in t.columns])
print('constraints:', t.table_constraints)

# counts and rule re-check
r = run(f"SELECT count(*), min(amount), max(amount), count(DISTINCT category) FROM {tbl}")
print('count/min/max/ncat:', r.result.data_array[0])
r = run(f"SELECT count(*) FROM {tbl} WHERE amount IS NULL OR amount<0 OR amount>10000 OR category NOT IN ('grocery','travel','salary','rent','other')")
print('invalid rows in table (should be 0):', r.result.data_array[0][0])

# online synced table
ot = w.tables.get(f'{cat}.{sch}.events88b330_online')
print('online table exists:', ot.full_name)

# answers
ans = json.load(open('submission/answers.json'))
print('rejected count:', len(ans['rejected']))
print('total reconciliation: loaded+rejected =', 651 + len(ans['rejected']))
