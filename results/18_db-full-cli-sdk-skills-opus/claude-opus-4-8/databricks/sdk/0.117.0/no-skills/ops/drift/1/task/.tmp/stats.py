from databricks.sdk import WorkspaceClient
import os
w = WorkspaceClient()
cat, sch = os.environ['MLPAB_DATABRICKS_SCHEMA'].split('.')
wh = '4dfab06c923fe3cc'
tbl = f'{cat}.{sch}.features'

def sql(q):
    r = w.statement_execution.execute_statement(warehouse_id=wh, statement=q, wait_timeout='50s')
    if r.status.state.value != 'SUCCEEDED':
        print('ERR', r.status.error)
        return None
    return r.result.data_array

q = f'''SELECT date(event_time) d,
  round(avg(f1),3), round(stddev(f1),3),
  round(avg(f2),3), round(stddev(f2),3),
  round(avg(f3),3), round(stddev(f3),3),
  round(avg(f4),3), round(stddev(f4),3),
  round(avg(f5),3), round(stddev(f5),3),
  round(avg(f6),3), round(stddev(f6),3)
FROM {tbl} GROUP BY date(event_time) ORDER BY d'''
rows = sql(q)
hdr = ['date','f1m','f1s','f2m','f2s','f3m','f3s','f4m','f4s','f5m','f5s','f6m','f6s']
print(' '.join(h.rjust(7) for h in hdr))
for r in rows:
    print(' '.join(str(x).rjust(7) for x in r))
