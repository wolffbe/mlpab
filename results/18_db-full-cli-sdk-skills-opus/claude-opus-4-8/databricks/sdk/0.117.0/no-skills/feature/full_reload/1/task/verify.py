from databricks.sdk import WorkspaceClient
import os, time

w = WorkspaceClient()
WH = '4dfab06c923fe3cc'
cat, sch = os.environ['MLPAB_DATABRICKS_SCHEMA'].split('.')
PIPELINE = 'fcc5a21a-66c9-4f28-987b-5abfdd855745'
ONLINE = "`%s`.`%s`.`customers302d18_online`" % (cat, sch)
FQ = "`%s`.`%s`.`customers302d18`" % (cat, sch)


def runsql(sql, fetch=True):
    r = w.statement_execution.execute_statement(statement=sql, warehouse_id=WH, wait_timeout='50s')
    sid = r.statement_id
    st = r.status.state.value
    while st in ('PENDING', 'RUNNING'):
        time.sleep(2)
        r = w.statement_execution.get_statement(sid)
        st = r.status.state.value
    if st != 'SUCCEEDED':
        raise RuntimeError("SQL FAILED: %s :: %s" % (st, r.status.error))
    if fetch and r.result and r.result.data_array is not None:
        return r.result.data_array
    return None


# wait for sync pipeline
for _ in range(60):
    p = w.pipelines.get(PIPELINE)
    state = p.state.value if p.state else None
    latest = None
    try:
        upd = w.pipelines.list_pipeline_events(PIPELINE, max_results=5)
    except Exception:
        pass
    print("pipeline state:", state)
    if state in ('IDLE', 'FAILED'):
        break
    time.sleep(10)

print("=== OFFLINE v2 ===")
print("count:", runsql("SELECT count(*) FROM %s" % FQ))
print("columns:", [r[0] for r in runsql("DESCRIBE TABLE %s" % FQ) if r[0] and not r[0].startswith('#')])
print("=== ONLINE synced table ===")
try:
    print("online count:", runsql("SELECT count(*) FROM %s" % ONLINE))
    print("online sample:", runsql("SELECT * FROM %s LIMIT 2" % ONLINE))
except Exception as e:
    print("online query err:", str(e)[:300])
# confirm UC sees it as a synced/online object
t = w.tables.get("%s.%s.customers302d18_online" % (cat, sch))
print("online UC table type:", t.table_type, "| data_source_format:", t.data_source_format)
