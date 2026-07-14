from databricks.sdk import WorkspaceClient
import os, time

w = WorkspaceClient()
WH = '4dfab06c923fe3cc'
cat, sch = os.environ['MLPAB_DATABRICKS_SCHEMA'].split('.')


def runsql(sql, fetch=False):
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


fq = "`%s`.`%s`.`_tsk_test`" % (cat, sch)
runsql("DROP TABLE IF EXISTS %s" % fq)
try:
    runsql("CREATE TABLE %s (row_id STRING NOT NULL, updated_at BIGINT NOT NULL, "
           "CONSTRAINT pk PRIMARY KEY (row_id, updated_at TIMESERIES))" % fq)
    print("BIGINT TIMESERIES: OK")
except Exception as e:
    print("BIGINT TIMESERIES FAILED:", str(e)[:400])
runsql("DROP TABLE IF EXISTS %s" % fq)
