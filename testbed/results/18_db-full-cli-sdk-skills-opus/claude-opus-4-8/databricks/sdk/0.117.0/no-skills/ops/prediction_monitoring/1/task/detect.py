import time, sys
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
WID = '4dfab06c923fe3cc'
SCH = 'workspace.mlpabae9847'


def run(sql):
    r = w.statement_execution.execute_statement(warehouse_id=WID, statement=sql, wait_timeout='50s')
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(2)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != StatementState.SUCCEEDED:
        print('ERR', r.status.state, r.status.error)
        sys.exit(1)
    return r.result.data_array if r.result else None


if __name__ == '__main__':
    # Daily statistics
    daily = run(f"""
      SELECT date(ts) AS d, count(*) n, round(avg(prediction),4) mean, round(stddev(prediction),4) sd
      FROM {SCH}.pred_log GROUP BY date(ts) ORDER BY d""")
    print("DAILY (date, n, mean, sd):")
    for r in daily:
        print(r[0], r[1], r[2], r[3])
