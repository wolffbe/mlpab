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
    run(f'DROP TABLE IF EXISTS {SCH}.pred_log')
    run(f"""CREATE TABLE {SCH}.pred_log AS
    SELECT to_timestamp(ts) AS ts, CAST(prediction AS DOUBLE) AS prediction
    FROM read_files('/Volumes/workspace/mlpabae9847/predmon/prediction_log.csv',
                    format => 'csv', header => true)""")
    print('count/min/max:', run(f'SELECT COUNT(*), MIN(ts), MAX(ts) FROM {SCH}.pred_log'))
