from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
import time

w = WorkspaceClient()
warehouse_id = '4dfab06c923fe3cc'

# Verify row count and date range
sql = "SELECT COUNT(*) as cnt, MIN(ts) as min_ts, MAX(ts) as max_ts, AVG(prediction) as avg_pred FROM workspace.mlpab913631.prediction_log"

stmt = w.statement_execution.execute_statement(
    statement=sql,
    warehouse_id=warehouse_id,
    wait_timeout='30s'
)
print('Count query status:', stmt.status.state)
if stmt.result and stmt.result.data_array:
    print('Result:', stmt.result.data_array)
if stmt.status.error:
    print('Error:', stmt.status.error)

# Check daily averages
sql2 = "SELECT DATE(ts) as day, COUNT(*) as cnt, AVG(prediction) as avg_pred, STDDEV(prediction) as std_pred FROM workspace.mlpab913631.prediction_log GROUP BY DATE(ts) ORDER BY day"

stmt2 = w.statement_execution.execute_statement(
    statement=sql2,
    warehouse_id=warehouse_id,
    wait_timeout='30s'
)
print('\nDaily stats status:', stmt2.status.state)
if stmt2.result and stmt2.result.data_array:
    print('Daily stats (first 10):')
    for row in stmt2.result.data_array[:10]:
        print(row)
    print('...')
    print('Daily stats (last 10):')
    for row in stmt2.result.data_array[-10:]:
        print(row)
