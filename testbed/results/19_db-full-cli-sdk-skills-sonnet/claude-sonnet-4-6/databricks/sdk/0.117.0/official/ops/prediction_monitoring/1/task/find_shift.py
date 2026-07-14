from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
warehouse_id = '4dfab06c923fe3cc'

# Get all daily stats to find the shift
sql = "SELECT DATE(ts) as day, COUNT(*) as cnt, AVG(prediction) as avg_pred, STDDEV(prediction) as std_pred, MIN(prediction) as min_pred, MAX(prediction) as max_pred FROM workspace.mlpab913631.prediction_log GROUP BY DATE(ts) ORDER BY day"

stmt = w.statement_execution.execute_statement(
    statement=sql,
    warehouse_id=warehouse_id,
    wait_timeout='30s'
)
print('Status:', stmt.status.state)
if stmt.status.error:
    print('Error:', stmt.status.error)

if stmt.result and stmt.result.data_array:
    rows = stmt.result.data_array
    print(f'Total days: {len(rows)}')
    print()
    # Print all rows to find the shift
    prev_avg = None
    for row in rows:
        day, cnt, avg, std, mn, mx = row
        avg_f = float(avg)
        change = ''
        if prev_avg is not None:
            delta = avg_f - prev_avg
            if abs(delta) > 0.3:
                change = f' *** BIG CHANGE: {delta:+.3f}'
        print(f'{day}: avg={avg_f:.3f} std={float(std):.3f} cnt={cnt}{change}')
        prev_avg = avg_f
