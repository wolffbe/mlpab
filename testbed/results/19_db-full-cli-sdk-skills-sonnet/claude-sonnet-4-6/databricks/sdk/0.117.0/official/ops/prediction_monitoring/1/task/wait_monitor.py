from databricks.sdk import WorkspaceClient
import time

w = WorkspaceClient()
table_name = 'workspace.mlpab913631.prediction_log'

# Wait for monitor to be ready
for i in range(30):
    monitor = w.quality_monitors.get(table_name=table_name)
    print(f'[{i+1}] Monitor status: {monitor.status}')
    if str(monitor.status) not in ['MonitorInfoStatus.MONITOR_STATUS_PENDING', 'MONITOR_STATUS_PENDING']:
        break
    time.sleep(10)

print('Monitor ready, running refresh...')

# Run the refresh
try:
    refresh = w.quality_monitors.run_refresh(table_name=table_name)
    print('Refresh ID:', refresh.refresh_id)
    print('Refresh state:', refresh.state)

    for i in range(60):
        time.sleep(10)
        r = w.quality_monitors.get_refresh(table_name=table_name, refresh_id=refresh.refresh_id)
        print(f'  [{i+1}] Refresh state: {r.state}')
        state_str = str(r.state)
        if any(x in state_str for x in ['SUCCESS', 'FAILED', 'CANCELED']):
            break

    print('Final refresh state:', r.state)
except Exception as e:
    print(f'Error: {e}')
