from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import MonitorTimeSeries, MonitorInferenceLog, MonitorInferenceLogProblemType
import time

w = WorkspaceClient()

user = w.current_user.me().user_name
prefix = 'mlpab913631'
schema = 'workspace.mlpab913631'
table_name = 'workspace.mlpab913631.prediction_log'
assets_dir = f'/Users/{user}/{prefix}/prediction_monitoring'

print(f'Setting up quality monitor for: {table_name}')
print(f'Assets dir: {assets_dir}')

# Create the monitor with time_series configuration to detect distribution changes
try:
    monitor = w.quality_monitors.create(
        table_name=table_name,
        output_schema_name=schema,
        assets_dir=assets_dir,
        time_series=MonitorTimeSeries(
            timestamp_col='ts',
            granularities=['1 day']
        ),
        skip_builtin_dashboard=True
    )
    print('Monitor created:', monitor.monitor_version)
    print('Status:', monitor.status)
    print('Dashboard ID:', monitor.dashboard_id)
except Exception as e:
    print(f'Monitor creation error: {e}')
    # Try to get existing monitor
    try:
        monitor = w.quality_monitors.get(table_name=table_name)
        print('Existing monitor:', monitor.status)
    except Exception as e2:
        print(f'Get monitor error: {e2}')

# Run a refresh
print('\nRunning monitor refresh...')
try:
    refresh = w.quality_monitors.run_refresh(table_name=table_name)
    print('Refresh ID:', refresh.refresh_id)
    print('Refresh state:', refresh.state)

    # Wait for refresh to complete
    for i in range(60):
        time.sleep(10)
        r = w.quality_monitors.get_refresh(table_name=table_name, refresh_id=refresh.refresh_id)
        print(f'  [{i+1}] Refresh state: {r.state}')
        if str(r.state) in ['RefreshState.SUCCESS', 'RefreshState.FAILED', 'RefreshState.CANCELED', 'SUCCESS', 'FAILED', 'CANCELED']:
            break

    print('Final refresh state:', r.state)
except Exception as e:
    print(f'Refresh error: {e}')
