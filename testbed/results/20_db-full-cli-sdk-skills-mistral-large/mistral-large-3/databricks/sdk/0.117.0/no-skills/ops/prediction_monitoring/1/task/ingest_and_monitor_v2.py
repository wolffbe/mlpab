#!/usr/bin/env python3
"""
Ingest prediction log into a Unity Catalog table using SQL and set up a quality monitor to detect distribution shifts.
"""

import os
import pandas as pd
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog, sql

# Environment variables
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # workspace.mlpab39e6bb
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]  # mlpab39e6bb
TABLE_NAME = f"{PREFIX}_prediction_log"
FULL_TABLE_NAME = f"{SCHEMA}.{TABLE_NAME}"
OUTPUT_SCHEMA_NAME = SCHEMA  # Same schema for output tables
ASSETS_DIR = f"/Users/{os.environ.get('USER', 'unknown')}/{PREFIX}/monitor_assets"

# Initialize WorkspaceClient
w = WorkspaceClient()

# Read prediction log to get schema
csv_path = "data/prediction_log.csv"
df = pd.read_csv(csv_path)
df['ts'] = pd.to_datetime(df['ts']).dt.strftime('%Y-%m-%dT%H:%M:%SZ')

# Create table using SQL
create_table_sql = f"""
CREATE TABLE IF NOT EXISTS {FULL_TABLE_NAME} (
    ts TIMESTAMP,
    prediction DOUBLE
)
USING DELTA
COMMENT "Table for monitoring prediction distribution shifts"
"""

try:
    w.statement_execution.execute_statement(
        warehouse_id=None,  # Use default warehouse
        catalog=SCHEMA.split('.')[0],
        schema=SCHEMA.split('.')[1],
        statement=create_table_sql,
        disposition=sql.Disposition.INLINE,
        format=sql.Format.JSON_ARRAY
    )
    print("Table created or already exists")
except Exception as e:
    print(f"Failed to create table: {e}")
    raise

# Load data into the table
load_data_sql = f"""
COPY INTO {FULL_TABLE_NAME}
FROM '{csv_path}'
FILEFORMAT = CSV
FORMAT_OPTIONS ('mergeSchema' = 'true', 'header' = 'true')
"""

try:
    w.statement_execution.execute_statement(
        warehouse_id=None,  # Use default warehouse
        catalog=SCHEMA.split('.')[0],
        schema=SCHEMA.split('.')[1],
        statement=load_data_sql,
        disposition=sql.Disposition.INLINE,
        format=sql.Format.JSON_ARRAY
    )
    print("Data loaded into table")
except Exception as e:
    print(f"Failed to load data: {e}")
    raise

# Set up quality monitor
try:
    monitor_info = w.quality_monitors.create(
        table_name=FULL_TABLE_NAME,
        output_schema_name=OUTPUT_SCHEMA_NAME,
        assets_dir=ASSETS_DIR,
        time_series=catalog.MonitorTimeSeries(
            timestamp_col="ts",
            granularity="1 day"
        ),
        slicing_exprs=[],
        skip_builtin_dashboard=False,
        warehouse_id=None  # Use default warehouse
    )
    print(f"Monitor created: {monitor_info}")
except Exception as e:
    print(f"Failed to create monitor: {e}")
    raise

# Run a refresh to compute metrics
refresh_info = w.quality_monitors.run_refresh(table_name=FULL_TABLE_NAME)
print(f"Refresh triggered: {refresh_info}")

# Wait for refresh to complete (polling)
import time
refresh_id = refresh_info.refresh_id
while True:
    refresh_status = w.quality_monitors.get_refresh(table_name=FULL_TABLE_NAME, refresh_id=refresh_id)
    if refresh_status.status == catalog.MonitorRefreshStatusStatus.COMPLETED:
        print("Refresh completed")
        break
    elif refresh_status.status == catalog.MonitorRefreshStatusStatus.FAILED:
        print(f"Refresh failed: {refresh_status.message}")
        raise Exception(f"Refresh failed: {refresh_status.message}")
    else:
        print(f"Refresh status: {refresh_status.status}")
        time.sleep(10)

# Query the metrics table to detect the shift
# The metrics table is named `<output_schema>.monitor_metrics_<table_id>`
metrics_table_name = f"{OUTPUT_SCHEMA_NAME}.monitor_metrics_{TABLE_NAME}"

# Query the metrics table to find the onset of the shift
query = f"""
SELECT 
    window.start as window_start,
    window.end as window_end,
    AVG(metrics.prediction) as avg_prediction,
    STDDEV(metrics.prediction) as stddev_prediction,
    COUNT(*) as count
FROM {metrics_table_name}
GROUP BY window.start, window.end
ORDER BY window.start
"""

try:
    result = w.statement_execution.execute_statement(
        warehouse_id=None,  # Use default warehouse
        catalog=SCHEMA.split('.')[0],
        schema=SCHEMA.split('.')[1],
        statement=query,
        disposition=sql.Disposition.INLINE,
        format=sql.Format.JSON_ARRAY
    )
    
    # Poll for query result
    while True:
        status = w.statement_execution.get_statement_result(result.statement_id)
        if status.status.state == sql.StatementState.SUCCEEDED:
            rows = status.result.data_array
            break
        elif status.status.state == sql.StatementState.FAILED:
            print(f"Query failed: {status.status.error}")
            raise Exception(f"Query failed: {status.status.error}")
        else:
            time.sleep(5)
    
    # Analyze the results to find the shift
    onset_date = None
    prev_avg = None
    for row in rows:
        window_start = row[0]
        avg_prediction = float(row[2])
        stddev_prediction = float(row[3])
        count = int(row[4])
        
        if prev_avg is not None and count > 100:  # Ensure enough data points
            # Detect a significant shift (e.g., > 20% change in mean)
            if abs(avg_prediction - prev_avg) > 0.2 * prev_avg:
                onset_date = window_start.split('T')[0]  # Extract date
                print(f"Shift detected on {onset_date}: {prev_avg} -> {avg_prediction}")
                break
        prev_avg = avg_prediction
    
    if onset_date is None:
        print("No significant shift detected.")
        onset_date = "1970-01-01"  # Fallback
    
    # Write the result to submission/answers.json
    import json
    with open("submission/answers.json", "w") as f:
        json.dump({"onset": onset_date}, f)
    
    print(f"Onset date written to submission/answers.json: {onset_date}")
    
except Exception as e:
    print(f"Failed to query metrics table: {e}")
    raise