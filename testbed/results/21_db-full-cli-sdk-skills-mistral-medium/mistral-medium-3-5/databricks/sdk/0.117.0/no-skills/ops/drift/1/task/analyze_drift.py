#!/usr/bin/env python3
import os
import json
import tempfile
from datetime import datetime, timedelta
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import sql

# Initialize workspace client
w = WorkspaceClient()

# Get environment variables
schema_name = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpabc6cfe1')
prefix = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpabc6cfe1')

# Create a catalog and schema if they don't exist
catalog_name = schema_name.split('.')[0]  # "workspace"
schema_name_only = schema_name.split('.')[1]  # "mlpabc6cfe1"

print(f"Using catalog: {catalog_name}, schema: {schema_name_only}")

# First, let's upload the data to DBFS and create a table
# Read the CSV file
with open('data/features.csv', 'r') as f:
    content = f.read()

# Write to a temp file for upload
temp_csv = tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False)
temp_csv.write(content)
temp_csv.close()

# Upload to DBFS
local_path = temp_csv.name
dbfs_path = f"/FileStore/tables/{prefix}_features.csv"
print(f"Uploading {local_path} to {dbfs_path}")

# Use dbfs API to upload
try:
    w.dbfs.upload(local_path, dbfs_path, overwrite=True)
    print("Upload successful")
except Exception as e:
    print(f"Upload failed: {e}")
    # Try alternative approach - create table directly via SQL
    pass

# Create a table in the specified schema
table_name = f"{prefix}_features"
full_table_name = f"{catalog_name}.{schema_name_only}.{table_name}"

print(f"Creating table: {full_table_name}")

# Create the table using SQL
create_table_sql = f"""
CREATE TABLE IF NOT EXISTS {full_table_name} (
    entity_id STRING,
    event_time TIMESTAMP,
    f1 DOUBLE,
    f2 DOUBLE,
    f3 DOUBLE,
    f4 DOUBLE,
    f5 DOUBLE,
    f6 DOUBLE
) USING CSV OPTIONS (
    path '{dbfs_path}',
    header 'true',
    inferSchema 'true'
)
"""

try:
    w.sql.execute(create_table_sql)
    print("Table created successfully")
except Exception as e:
    print(f"Table creation failed: {e}")
    # Try with different approach
    create_table_sql_alt = f"""
CREATE TABLE IF NOT EXISTS {full_table_name} (
    entity_id STRING,
    event_time TIMESTAMP,
    f1 DOUBLE,
    f2 DOUBLE,
    f3 DOUBLE,
    f4 DOUBLE,
    f5 DOUBLE,
    f6 DOUBLE
) USING DELTA
"""
    try:
        w.sql.execute(create_table_sql_alt)
        print("Table created with DELTA format")
        # Now load data
        load_sql = f"""
COPY INTO {full_table_name} 
FROM '{dbfs_path}' 
FILEFORMAT = CSV 
FORMAT_OPTIONS ('header' = 'true', 'inferSchema' = 'true')
"""
        w.sql.execute(load_sql)
        print("Data loaded successfully")
    except Exception as e2:
        print(f"Alternative approach also failed: {e2}")

# Now analyze the data for drift
# We'll query the data to get statistics over time for each feature
print("\nAnalyzing data for drift...")

# Get the date range
query_dates = f"""
SELECT 
    MIN(event_time) as min_date,
    MAX(event_time) as max_date
FROM {full_table_name}
"""

try:
    result = w.sql.execute(query_dates)
    dates = result.result().as_dict()
    print(f"Date range: {dates}")
except Exception as e:
    print(f"Date query failed: {e}")

# For each feature, we'll calculate statistics over time windows
# Let's check daily statistics for each feature
features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']

print("\nCalculating daily statistics for each feature...")

# Query to get daily averages for all features
daily_stats_query = f"""
SELECT 
    DATE(event_time) as date,
    AVG(f1) as avg_f1, STDDEV(f1) as std_f1,
    AVG(f2) as avg_f2, STDDEV(f2) as std_f2,
    AVG(f3) as avg_f3, STDDEV(f3) as std_f3,
    AVG(f4) as avg_f4, STDDEV(f4) as std_f4,
    AVG(f5) as avg_f5, STDDEV(f5) as std_f5,
    AVG(f6) as avg_f6, STDDEV(f6) as std_f6,
    COUNT(*) as count
FROM {full_table_name}
GROUP BY DATE(event_time)
ORDER BY date
"""

try:
    result = w.sql.execute(daily_stats_query)
    daily_stats = result.result().as_dict()
    
    print("\nDaily statistics:")
    for row in daily_stats:
        date = row['date']
        print(f"{date}: f1={row['avg_f1']:.2f}(±{row['std_f1']:.2f}), "
              f"f2={row['avg_f2']:.2f}(±{row['std_f2']:.2f}), "
              f"f3={row['avg_f3']:.2f}(±{row['std_f3']:.2f}), "
              f"f4={row['avg_f4']:.2f}(±{row['std_f4']:.2f}), "
              f"f5={row['avg_f5']:.2f}(±{row['std_f5']:.2f}), "
              f"f6={row['avg_f6']:.2f}(±{row['std_f6']:.2f})")
    
    # Now analyze for drift - look for significant changes in mean
    print("\nAnalyzing for drift...")
    
    # For each feature, calculate the change in mean between consecutive days
    feature_drift = {}
    for feature in features:
        changes = []
        dates_list = []
        for i in range(1, len(daily_stats)):
            prev_avg = daily_stats[i-1][f'avg_{feature}']
            curr_avg = daily_stats[i][f'avg_{feature}']
            change = abs(curr_avg - prev_avg)
            relative_change = change / abs(prev_avg) if prev_avg != 0 else 0
            changes.append((daily_stats[i]['date'], change, relative_change))
        
        # Find the maximum change
        if changes:
            max_change = max(changes, key=lambda x: x[2])
            feature_drift[feature] = max_change
            print(f"{feature}: max relative change = {max_change[2]:.4f} on {max_change[0]} (absolute: {max_change[1]:.4f})")
    
    # Find the feature with the most significant drift
    drifted_feature = max(feature_drift.items(), key=lambda x: x[1][2])
    onset_date = drifted_feature[1][0]
    
    print(f"\nDrift detected: feature={drifted_feature[0]}, onset={onset_date}")
    
    # Write the answer
    answer = {
        "feature": drifted_feature[0],
        "onset": onset_date
    }
    
    with open('submission/answers.json', 'w') as f:
        json.dump(answer, f)
    
    print(f"Answer written to submission/answers.json: {answer}")
    
except Exception as e:
    print(f"Daily stats query failed: {e}")
    import traceback
    traceback.print_exc()

# Clean up temp file
try:
    os.unlink(temp_csv.name)
except:
    pass
