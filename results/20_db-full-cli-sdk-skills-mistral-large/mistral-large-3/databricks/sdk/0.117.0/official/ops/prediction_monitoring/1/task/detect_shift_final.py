#!/usr/bin/env python3
"""
Detect the onset of a prediction distribution shift using SQL and Volumes.
"""

import os
import json
import tempfile
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog, sql

# Environment variables
SCHEMA = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # workspace.<run-id>
PREFIX = os.getenv("MLPAB_DATABRICKS_PREFIX")  # mlpabdda478
CATALOG = SCHEMA.split(".")[0]  # workspace
SCHEMA_NAME = SCHEMA.split(".")[1]  # mlpabdda478

# Names for UC objects
TABLE_NAME = f"{PREFIX}_prediction_logs"
VOLUME_NAME = f"{PREFIX}_volume"

# Initialize WorkspaceClient
w = WorkspaceClient()


def get_warehouse_id():
    """Get the first available SQL warehouse ID."""
    warehouses = list(w.warehouses.list())
    if not warehouses:
        raise Exception("No SQL warehouses available")
    return warehouses[0].id


def execute_sql(statement: str):
    """Execute a SQL statement and return the result."""
    warehouse_id = get_warehouse_id()
    
    try:
        result = w.statement_execution.execute_statement(
            warehouse_id=warehouse_id,
            catalog=CATALOG,
            schema=SCHEMA_NAME,
            statement=statement
        )
        
        # Wait for completion
        while result.status.state in (sql.StatementState.PENDING, sql.StatementState.RUNNING):
            result = w.statement_execution.get_statement_result(result.statement_id)
        
        if result.status.state != sql.StatementState.SUCCEEDED:
            raise Exception(f"SQL failed: {result.status}")
        
        return result
    except Exception as e:
        raise Exception(f"Failed to execute SQL: {e}")


def create_schema_if_not_exists():
    """Create the schema if it doesn't exist."""
    try:
        w.schemas.create(name=SCHEMA_NAME, catalog_name=CATALOG)
        print(f"Created schema: {CATALOG}.{SCHEMA_NAME}")
    except Exception as e:
        if "already exists" not in str(e):
            raise
        print(f"Schema {CATALOG}.{SCHEMA_NAME} already exists")


def create_volume():
    """Create a Volume to store the CSV file."""
    full_volume_name = f"{CATALOG}.{SCHEMA_NAME}.{VOLUME_NAME}"
    
    try:
        w.volumes.create(
            name=VOLUME_NAME,
            catalog_name=CATALOG,
            schema_name=SCHEMA_NAME,
            volume_type=catalog.VolumeType.MANAGED
        )
        print(f"Created volume: {full_volume_name}")
        return full_volume_name
    except Exception as e:
        if "already exists" not in str(e):
            raise
        print(f"Volume {full_volume_name} already exists")
        return full_volume_name


def create_delta_table():
    """Create a Delta table to ingest prediction logs using SQL."""
    full_table_name = f"{CATALOG}.{SCHEMA_NAME}.{TABLE_NAME}"
    
    # Check if table exists
    try:
        w.tables.get(full_table_name)
        print(f"Table {full_table_name} already exists")
        return full_table_name
    except:
        pass
    
    # Create table using SQL
    create_table_sql = f"""
    CREATE TABLE {full_table_name} (
        ts TIMESTAMP,
        prediction DOUBLE
    )
    USING DELTA
    """
    
    try:
        execute_sql(create_table_sql)
        print(f"Created table: {full_table_name}")
        return full_table_name
    except Exception as e:
        raise Exception(f"Failed to create table: {e}")


def upload_to_volume():
    """Upload prediction_log.csv to the Volume."""
    # Read the local file
    data_path = "./data/prediction_log.csv"
    with open(data_path, "rb") as f:
        file_content = f.read()
    
    # Upload to Volume
    volume_path = f"/Volumes/{CATALOG}/{SCHEMA_NAME}/{VOLUME_NAME}/prediction_log.csv"
    try:
        w.files.upload(volume_path, file_content, overwrite=True)
        print(f"Uploaded file to volume: {volume_path}")
        return volume_path
    except Exception as e:
        raise Exception(f"Failed to upload file to volume: {e}")


def ingest_data(table_name: str):
    """Ingest prediction_log.csv into the Delta table using SQL."""
    # Ingest using SQL
    ingest_sql = f"""
    COPY INTO {table_name}
    FROM '/Volumes/{CATALOG}/{SCHEMA_NAME}/{VOLUME_NAME}/prediction_log.csv'
    FILEFORMAT = CSV
    FORMAT_OPTIONS('header' = 'true', 'inferSchema' = 'true')
    """
    
    try:
        execute_sql(ingest_sql)
        print(f"Ingested data into {table_name}")
    except Exception as e:
        raise Exception(f"Failed to ingest data: {e}")


def detect_distribution_shift(table_name: str) -> str:
    """Detect the onset of a prediction distribution shift."""
    # Query daily statistics
    query = f"""
    SELECT
        DATE(ts) as date,
        AVG(prediction) as mean,
        STDDEV(prediction) as stddev
    FROM {table_name}
    GROUP BY DATE(ts)
    ORDER BY date
    """
    
    try:
        result = execute_sql(query)
        
        # Fetch results
        rows = w.statement_execution.get_statement_result(result.statement_id).result.data_array
        
        if not rows:
            raise Exception("No results returned from query")
        
        # Process results to detect shift
        dates = []
        means = []
        stddevs = []
        
        for row in rows:
            date = row[0]
            mean = float(row[1])
            stddev = float(row[2])
            dates.append(date)
            means.append(mean)
            stddevs.append(stddev)
        
        # Detect shift using rolling z-score
        onset_date = None
        for i in range(1, len(means)):
            prev_mean = means[i-1]
            prev_std = stddevs[i-1]
            current_mean = means[i]
            
            if prev_std == 0:
                continue
            
            z_score = abs(current_mean - prev_mean) / prev_std
            if z_score > 3.0:  # Significant shift
                onset_date = dates[i]
                break
        
        if onset_date is None:
            # Fallback: use the date with the largest mean change
            max_change = 0
            for i in range(1, len(means)):
                change = abs(means[i] - means[i-1])
                if change > max_change:
                    max_change = change
                    onset_date = dates[i]
        
        return onset_date.strftime("%Y-%m-%d")
        
    except Exception as e:
        raise Exception(f"Failed to detect distribution shift: {e}")


def write_answer(onset_date: str):
    """Write the onset date to submission/answers.json."""
    answer = {"onset": onset_date}
    with open("submission/answers.json", "w") as f:
        json.dump(answer, f, indent=2)
    print(f"Wrote onset date: {onset_date}")


def main():
    try:
        # Set up Unity Catalog objects
        create_schema_if_not_exists()
        create_volume()
        table_name = create_delta_table()
        
        # Upload and ingest data
        upload_to_volume()
        ingest_data(table_name)
        
        # Detect distribution shift
        onset_date = detect_distribution_shift(table_name)
        print(f"Detected distribution shift onset: {onset_date}")
        
        # Write answer
        write_answer(onset_date)
        
    except Exception as e:
        print(f"Error: {e}")
        raise


if __name__ == "__main__":
    main()