#!/usr/bin/env python3
"""
Ingest prediction logs into Databricks Unity Catalog, enable prediction logging,
and detect distribution shift.
"""

import os
from datetime import datetime
import json
import tempfile
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog

# Environment variables
SCHEMA = os.getenv("MLPAB_DATABRICKS_SCHEMA")  # workspace.<run-id>
PREFIX = os.getenv("MLPAB_DATABRICKS_PREFIX")  # mlpabdda478
CATALOG = SCHEMA.split(".")[0]  # workspace
SCHEMA_NAME = SCHEMA.split(".")[1]  # mlpabdda478

# Names for UC objects
TABLE_NAME = f"{PREFIX}_prediction_logs"
MODEL_NAME = f"{PREFIX}_monitored_model"
ONLINE_TABLE_NAME = f"{PREFIX}_online_prediction_logs"

# Initialize WorkspaceClient
w = WorkspaceClient()


def create_schema_if_not_exists():
    """Create the schema if it doesn't exist."""
    try:
        w.schemas.create(name=SCHEMA_NAME, catalog_name=CATALOG)
        print(f"Created schema: {CATALOG}.{SCHEMA_NAME}")
    except Exception as e:
        if "already exists" not in str(e):
            raise
        print(f"Schema {CATALOG}.{SCHEMA_NAME} already exists")


def create_delta_table():
    """Create a Delta table to ingest prediction logs."""
    full_table_name = f"{CATALOG}.{SCHEMA_NAME}.{TABLE_NAME}"
    
    # Check if table exists
    try:
        w.tables.get(full_table_name)
        print(f"Table {full_table_name} already exists")
        return full_table_name
    except:
        pass
    
    # Create table using Delta Lake API
    try:
        w.tables.create(
            name=TABLE_NAME,
            catalog_name=CATALOG,
            schema_name=SCHEMA_NAME,
            table_type=catalog.TableType.MANAGED,
            data_source_format=catalog.DataSourceFormat.DELTA,
            columns=[
                catalog.Column(name="ts", type_name=catalog.ColumnTypeName.TIMESTAMP),
                catalog.Column(name="prediction", type_name=catalog.ColumnTypeName.DOUBLE)
            ]
        )
        print(f"Created table: {full_table_name}")
        return full_table_name
    except Exception as e:
        raise Exception(f"Failed to create table: {e}")


def ingest_data(table_name: str):
    """Ingest prediction_log.csv into the Delta table."""
    # Read the local file
    data_path = "./data/prediction_log.csv"
    with open(data_path, "r") as f:
        lines = f.readlines()
    
    # Skip header and prepare rows
    rows = [line.strip().split(",") for line in lines[1:]][:13600]
    
    # Create a temporary file for ingestion
    with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False) as tmp:
        tmp.write("ts,prediction\n")
        for row in rows:
            tmp.write(",".join(row) + "\n")
        tmp_path = tmp.name
    
    # Upload to DBFS
    dbfs_path = f"/tmp/{PREFIX}_prediction_log.csv"
    w.dbfs.upload(dbfs_path, tmp_path, overwrite=True)
    
    # Ingest using Delta Lake API
    try:
        w.dbfs.copy(
            src_path=dbfs_path,
            dst_path=f"dbfs:/user/hive/warehouse/{CATALOG}.db/{SCHEMA_NAME}/{TABLE_NAME}/_ingest.csv",
            overwrite=True
        )
        
        # Use SQL to load the data into the Delta table
        load_sql = f"""
        COPY INTO {table_name}
        FROM 'dbfs:/user/hive/warehouse/{CATALOG}.db/{SCHEMA_NAME}/{TABLE_NAME}/_ingest.csv'
        FILEFORMAT = CSV
        FORMAT_OPTIONS('header' = 'true', 'inferSchema' = 'true')
        """
        
        # Execute the SQL using the first available warehouse
        warehouses = list(w.warehouses.list())
        if not warehouses:
            raise Exception("No SQL warehouses available")
        
        statement = w.statement_execution.execute_statement(
            warehouse_id=warehouses[0].id,
            catalog=CATALOG,
            schema=SCHEMA_NAME,
            statement=load_sql
        ).result()
        
        if statement.status.state != "SUCCEEDED":
            raise Exception(f"Failed to ingest data: {statement.status}")
        
        print(f"Ingested data into {table_name}")
    except Exception as e:
        raise Exception(f"Failed to ingest data: {e}")
    finally:
        # Clean up
        os.unlink(tmp_path)
        w.dbfs.delete(dbfs_path)
        w.dbfs.delete(f"dbfs:/user/hive/warehouse/{CATALOG}.db/{SCHEMA_NAME}/{TABLE_NAME}/_ingest.csv")


def register_model():
    """Register a model in Unity Catalog for monitoring."""
    full_model_name = f"{CATALOG}.{SCHEMA_NAME}.{MODEL_NAME}"
    
    try:
        w.model_versions.create(
            name=full_model_name,
            description="Model for prediction monitoring",
            input_example={"ts": "TIMESTAMP", "prediction": "DOUBLE"}
        )
        print(f"Registered model: {full_model_name}")
        return full_model_name
    except Exception as e:
        if "already exists" not in str(e):
            raise
        print(f"Model {full_model_name} already exists")
        return full_model_name


def enable_prediction_logging(table_name: str, model_name: str):
    """Enable prediction logging for the model."""
    # Create an online table for prediction logging
    online_table_full_name = f"{CATALOG}.{SCHEMA_NAME}.{ONLINE_TABLE_NAME}"
    
    try:
        w.online_tables.create(
            name=online_table_full_name,
            spec={
                "source_table_full_name": table_name,
                "primary_key_columns": ["ts"],
                "timeseries_key": "ts"
            }
        )
        print(f"Created online table: {online_table_full_name}")
    except Exception as e:
        if "already exists" not in str(e):
            raise
        print(f"Online table {online_table_full_name} already exists")
    
    # Enable prediction logging for the model
    try:
        w.model_versions.update(
            full_name=model_name,
            prediction_logging_spec={
                "online_table_name": online_table_full_name,
                "granularities": ["1 day"]
            }
        )
        print(f"Enabled prediction logging for model: {model_name}")
    except Exception as e:
        raise Exception(f"Failed to enable prediction logging: {e}")


def detect_distribution_shift(table_name: str) -> str:
    """Detect the onset of a prediction distribution shift."""
    # Query daily statistics
    query = f"""
    SELECT
        DATE(ts) as date,
        AVG(prediction) as mean,
        STDDEV(prediction) as stddev,
        PERCENTILE_CONT(0.25) WITHIN GROUP (ORDER BY prediction) as q1,
        PERCENTILE_CONT(0.5) WITHIN GROUP (ORDER BY prediction) as q2,
        PERCENTILE_CONT(0.75) WITHIN GROUP (ORDER BY prediction) as q3
    FROM {table_name}
    GROUP BY DATE(ts)
    ORDER BY date
    """
    
    # Execute the SQL using the first available warehouse
    warehouses = list(w.warehouses.list())
    if not warehouses:
        raise Exception("No SQL warehouses available")
    
    try:
        statement = w.statement_execution.execute_statement(
            warehouse_id=warehouses[0].id,
            catalog=CATALOG,
            schema=SCHEMA_NAME,
            statement=query
        ).result()
        
        if statement.status.state != "SUCCEEDED":
            raise Exception(f"Query failed: {statement.status}")
        
        # Fetch results
        results = w.statement_execution.get_statement_result(statement.statement_id).result()
        rows = results.get("result", {}).get("data_array", [])
        
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
        table_name = create_delta_table()
        
        # Ingest data
        ingest_data(table_name)
        
        # Register model and enable monitoring
        model_name = register_model()
        enable_prediction_logging(table_name, model_name)
        
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