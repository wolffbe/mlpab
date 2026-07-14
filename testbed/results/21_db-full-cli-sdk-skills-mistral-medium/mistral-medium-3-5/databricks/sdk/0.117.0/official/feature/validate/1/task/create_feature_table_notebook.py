#!/usr/bin/env python3

import csv
import json
import os
from databricks.sdk import WorkspaceClient

def filter_data():
    """Filter the CSV to only include valid rows and return the filtered data."""
    valid_categories = {'grocery', 'travel', 'salary', 'rent', 'other'}
    valid_rows = []
    rejected_rows = []
    
    with open('data/events.csv', 'r') as f:
        reader = csv.DictReader(f)
        for row in reader:
            row_id = row['row_id']
            amount_str = row['amount']
            category = row['category']
            
            # Rule 1: amount is present (not null/empty)
            if amount_str.strip() == '':
                rejected_rows.append(row_id)
                continue
            
            # Rule 2: amount is within [0, 10000] (inclusive)
            try:
                amount = float(amount_str)
                if amount < 0 or amount > 10000:
                    rejected_rows.append(row_id)
                    continue
            except ValueError:
                rejected_rows.append(row_id)
                continue
            
            # Rule 3: category is one of the valid ones
            if category not in valid_categories:
                rejected_rows.append(row_id)
                continue
            
            # If we get here, the row is valid
            valid_rows.append(row)
    
    return valid_rows, rejected_rows

def create_answers_json(rejected_rows):
    """Create the answers.json file with rejected row IDs."""
    # Sort the rejected rows
    rejected_rows.sort()
    
    # Create submission directory if it doesn't exist
    os.makedirs('submission', exist_ok=True)
    
    # Write the answers.json file
    with open('submission/answers.json', 'w') as f:
        json.dump({"rejected": rejected_rows}, f)
    
    print(f"✓ Created submission/answers.json with {len(rejected_rows)} rejected rows")

def main():
    print("Starting feature table creation process...")
    
    # Step 1: Filter the data and create answers.json
    valid_rows, rejected_rows = filter_data()
    create_answers_json(rejected_rows)
    
    print(f"✓ Valid rows: {len(valid_rows)}")
    print(f"✓ Rejected rows: {len(rejected_rows)}")
    
    # Step 2: Connect to Databricks
    ws = WorkspaceClient()
    print("✓ Connected to Databricks workspace")
    
    # Get environment variables
    schema_name = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpab774429')
    prefix = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpab774429')
    
    # Parse schema to get catalog and schema parts
    if '.' in schema_name:
        catalog_name, schema_part = schema_name.split('.', 1)
    else:
        catalog_name = None
        schema_part = schema_name
    
    print(f"✓ Catalog: {catalog_name}")
    print(f"✓ Schema: {schema_part}")
    
    # Get current user
    current_user = ws.current_user.me().user_name
    print(f"✓ Current user: {current_user}")
    
    # Get warehouse
    warehouses = list(ws.warehouses.list())
    warehouse_id = warehouses[0].id if warehouses else None
    print(f"✓ Warehouse ID: {warehouse_id}")
    
    # Step 3: Check if schema exists, create if not
    try:
        schema_info = ws.catalogs.get_schema(catalog_name=catalog_name, schema_name=schema_part)
        print(f"✓ Schema exists: {schema_info.full_name}")
    except Exception as e:
        print(f"✓ Schema doesn't exist, creating: {e}")
        # Try to create it
        try:
            ws.catalogs.create_schema(
                catalog_name=catalog_name,
                schema_name=schema_part,
                comment=f"Schema for {prefix} run"
            )
            print(f"✓ Created schema: {schema_name}")
        except Exception as e2:
            print(f"✗ Error creating schema: {e2}")
            # Maybe the catalog doesn't exist, try to create it
            try:
                ws.catalogs.create_catalog(
                    name=catalog_name,
                    comment=f"Catalog for {prefix} run"
                )
                print(f"✓ Created catalog: {catalog_name}")
                
                # Now create the schema
                ws.catalogs.create_schema(
                    catalog_name=catalog_name,
                    schema_name=schema_part,
                    comment=f"Schema for {prefix} run"
                )
                print(f"✓ Created schema: {schema_name}")
            except Exception as e3:
                print(f"✗ Error creating catalog: {e3}")
    
    # Step 4: Create a notebook that will do the data processing on the platform
    notebook_content = f"""# Databricks notebook source
# MAGIC %md ## Feature Table Creation Notebook

# COMMAND ----------

# MAGIC %md ### Step 1: Define the valid data and create the feature table

# COMMAND ----------

# Define the valid rows as a list of tuples
valid_rows_data = [
"""

    # Add the valid rows data as Python tuples
for row in valid_rows:
    row_id = row['row_id']
    account_id = row['account_id']
    event_time = row['event_time']
    amount = row['amount']
    category = row['category']
    
    # Escape quotes in strings
    row_id_escaped = row_id.replace('"', '\\"').replace("'", "\\'")
    account_id_escaped = account_id.replace('"', '\\"').replace("'", "\\'")
    category_escaped = category.replace('"', '\\"').replace("'", "\\'")
    
    notebook_content += f"    ('{row_id_escaped}', '{account_id_escaped}', {event_time}, {amount}, '{category_escaped}'),\n"

notebook_content += """
]

# COMMAND ----------

# MAGIC %md ### Step 2: Create the feature table

# COMMAND ----------

# Create DataFrame from the valid data
from pyspark.sql import SparkSession
from pyspark.sql.types import StructType, StructField, StringType, DoubleType, LongType

spark = SparkSession.builder.getOrCreate()

# Define schema
schema = StructType([
    StructField("row_id", StringType(), True),
    StructField("account_id", StringType(), True),
    StructField("event_time", LongType(), True),
    StructField("amount", DoubleType(), True),
    StructField("category", StringType(), True)
])

# Create DataFrame
df = spark.createDataFrame(valid_rows_data, schema)

# COMMAND ----------

# MAGIC %md ### Step 3: Save as Delta table

# COMMAND ----------

# Save to the feature table location
table_name = "{catalog_name}.{schema_part}.eventsa45e2a"
df.write.format("delta").mode("overwrite").saveAsTable(table_name)

print(f"✓ Feature table created: {table_name}")
print(f"✓ Row count: {df.count()}")

# COMMAND ----------

# MAGIC %md ### Step 4: Create online table for low-latency access

# COMMAND ----------

from databricks.feature_store import FeatureStoreClient

fs = FeatureStoreClient()

# Create online table
online_table_name = f"{catalog_name}.{schema_part}.eventsa45e2a_online"

try:
    # Try to create online table using the catalog API
    fs.create_online_table(
        name=online_table_name,
        source_table=table_name,
        primary_key=["row_id"],
        timestamp_column="event_time"
    )
    print(f"✓ Online table created: {online_table_name}")
except Exception as e:
    print(f"Error creating online table: {e}")
    
    # Try alternative approach
    try:
        # Publish the table to feature store
        fs.publish_table(
            source_table=table_name,
            online_table_name=online_table_name,
            primary_key=["row_id"],
            timestamp_column="event_time"
        )
        print(f"✓ Feature table published for online access: {online_table_name}")
    except Exception as e2:
        print(f"Error publishing to feature store: {e2}")

# COMMAND ----------

# MAGIC %md ### Step 5: Verify the table

# COMMAND ----------

# Check the table
df_check = spark.read.table(table_name)
print(f"✓ Table verification - row count: {df_check.count()}")
print(f"✓ Columns: {df_check.columns}")

# Show some sample data
df_check.show(5)
"""

    # Create the notebook path
    notebook_path = f"/Users/{current_user}/{prefix}/create_eventsa45e2a_table"
    
    print(f"✓ Creating notebook at: {notebook_path}")
    
    # Create the notebook
    try:
        ws.notebooks.create(
            path=notebook_path,
            content=notebook_content,
            language="PYTHON",
            format="SOURCE"
        )
        print(f"✓ Notebook created successfully")
    except Exception as e:
        print(f"✗ Error creating notebook: {e}")
        # Try to overwrite if it exists
        try:
            ws.notebooks.overwrite(
                path=notebook_path,
                content=notebook_content,
                language="PYTHON",
                format="SOURCE"
            )
            print(f"✓ Notebook overwritten successfully")
        except Exception as e2:
            print(f"✗ Error overwriting notebook: {e2}")
    
    # Step 5: Execute the notebook using a job
    print(f"✓ Setting up notebook execution...")
    
    try:
        # Create a job to run the notebook
        job_name = f"{prefix}_create_eventsa45e2a_table"
        
        job = ws.jobs.create(
            name=job_name,
            tasks=[
                {
                    "task_key": "create_feature_table",
                    "notebook_task": {
                        "notebook_path": notebook_path
                    },
                    "existing_cluster_id": warehouse_id  # Use the warehouse as cluster
                }
            ]
        )
        print(f"✓ Job created: {job.job_id}")
        
        # Run the job
        run = ws.jobs.run_now(job_id=job.job_id)
        print(f"✓ Job run initiated: {run.run_id}")
        
        # Wait for the job to complete
        run_result = ws.jobs.wait_get_run_terminated_or_skipped(run_id=run.run_id)
        print(f"✓ Job run completed with state: {run_result.state}")
        
        if run_result.state == "SUCCESS":
            print("✓ Feature table and online table created successfully!")
        else:
            print(f"✗ Job failed with result: {run_result}")
            
    except Exception as e:
        print(f"✗ Error with job execution: {e}")
    
    print("✓ Script completed")

if __name__ == "__main__":
    main()
