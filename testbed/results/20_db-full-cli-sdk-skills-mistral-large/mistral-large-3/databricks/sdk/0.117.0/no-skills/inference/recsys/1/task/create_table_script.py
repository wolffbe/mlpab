from pyspark.sql import SparkSession

# Initialize SparkSession
spark = SparkSession.builder.getOrCreate()

# Create the table from the CSV
spark.sql("""
CREATE TABLE IF NOT EXISTS workspace.mlpab07ec26.recse3a36e 
USING CSV
OPTIONS (path '/Volumes/workspace/mlpab07ec26/mlpab07ec26_volume/recommendations.csv', header 'true', inferSchema 'true')
""")

# Enable online access for low-latency lookup
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog

w = WorkspaceClient()
try:
    w.online_tables.create(
        name="workspace.mlpab07ec26.recse3a36e",
        spec=catalog.OnlineTableSpec(
            source_table_full_name="workspace.mlpab07ec26.recse3a36e",
            primary_key_columns=["rec_id"]
        )
    )
    print("Online table created successfully.")
except Exception as e:
    if "already exists" not in str(e):
        raise