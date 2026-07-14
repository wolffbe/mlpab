from databricks import sql
import os

# Connect to Databricks SQL
connection = sql.connect(
    server_hostname=os.getenv("DATABRICKS_HOST"),
    http_path="/sql/1.0/warehouses/4dfab06c923fe3cc",
    access_token=os.getenv("DATABRICKS_TOKEN"),
)

cursor = connection.cursor()
cursor.execute("""
    CREATE TABLE workspace.mlpabbc4768.prediction_log 
    USING CSV 
    OPTIONS (
        path 'dbfs:/Volumes/workspace/mlpabbc4768/prediction_volume/prediction_log.csv',
        header 'true',
        inferSchema 'true'
    )
""")

cursor.close()
connection.close()