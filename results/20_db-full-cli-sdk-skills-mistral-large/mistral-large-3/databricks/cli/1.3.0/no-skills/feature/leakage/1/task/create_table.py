from pyspark.sql import SparkSession

# Initialize Spark session
spark = SparkSession.builder.getOrCreate()

# Define the schema and table name
schema = "${MLPAB_DATABRICKS_SCHEMA}"
table_name = "${MLPAB_DATABRICKS_PREFIX}_training_data"
volume_path = "dbfs:/Volumes/workspace/${MLPAB_DATABRICKS_SCHEMA#*.}/${MLPAB_DATABRICKS_PREFIX}_volume/training_data.csv"

# Create the table
spark.read.csv(volume_path, header=True, inferSchema=True).write.saveAsTable(f"{schema}.{table_name}")