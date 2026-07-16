from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sqrt, round

# Initialize Spark
spark = SparkSession.builder.getOrCreate()

# Load data
requests_df = spark.read.csv("/data/requests.csv", header=True, inferSchema=True)
profiles_df = spark.read.csv("/data/profiles.csv", header=True, inferSchema=True)

# Join data
joined_df = requests_df.join(profiles_df, "account_id", "inner")

# Compute distance_deg
joined_df = joined_df.withColumn(
    "distance_deg",
    round(
        sqrt(
            (col("request_lat") - col("home_lat")) ** 2 + 
            (col("request_lon") - col("home_lon")) ** 2
        ),
        6
    )
)

# Compute score
joined_df = joined_df.withColumn(
    "score",
    round(col("base_score") - 0.1 * col("distance_deg"), 6)
)

# Select and write output
output_df = joined_df.select("request_id", "account_id", "distance_deg", "score")
output_table_name = "scoreda4f6e2"
output_schema = "workspace.mlpabc9a00f"

output_df.write.format("delta").mode("overwrite").saveAsTable(f"{output_schema}.{output_table_name}")

# Enable online/real-time access
spark.sql(f"CREATE TABLE IF NOT EXISTS {output_schema}.{output_table_name}_online AS SELECT * FROM {output_schema}.{output_table_name} WITH (ENABLE ONLINE = TRUE);")

print("Table created successfully.")