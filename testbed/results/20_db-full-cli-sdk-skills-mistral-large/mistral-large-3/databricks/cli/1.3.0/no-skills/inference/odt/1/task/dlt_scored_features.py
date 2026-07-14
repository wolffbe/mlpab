import dlt
from pyspark.sql.functions import col, sqrt, round

# Load data from Volume
requests_df = spark.read.csv("/Volumes/workspace/mlpabc9a00f/data_volume/requests.csv", header=True, inferSchema=True)
profiles_df = spark.read.csv("/Volumes/workspace/mlpabc9a00f/data_volume/profiles.csv", header=True, inferSchema=True)

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

# Create DLT table
@dlt.table(
    name="scoreda4f6e2",
    comment="Feature table for scoring requests"
)
def create_scored_table():
    return output_df