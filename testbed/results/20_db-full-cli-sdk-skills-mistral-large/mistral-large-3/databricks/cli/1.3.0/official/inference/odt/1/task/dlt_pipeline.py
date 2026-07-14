import dlt
from pyspark.sql.functions import col, udf, round as F_round
from pyspark.sql.types import DoubleType

# Define the UDF for distance calculation
distance_udf = udf(
    lambda req_lat, req_lon, home_lat, home_lon: round(((req_lat - home_lat) ** 2 + (req_lon - home_lon) ** 2) ** 0.5, 6),
    DoubleType()
)

# Read input files
requests_df = spark.read.csv("/Volumes/workspace/mlpab3f631d/input_volume/requests.csv", header=True, inferSchema=True)
profiles_df = spark.read.csv("/Volumes/workspace/mlpab3f631d/input_volume/profiles.csv", header=True, inferSchema=True)

# Join and compute features
joined_df = requests_df.join(profiles_df, "account_id", "inner")
scored_df = joined_df.withColumn(
    "distance_deg", 
    distance_udf(col("request_lat"), col("request_lon"), col("home_lat"), col("home_lon"))
).withColumn(
    "score", 
    F_round(col("base_score") - 0.1 * col("distance_deg"), 6)
).select(
    "request_id", 
    "account_id", 
    "distance_deg", 
    "score"
)

# Write to feature table
@dlt.table(
    name="scoreda4f6e2",
    comment="Feature table for scored requests",
    table_properties={
        "quality": "gold"
    }
)
def create_feature_table():
    return scored_df

# Online access can be enabled later using the Databricks CLI or UI.