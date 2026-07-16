"""On-platform feature pipeline: compute the on-demand transformation
(distance_deg, score) by joining each request with its stored profile, then
write the result to an online-enabled feature group `scoredf7707e` v1.

Runs as a Hopsworks PySpark job (all join/arithmetic happens on the platform).
"""
import hopsworks
from pyspark.sql import SparkSession
from pyspark.sql.functions import col, sqrt, pow as ppow, round as sround

spark = SparkSession.builder.appName("odt_score").getOrCreate()

project = hopsworks.login()
fs = project.get_feature_store()

base = "/Projects/{}/Resources/odt".format(project.name)
req = spark.read.csv(base + "/requests.csv", header=True, inferSchema=True)
prof = spark.read.csv(base + "/profiles.csv", header=True, inferSchema=True)

print("REQUEST_ROWS", req.count(), "PROFILE_ROWS", prof.count())

# Join request-time coordinates with the stored profile on account_id.
j = req.join(prof, on="account_id", how="inner")

# On-demand transformation:
#   distance_deg = sqrt((req_lat-home_lat)^2 + (req_lon-home_lon)^2), round 6
#   score = base_score - 0.1 * distance_deg (rounded distance), round 6
j = j.withColumn(
    "distance_deg",
    sround(
        sqrt(
            ppow(col("request_lat") - col("home_lat"), 2)
            + ppow(col("request_lon") - col("home_lon"), 2)
        ),
        6,
    ),
)
j = j.withColumn("score", sround(col("base_score") - 0.1 * col("distance_deg"), 6))

out = j.select(
    col("request_id").cast("string"),
    col("account_id").cast("string"),
    col("distance_deg").cast("double"),
    col("score").cast("double"),
)

n = out.count()
print("OUTPUT_ROWS", n)
out.show(5, truncate=False)

fg = fs.get_or_create_feature_group(
    name="scoredf7707e",
    version=1,
    description="On-demand distance_deg and score per scoring request",
    primary_key=["request_id"],
    online_enabled=True,
)
fg.insert(out)
print("DONE_INSERT scoredf7707e v1 rows=", n)
