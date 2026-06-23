"""On-demand scoring feature pipeline.

Runs as a Hopsworks PySpark job (platform-side). Reads the raw request and
profile feature groups, joins them on account_id, applies the on-demand
transformation (distance_deg + score) using Spark, and materializes the
result into feature group `scored8a7e3e` v1 with the online store enabled.
"""
import hopsworks
from pyspark.sql import functions as F

project = hopsworks.login()
fs = project.get_feature_store()

requests = fs.get_feature_group("requests_raw", version=1).read()
profiles = fs.get_feature_group("profiles_raw", version=1).read()

# Join request-time coordinates with the stored profile.
joined = requests.join(profiles, on="account_id", how="inner")

# On-demand transformation, computed at request time from the request's
# coordinates combined with the stored profile.
distance = F.round(
    F.sqrt(
        F.pow(F.col("request_lat") - F.col("home_lat"), F.lit(2))
        + F.pow(F.col("request_lon") - F.col("home_lon"), F.lit(2))
    ),
    6,
)

result = joined.withColumn("distance_deg", distance)
result = result.withColumn(
    "score", F.round(F.col("base_score") - F.lit(0.1) * F.col("distance_deg"), 6)
)
result = result.select("request_id", "account_id", "distance_deg", "score")

print("Computed rows:", result.count())
result.show(5, truncate=False)

scored = fs.get_or_create_feature_group(
    name="scored8a7e3e",
    version=1,
    primary_key=["request_id"],
    online_enabled=True,
    description="On-demand scored requests: distance_deg and score per request_id",
)
scored.insert(result)
print("Inserted into scored8a7e3e v1 (online enabled).")
