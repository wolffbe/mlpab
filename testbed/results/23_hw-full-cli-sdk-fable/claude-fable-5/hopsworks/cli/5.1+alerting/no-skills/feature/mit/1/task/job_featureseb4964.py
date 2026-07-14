"""PySpark job: build derived feature group featureseb4964 v1."""
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F
from pyspark.sql.types import IntegerType, LongType, DoubleType

import hopsworks

DAY_MS = 86400000

spark = SparkSession.builder.getOrCreate()

project = hopsworks.login()
fs = project.get_feature_store()

base = "hdfs:///Projects/" + project.name + "/Resources/eb4964"

tx = spark.read.csv(base + "/transactions.csv", header=True)
fx = spark.read.csv(base + "/fx_rates.csv", header=True)

tx = (
    tx.withColumn("event_time", F.col("event_time").cast(LongType()))
    .withColumn("amount", F.col("amount").cast(DoubleType()))
)
fx = fx.withColumn("fx_rate", F.col("fx_rate").cast(DoubleType()))

df = tx.join(fx, on="currency", how="left")

df = df.withColumn("amount_usd", F.col("amount") * F.col("fx_rate"))

# Day-of-week from epoch millis, timezone-independent (UTC by construction).
# Epoch day 0 (1970-01-01) was a Thursday; mapping 0=Sun .. 6=Sat.
dow = (F.floor(F.col("event_time") / F.lit(DAY_MS)) + F.lit(4)) % F.lit(7)
df = df.withColumn(
    "is_weekend",
    F.when((dow == 0) | (dow == 6), F.lit(1)).otherwise(F.lit(0)).cast(IntegerType()),
)

w = (
    Window.partitionBy("account_id")
    .orderBy(F.col("event_time"))
    .rangeBetween(-7 * DAY_MS, 0)
)
df = df.withColumn("amount_7d", F.sum("amount").over(w).cast(DoubleType()))

out = df.select(
    "row_id", "account_id", "event_time", "amount_usd", "is_weekend", "amount_7d"
)

print("row count:", out.count())
out.show(5)

fg = fs.get_or_create_feature_group(
    name="featureseb4964",
    version=1,
    description="Derived transaction features: USD amount, weekend flag, 7d rolling sum",
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
)
fg.insert(out, write_options={"wait_for_job": True})
print("insert complete")
