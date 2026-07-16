"""Feature pipeline for `features48dc48` v1.

Runs as a Hopsworks PySpark job on the cluster. All joins/transforms execute
in Spark (platform-side). Reads the transactions + fx-rate CSVs from HopsFS,
derives amount_usd / is_weekend / amount_7d, and writes an online+offline
feature group.
"""
import hopsworks
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

spark = SparkSession.builder.getOrCreate()
# day-of-week / timestamp math must be evaluated in UTC
spark.conf.set("spark.sql.session.timeZone", "UTC")

project = hopsworks.login()
fs = project.get_feature_store()

base = "/Projects/{}/Resources/features48dc48".format(project.name)
tx = spark.read.csv(base + "/transactions.csv", header=True, inferSchema=True)
fx = spark.read.csv(base + "/fx_rates.csv", header=True, inferSchema=True)

# amount_usd = amount * fx_rate of the row's currency
df = tx.join(F.broadcast(fx), on="currency", how="left")
df = df.withColumn("amount_usd", (F.col("amount") * F.col("fx_rate")).cast("double"))

# is_weekend: 1 if event_time (epoch ms) falls on Sat/Sun in UTC
ts = (F.col("event_time") / 1000.0).cast("timestamp")
dow = F.dayofweek(ts)  # 1=Sunday ... 7=Saturday
df = df.withColumn("is_weekend", F.when(dow.isin(1, 7), F.lit(1)).otherwise(F.lit(0)))

# amount_7d: sum of THIS account's amount over [event_time - 7 days, event_time]
# inclusive on both ends. event_time is epoch ms, so the range is in ms.
SEVEN_DAYS_MS = 7 * 24 * 60 * 60 * 1000
w = (
    Window.partitionBy("account_id")
    .orderBy(F.col("event_time").asc())
    .rangeBetween(-SEVEN_DAYS_MS, 0)
)
df = df.withColumn("amount_7d", F.sum(F.col("amount")).over(w).cast("double"))

out = df.select(
    F.col("row_id").cast("string").alias("row_id"),
    F.col("account_id").cast("string").alias("account_id"),
    F.col("event_time").cast("long").alias("event_time"),
    F.col("amount_usd").alias("amount_usd"),
    F.col("is_weekend").alias("is_weekend"),
    F.col("amount_7d").alias("amount_7d"),
)

n = out.count()
print("Computed feature rows:", n)
out.show(5, truncate=False)

fg = fs.get_or_create_feature_group(
    name="features48dc48",
    version=1,
    description="Derived transaction features: USD amount, weekend flag, 7-day rolling account amount.",
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
    stream=True,
    statistics_config=False,
)

fg.insert(out, wait=True)
print("Inserted into features48dc48 v1; rows:", n)
