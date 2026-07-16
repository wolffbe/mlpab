"""Feature pipeline: build derived feature group `featuresa6c730` (v1).

Runs on the Hopsworks platform as a PySpark job. Reads the raw source
feature groups, computes the derived columns with Spark, and writes the
result to an online-enabled feature group.

  amount_usd  = amount * fx_rate(currency)
  is_weekend  = 1 if event_time (epoch ms, UTC) is Sat/Sun else 0
  amount_7d   = sum of this account's `amount` over [event_time - 7d, event_time]
"""
import hopsworks
from pyspark.sql import SparkSession, Window
from pyspark.sql import functions as F

MS_PER_DAY = 86400000
SEVEN_DAYS_MS = 7 * MS_PER_DAY

spark = SparkSession.builder.getOrCreate()
# All time logic is pure epoch arithmetic (UTC); pin the session zone for safety.
spark.conf.set("spark.sql.session.timeZone", "UTC")

project = hopsworks.login()
fs = project.get_feature_store()

tx = fs.get_feature_group("transactions_raw", version=1).read()
fx = fs.get_feature_group("fx_rates", version=1).read()

# amount_7d: per-account rolling 7-day sum, inclusive on both ends, on raw amount.
w = (
    Window.partitionBy("account_id")
    .orderBy("event_time")
    .rangeBetween(-SEVEN_DAYS_MS, 0)
)
tx = tx.withColumn("amount_7d", F.sum(F.col("amount")).over(w))

# is_weekend: epoch-day since 1970-01-01 (a Thursday). (day + 4) % 7 -> 0=Sun..6=Sat.
dow = (F.floor(F.col("event_time") / F.lit(MS_PER_DAY)).cast("long") + F.lit(4)) % F.lit(7)
tx = tx.withColumn(
    "is_weekend", F.when(dow.isin(0, 6), F.lit(1)).otherwise(F.lit(0)).cast("int")
)

# amount_usd: join the currency's fx_rate (fx_rates has a unique currency key -> 1:1).
df = tx.join(F.broadcast(fx), on="currency", how="left")
df = df.withColumn("amount_usd", (F.col("amount") * F.col("fx_rate")).cast("double"))

out = df.select(
    "row_id",
    "account_id",
    F.col("event_time").cast("long").alias("event_time"),
    F.col("amount_usd").cast("double").alias("amount_usd"),
    F.col("is_weekend").cast("int").alias("is_weekend"),
    F.col("amount_7d").cast("double").alias("amount_7d"),
)

print("Derived row count:", out.count())
out.show(10, truncate=False)

target = fs.get_or_create_feature_group(
    name="featuresa6c730",
    version=1,
    description="Derived transaction features (USD amount, weekend flag, 7d rolling sum)",
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
    parents=[
        fs.get_feature_group("transactions_raw", version=1),
        fs.get_feature_group("fx_rates", version=1),
    ],
)

target.insert(out)
print("Inserted into featuresa6c730 v1; online_enabled=True")
