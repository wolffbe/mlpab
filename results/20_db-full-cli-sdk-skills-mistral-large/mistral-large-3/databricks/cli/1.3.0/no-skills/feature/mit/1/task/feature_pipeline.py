# Databricks DLT Pipeline for Feature Engineering
import dlt
from pyspark.sql.functions import col, expr, sum as spark_sum, window
from pyspark.sql.types import IntegerType

# Read transactions data
transactions = dlt.read("transactions")\
    .option("header", "true")\
    .option("inferSchema", "true")\
    .csv("file:/workspace/data/transactions.csv")

# Read fx_rates data
fx_rates = dlt.read("fx_rates")\
    .option("header", "true")\
    .option("inferSchema", "true")\
    .csv("file:/workspace/data/fx_rates.csv")

# Join transactions with fx_rates to compute amount_usd
joined_df = transactions.join(fx_rates, "currency", "left")\
    .withColumn("amount_usd", col("amount") * col("fx_rate"))\
    .withColumn("event_time", (col("event_time") / 1000).cast("timestamp"))

# Compute is_weekend
joined_df = joined_df.withColumn(
    "is_weekend",
    expr("CASE WHEN dayofweek(event_time) IN (1, 7) THEN 1 ELSE 0 END").cast(IntegerType())
)

# Compute amount_7d (7-day rolling sum for each account)
window_spec = (
    window(col("event_time"), "7 days")
    .partitionBy("account_id")
    .orderBy(col("event_time").cast("long"))
)

features_df = joined_df.withWatermark("event_time", "7 days")\
    .groupBy("account_id", "row_id", "event_time", "amount_usd", "is_weekend", window_spec)\
    .agg(spark_sum("amount").alias("amount_7d"))\
    .select(
        "row_id",
        "account_id",
        (col("event_time").cast("long") * 1000).alias("event_time"),
        "amount_usd",
        "is_weekend",
        "amount_7d"
    )

# Create the target table
dlt.create_target_table(
    name="featuresfbc05f",
    schema_="row_id STRING, account_id STRING, event_time LONG, amount_usd DOUBLE, is_weekend INT, amount_7d DOUBLE",
    table_properties={
        "quality": "gold",
        "delta.autoOptimize.optimizeWrite": "true",
        "delta.autoOptimize.autoCompact": "true"
    },
    expect="valid_features",
    expect_or_drop="amount_usd IS NOT NULL AND amount_7d IS NOT NULL"
)

# Write the features to the target table
dlt.apply_changes(
    target="featuresfbc05f",
    source=features_df,
    keys=["row_id"],
    sequence_by="event_time",
    apply_as_deletes=None,
    except_column_list=["window"]
)

# Enable online access for low-latency lookup
dlt.create_online_table(
    name="featuresfbc05f_online",
    source="featuresfbc05f",
    primary_key="row_id"
)