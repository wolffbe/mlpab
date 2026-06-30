# Databricks notebook source
from pyspark.sql import functions as F

CATALOG = "workspace"
SCHEMA = "MLPAB_SCHEMA_PLACEHOLDER"
VOL = f"/Volumes/{CATALOG}/{SCHEMA}/ingest"
TABLE = f"{CATALOG}.{SCHEMA}.scaledd437a3"
FEATS = ["f1", "f2", "f3", "f4"]

schema_str = "row_id STRING, f1 DOUBLE, f2 DOUBLE, f3 DOUBLE, f4 DOUBLE"
train = spark.read.option("header", True).schema(schema_str).csv(f"{VOL}/features_train.csv")
serve = spark.read.option("header", True).schema(schema_str).csv(f"{VOL}/features_serve.csv")

# Compute mean and population standard deviation over TRAINING rows only.
agg_exprs = []
for f in FEATS:
    agg_exprs.append(F.mean(F.col(f)).alias(f"{f}_mean"))
    agg_exprs.append(F.stddev_pop(F.col(f)).alias(f"{f}_std"))
stats = train.agg(*agg_exprs).collect()[0]
print({c: stats[c] for c in stats.asDict()})

def standardize(df, split_name):
    df = df.withColumn("split", F.lit(split_name))
    for f in FEATS:
        m = float(stats[f"{f}_mean"])
        s = float(stats[f"{f}_std"])
        df = df.withColumn(f, F.round((F.col(f) - F.lit(m)) / F.lit(s), 6))
    return df.select("row_id", "split", *FEATS)

out = standardize(train, "train").unionByName(standardize(serve, "serve"))

spark.sql(f"DROP TABLE IF EXISTS {TABLE}")
(out.write.format("delta").mode("overwrite").saveAsTable(TABLE))

spark.sql(f"ALTER TABLE {TABLE} ALTER COLUMN row_id SET NOT NULL")
spark.sql(f"ALTER TABLE {TABLE} ADD CONSTRAINT pk_scaledd437a3 PRIMARY KEY (row_id)")
spark.sql(f"ALTER TABLE {TABLE} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")

n = spark.table(TABLE).count()
print("ROW_COUNT", n)
spark.table(TABLE).orderBy("row_id").show(5, truncate=False)
print("DONE")
