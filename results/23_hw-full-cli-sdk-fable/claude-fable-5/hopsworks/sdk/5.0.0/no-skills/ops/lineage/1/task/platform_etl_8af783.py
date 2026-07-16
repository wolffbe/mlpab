# Runs INSIDE the Hopsworks cluster as a PySpark job.
import hopsworks
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

project = hopsworks.login()
fs = project.get_feature_store()
spark = SparkSession.builder.getOrCreate()

base = "/Projects/" + project.name + "/Resources"
df_a = spark.read.csv(base + "/raw_a.csv", header=True, inferSchema=True)
df_b = spark.read.csv(base + "/raw_b.csv", header=True, inferSchema=True)
print("raw_a rows:", df_a.count(), "raw_b rows:", df_b.count())

fg_a = fs.get_or_create_feature_group(
    name="rawa8af783",
    version=1,
    description="Raw table A loaded from raw_a.csv",
    primary_key=["row_id"],
    online_enabled=False,
)
fg_a.insert(df_a)
print("rawa8af783 inserted")

fg_b = fs.get_or_create_feature_group(
    name="rawb8af783",
    version=1,
    description="Raw table B loaded from raw_b.csv",
    primary_key=["row_id"],
    online_enabled=False,
)
fg_b.insert(df_b)
print("rawb8af783 inserted")

# platform-side inner join on row_id via the feature store query API
q = fg_a.select_all().join(fg_b.select_all(), on=["row_id"], join_type="inner")
joined = q.read()
print("joined columns:", joined.columns)

# join auto-prefixes ambiguous columns (e.g. rawb8af783_b_val)
a_col = [c for c in joined.columns if c.endswith("a_val")][0]
b_col = [c for c in joined.columns if c.endswith("b_val")][0]
derived_df = joined.select(
    F.col("row_id"),
    F.round(F.col(a_col) + F.col(b_col), 6).alias("col_sum"),
)
n_derived = derived_df.count()
print("derived rows:", n_derived)

fg_d = fs.get_or_create_feature_group(
    name="derived8af783",
    version=1,
    description="col_sum = round(a_val + b_val, 6) for row_ids present in both rawa8af783 and rawb8af783",
    primary_key=["row_id"],
    online_enabled=True,
    parents=[fg_a, fg_b],
)
fg_d.insert(derived_df)
print("derived8af783 inserted, online_enabled:", fg_d.online_enabled)

# export a copy for client-side verification via the dataset REST API
(
    derived_df.coalesce(1)
    .write.mode("overwrite")
    .option("header", True)
    .csv(base + "/derived8af783_out")
)
print("done")
