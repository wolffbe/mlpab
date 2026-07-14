"""PySpark job: load accounts batches, keep latest revision per row_id,
insert into feature group accountsd00439 v1 (offline + online)."""

from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql.window import Window

import hopsworks

spark = SparkSession.builder.getOrCreate()

project = hopsworks.login()
fs = project.get_feature_store()

base = f"/Projects/{project.name}/Resources/accounts_batches"
df = spark.read.csv(base, header=True)
df = df.select(
    F.col("row_id").cast("string"),
    F.col("status").cast("string"),
    F.col("balance").cast("double"),
    F.col("updated_at").cast("bigint"),
)
print("total rows read:", df.count())

w = Window.partitionBy("row_id").orderBy(F.col("updated_at").desc())
latest = df.withColumn("_rn", F.row_number().over(w)).filter(F.col("_rn") == 1).drop("_rn")
print("unique row_ids:", latest.count())

fg = fs.get_or_create_feature_group(
    name="accountsd00439",
    version=1,
    description="Accounts table, latest revision per row_id",
    primary_key=["row_id"],
    event_time="updated_at",
    online_enabled=True,
)

fg.insert(latest, wait=True)
print("insert complete")
