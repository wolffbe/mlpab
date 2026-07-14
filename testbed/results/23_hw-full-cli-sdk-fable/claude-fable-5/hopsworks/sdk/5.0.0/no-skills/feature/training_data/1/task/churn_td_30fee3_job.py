# Runs INSIDE the Hopsworks cluster as a PySpark job.
# Ingests the churn CSVs into feature groups, builds a point-in-time-correct
# feature view, and materializes training dataset version 1.
import hopsworks
from pyspark.sql import SparkSession
from pyspark.sql import functions as F

spark = SparkSession.builder.getOrCreate()
project = hopsworks.login()
fs = project.get_feature_store()

SUF = "30fee3"
BASE = f"/Projects/{project.name}/Resources/churn{SUF}"


def read_csv(fname):
    return spark.read.csv(f"{BASE}/{fname}", header=True, inferSchema=True)


trans = (
    read_csv("transactions.csv")
    .unionByName(read_csv("transactions_late.csv"))
    .dropDuplicates()
    .withColumn("event_time", F.col("event_time").cast("long"))
)
prof = read_csv("profiles.csv").dropDuplicates().withColumn(
    "event_time", F.col("event_time").cast("long")
)
act = read_csv("activity.csv").dropDuplicates().withColumn(
    "event_time", F.col("event_time").cast("long")
)
health = read_csv("account_health.csv").dropDuplicates().withColumn(
    "event_time", F.col("event_time").cast("long")
)
labels = read_csv("labels.csv").dropDuplicates().withColumn(
    "label_time", F.col("label_time").cast("long")
)


def make_fg(name, evt, df, desc):
    fg = fs.get_or_create_feature_group(
        name=name,
        version=1,
        primary_key=["account_id"],
        event_time=evt,
        online_enabled=False,
        description=desc,
    )
    fg.insert(df, write_options={"wait_for_job": True})
    print(f"inserted {df.count()} rows into {name}")
    return fg


trans_fg = make_fg("transactions" + SUF, "event_time", trans, "transaction features")
prof_fg = make_fg("profiles" + SUF, "event_time", prof, "profile features")
act_fg = make_fg("activity" + SUF, "event_time", act, "activity features")
health_fg = make_fg("account_health" + SUF, "event_time", health, "account health features")
labels_fg = make_fg("labels" + SUF, "label_time", labels, "churn labels")

query = (
    labels_fg.select_all()
    .join(trans_fg.select(["amount", "balance"]), on=["account_id"])
    .join(prof_fg.select(["credit_score", "tier"]), on=["account_id"])
    .join(act_fg.select(["sessions_7d"]), on=["account_id"])
    .join(health_fg.select(["health_score"]), on=["account_id"])
)

fv = fs.get_or_create_feature_view(
    name="churntraining" + SUF,
    version=1,
    query=query,
    labels=["churned"],
    description="Point-in-time correct churn training features",
)
print("feature view:", fv.name, fv.version)

td_version, td_job = fv.create_training_data(
    description="churn training dataset v1",
    data_format="parquet",
)
print("TRAINING DATASET VERSION:", td_version)
