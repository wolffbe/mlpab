"""Batch as-of-T logistic scoring job. Runs on the Hopsworks cluster (PySpark)."""
import hopsworks
from pyspark.sql import functions as F
from pyspark.sql.window import Window

# Model (data/model.json) and as-of timestamp T (scoring_request.md)
T = 1773313200000
W1, W2, W3, BIAS = -0.8322, -0.6253, -0.3226, -0.3168

project = hopsworks.login()
fs = project.get_feature_store()

src = fs.get_feature_group("feathist_raw", version=1)
df = src.read()

# Most recent revision at or before T, per account
valid = df.filter(F.col("event_time") <= F.lit(T))
w = Window.partitionBy("account_id").orderBy(F.col("event_time").desc())
asof = valid.withColumn("rn", F.row_number().over(w)).filter(F.col("rn") == 1)

z = F.col("f1") * F.lit(W1) + F.col("f2") * F.lit(W2) + F.col("f3") * F.lit(W3) + F.lit(BIAS)
sigmoid = F.lit(1.0) / (F.lit(1.0) + F.exp(-z))
scored = asof.withColumn("score", F.round(sigmoid, 6)).select("account_id", "score")

print("Scored rows:", scored.count())

out = fs.get_or_create_feature_group(
    name="scoresdc90f7",
    version=1,
    primary_key=["account_id"],
    online_enabled=True,
    description="As-of-T (1773313200000) logistic batch scores: account_id, score",
)
out.insert(scored, write_options={"wait_for_job": True})
print("Done writing scoresdc90f7")
