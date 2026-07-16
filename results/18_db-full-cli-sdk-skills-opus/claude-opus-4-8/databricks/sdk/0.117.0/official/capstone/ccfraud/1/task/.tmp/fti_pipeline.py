# Databricks notebook source
%pip install databricks-feature-engineering scikit-learn

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
import math
import pandas as pd
import numpy as np
from pyspark.sql import functions as F, Window
import mlflow
from mlflow.models.signature import infer_signature
from databricks.feature_engineering import FeatureEngineeringClient, FeatureLookup
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split

CAT = "workspace"
SCH = "mlpabea3b07"
SCHEMA = f"{CAT}.{SCH}"
FG = f"{SCHEMA}.cctxn2dbe0a"
TD = f"{SCHEMA}.cctd2dbe0a"
PRED = f"{SCHEMA}.ccpred2dbe0a"
MODEL = f"{SCHEMA}.ccmodel2dbe0a"
USER = "benedict@logicalclocks.com"
base = "/Volumes/workspace/mlpabea3b07/ccdata"

spark.sql(f"USE CATALOG {CAT}")
spark.sql(f"USE SCHEMA {SCH}")

train = spark.read.option("header", True).option("inferSchema", True).csv(f"{base}/transactions.csv")
score = spark.read.option("header", True).option("inferSchema", True).csv(f"{base}/score_transactions.csv")

def prep(df):
    df = df.withColumn("ts", F.to_timestamp("datetime"))
    df = df.withColumn("amount", F.col("amount").cast("double"))
    df = df.withColumn("lat", F.col("lat").cast("double")).withColumn("long", F.col("long").cast("double"))
    return df

train = prep(train)
score = prep(score)
print("train", train.count(), "score", score.count())

# COMMAND ----------
# Card-level historical stats from TRAINING data
card_stats = train.groupBy("cc_num").agg(
    F.avg("lat").alias("home_lat"),
    F.avg("long").alias("home_long"),
    F.avg("amount").alias("amt_mean"),
    F.stddev("amount").alias("amt_std"),
    F.count("*").alias("card_txn_count"),
)
# Category fraud rate (target encoding) from TRAINING labels
global_rate = train.agg(F.avg("is_fraud")).first()[0]
cat_rate = train.groupBy("category").agg(F.avg("is_fraud").alias("cat_fraud_rate"))

# Combined ordered timeline for rolling/velocity features
t = train.select("transaction_id", "cc_num", "ts", "amount", "lat", "long", "category").withColumn("src", F.lit("train"))
s = score.select("transaction_id", "cc_num", "ts", "amount", "lat", "long", "category").withColumn("src", F.lit("score"))
allrows = t.unionByName(s)

w_card = Window.partitionBy("cc_num").orderBy("ts")
allrows = allrows.withColumn("prev_ts", F.lag("ts").over(w_card))
allrows = allrows.withColumn("prev_lat", F.lag("lat").over(w_card))
allrows = allrows.withColumn("prev_long", F.lag("long").over(w_card))
allrows = allrows.withColumn("secs_since_prev", F.col("ts").cast("long") - F.col("prev_ts").cast("long"))

w_1h = Window.partitionBy("cc_num").orderBy(F.col("ts").cast("long")).rangeBetween(-3600, -1)
w_24h = Window.partitionBy("cc_num").orderBy(F.col("ts").cast("long")).rangeBetween(-86400, -1)
allrows = allrows.withColumn("velocity_1h", F.count(F.lit(1)).over(w_1h))
allrows = allrows.withColumn("velocity_24h", F.count(F.lit(1)).over(w_24h))
allrows = allrows.withColumn("amt_sum_24h", F.sum("amount").over(w_24h))

def hav(lat1, lon1, lat2, lon2):
    p = F.lit(math.pi / 180.0)
    a = F.sin((lat2 - lat1) * p / 2) ** 2 + F.cos(lat1 * p) * F.cos(lat2 * p) * F.sin((lon2 - lon1) * p / 2) ** 2
    return F.lit(6371.0) * 2 * F.asin(F.sqrt(a))

allrows = allrows.join(card_stats, "cc_num", "left")
allrows = allrows.join(cat_rate, "category", "left")
allrows = allrows.withColumn("dist_from_home", hav(F.col("lat"), F.col("long"), F.col("home_lat"), F.col("home_long")))
allrows = allrows.withColumn(
    "dist_from_prev",
    F.when(F.col("prev_lat").isNotNull(), hav(F.col("lat"), F.col("long"), F.col("prev_lat"), F.col("prev_long"))).otherwise(F.lit(0.0)),
)
allrows = allrows.withColumn("amt_z", (F.col("amount") - F.col("amt_mean")) / (F.col("amt_std") + F.lit(1.0)))
allrows = allrows.withColumn("hour", F.hour("ts"))
allrows = allrows.withColumn("dow", F.dayofweek("ts"))
allrows = allrows.withColumn("is_night", (F.col("hour") < 6).cast("int"))
allrows = allrows.withColumn("log_amount", F.log1p("amount"))
allrows = allrows.withColumn(
    "speed_kmh",
    F.when(F.col("secs_since_prev") > 0, F.col("dist_from_prev") / (F.col("secs_since_prev") / 3600.0)).otherwise(F.lit(0.0)),
)
allrows = allrows.fillna({
    "secs_since_prev": -1, "velocity_1h": 0, "velocity_24h": 0, "amt_sum_24h": 0.0,
    "amt_z": 0.0, "dist_from_home": 0.0, "amt_mean": 0.0, "amt_std": 0.0,
    "card_txn_count": 0, "cat_fraud_rate": float(global_rate),
})

FEAT = ["amount", "log_amount", "amt_z", "dist_from_home", "dist_from_prev", "speed_kmh",
        "secs_since_prev", "velocity_1h", "velocity_24h", "amt_sum_24h",
        "hour", "dow", "is_night", "card_txn_count", "cat_fraud_rate"]

train_feat = allrows.filter(F.col("src") == "train").select(["transaction_id"] + FEAT)
score_feat = allrows.filter(F.col("src") == "score").select(["transaction_id"] + FEAT)

# COMMAND ----------
# 1) Feature group
fe = FeatureEngineeringClient()
spark.sql(f"DROP TABLE IF EXISTS {FG}")
fe.create_table(name=FG, primary_keys=["transaction_id"], df=train_feat,
                description="Engineered credit-card fraud features per transaction")
print("feature group created:", FG)

# COMMAND ----------
# 2) Training dataset assembled from the feature group
label_df = train.select("transaction_id", "is_fraud")
lookups = [FeatureLookup(table_name=FG, lookup_key="transaction_id", feature_names=FEAT)]
training_set = fe.create_training_set(df=label_df, feature_lookups=lookups, label="is_fraud", exclude_columns=[])
td_df = training_set.load_df()
td_df.write.format("delta").mode("overwrite").option("overwriteSchema", "true").saveAsTable(TD)
print("training dataset created:", TD)

# COMMAND ----------
# 3) Train + register classifier with metrics
pdf = spark.table(TD).toPandas()
X = pdf[FEAT].astype(float).fillna(0.0).values
y = pdf["is_fraud"].astype(int).values
Xtr, Xv, ytr, yv = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

def sample_weights(yy):
    n = len(yy); npos = int(yy.sum()); nneg = n - npos
    w = np.where(yy == 1, n / (2.0 * max(npos, 1)), n / (2.0 * max(nneg, 1)))
    return w

params = dict(max_iter=400, learning_rate=0.06, max_leaf_nodes=31,
              l2_regularization=1.0, random_state=42)
clf = HistGradientBoostingClassifier(**params)
clf.fit(Xtr, ytr, sample_weight=sample_weights(ytr))
val_auc = roc_auc_score(yv, clf.predict_proba(Xv)[:, 1])
print("VAL ROC AUC:", val_auc)

clf_full = HistGradientBoostingClassifier(**params)
clf_full.fit(X, y, sample_weight=sample_weights(y))
full_auc = roc_auc_score(y, clf_full.predict_proba(X)[:, 1])

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(f"/Users/{USER}/mlpabea3b07/ccfraud_exp")
sig = infer_signature(pd.DataFrame(X, columns=FEAT), clf_full.predict_proba(X)[:, 1])
with mlflow.start_run(run_name="ccfraud") as run:
    mlflow.log_params(params)
    mlflow.log_metric("val_roc_auc", float(val_auc))
    mlflow.log_metric("train_roc_auc", float(full_auc))
    mlflow.sklearn.log_model(
        sk_model=clf_full, artifact_path="model",
        registered_model_name=MODEL,
        signature=sig,
        input_example=pd.DataFrame(X[:5], columns=FEAT),
    )
print("model registered:", MODEL, "val_auc", val_auc)

# COMMAND ----------
# 4) Score every row of score_transactions.csv -> predictions feature table
spdf = score_feat.toPandas()
Xs = spdf[FEAT].astype(float).fillna(0.0).values
spdf["fraud_probability"] = clf_full.predict_proba(Xs)[:, 1].astype(float)
pred_pdf = spdf[["transaction_id", "fraud_probability"]].copy()
pred_sdf = spark.createDataFrame(pred_pdf)
pred_sdf = pred_sdf.withColumn("fraud_probability", F.col("fraud_probability").cast("double"))
spark.sql(f"DROP TABLE IF EXISTS {PRED}")
fe.create_table(name=PRED, primary_keys=["transaction_id"], df=pred_sdf,
                description="Fraud probability predictions for scored transactions")
print("predictions feature table created:", PRED, "rows", pred_sdf.count())
print("prob range", float(pred_pdf.fraud_probability.min()), float(pred_pdf.fraud_probability.max()))
print("DONE")
