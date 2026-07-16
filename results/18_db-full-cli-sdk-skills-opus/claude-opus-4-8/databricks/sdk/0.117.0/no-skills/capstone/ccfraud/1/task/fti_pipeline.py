# Databricks notebook source
# MAGIC %pip install mlflow databricks-feature-engineering scikit-learn

# COMMAND ----------
dbutils.library.restartPython()

# COMMAND ----------
# FTI fraud pipeline — runs entirely on Databricks compute.
import math
import mlflow
import numpy as np
import pandas as pd
from pyspark.sql import functions as F, Window

CAT = "workspace"
SCH = "mlpab8bdf45"
BASE = f"/Volumes/{CAT}/{SCH}/ccdata"
FG = f"{CAT}.{SCH}.cctxn2dbe0a"
TD = f"{CAT}.{SCH}.cctd2dbe0a"
PRED = f"{CAT}.{SCH}.ccpred2dbe0a"
MODEL = f"{CAT}.{SCH}.ccmodel2dbe0a"
EXP = "/Users/benedict@logicalclocks.com/mlpab8bdf45/ccfraud_exp"

# COMMAND ----------
# ---- Load raw ----
tr = (spark.read.csv(f"{BASE}/transactions.csv", header=True, inferSchema=True)
      .withColumn("is_score", F.lit(0)))
sc = (spark.read.csv(f"{BASE}/score_transactions.csv", header=True, inferSchema=True)
      .withColumn("is_fraud", F.lit(None).cast("int"))
      .withColumn("is_score", F.lit(1)))
allp = tr.unionByName(sc, allowMissingColumns=True)
allp = allp.withColumn("ts", F.to_timestamp("datetime"))

# COMMAND ----------
# ---- Feature engineering (Spark) ----
wc = Window.partitionBy("cc_num")
allp = (allp
        .withColumn("card_mean_amt", F.avg("amount").over(wc))
        .withColumn("card_std_amt", F.stddev("amount").over(wc))
        .withColumn("card_mean_lat", F.avg("lat").over(wc))
        .withColumn("card_mean_long", F.avg("long").over(wc)))

word = Window.partitionBy("cc_num").orderBy("ts")
allp = allp.withColumn("prev_ts", F.lag("ts").over(word))
allp = allp.withColumn("secs_since_prev",
                       (F.col("ts").cast("long") - F.col("prev_ts").cast("long")))
allp = allp.fillna({"secs_since_prev": 999999})

wsec = Window.partitionBy("cc_num").orderBy(F.col("ts").cast("long"))
w1h = wsec.rangeBetween(-3600, 0)
w24h = wsec.rangeBetween(-86400, 0)
allp = allp.withColumn("cnt_1h", F.count(F.lit(1)).over(w1h))
allp = allp.withColumn("cnt_24h", F.count(F.lit(1)).over(w24h))
allp = allp.withColumn("txn_rank", F.row_number().over(word))

# haversine distance from card's usual location (km)
R = 6371.0
dlat = F.radians(F.col("lat") - F.col("card_mean_lat"))
dlon = F.radians(F.col("long") - F.col("card_mean_long"))
a = (F.sin(dlat / 2) ** 2
     + F.cos(F.radians(F.col("card_mean_lat"))) * F.cos(F.radians(F.col("lat")))
     * F.sin(dlon / 2) ** 2)
allp = allp.withColumn("geo_dist", R * 2 * F.asin(F.sqrt(a)))

allp = (allp
        .withColumn("hour", F.hour("ts"))
        .withColumn("dow", F.dayofweek("ts"))
        .withColumn("log_amt", F.log1p("amount"))
        .withColumn("amt_z", (F.col("amount") - F.col("card_mean_amt"))
                    / (F.col("card_std_amt") + F.lit(1.0)))
        .withColumn("amt_over_mean", F.col("amount") / (F.col("card_mean_amt") + F.lit(1.0))))
allp = allp.fillna({"geo_dist": 0.0, "amt_z": 0.0, "card_std_amt": 0.0})

feat_cols = ["amount", "log_amt", "hour", "dow", "amt_z", "amt_over_mean",
             "geo_dist", "secs_since_prev", "cnt_1h", "cnt_24h", "txn_rank",
             "card_mean_amt", "card_std_amt"]
fg_df = allp.select(["transaction_id", "cc_num", "category", "is_fraud", "is_score"] + feat_cols)
fg_df = fg_df.dropDuplicates(["transaction_id"])

# COMMAND ----------
# ---- Feature group cctxn2dbe0a ----
from databricks.feature_engineering import FeatureEngineeringClient
fe = FeatureEngineeringClient()
spark.sql(f"DROP TABLE IF EXISTS {FG}")
fe.create_table(
    name=FG,
    primary_keys=["transaction_id"],
    df=fg_df,
    description="Engineered fraud features per transaction (train+score).",
)
print("feature group created:", FG)

# COMMAND ----------
# ---- Training dataset cctd2dbe0a ----
td_df = fg_df.filter(F.col("is_score") == 0).select(
    ["transaction_id", "category", "is_fraud"] + feat_cols)
td_df.write.mode("overwrite").option("overwriteSchema", "true").saveAsTable(TD)
print("training dataset created:", TD, td_df.count())

# COMMAND ----------
# ---- Train model + metrics ----
from sklearn.ensemble import HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, average_precision_score, f1_score

pdf = td_df.toPandas()
cats = sorted([r for r in pdf["category"].dropna().unique().tolist()])
score_pdf = fg_df.filter(F.col("is_score") == 1).select(
    ["transaction_id", "category"] + feat_cols).toPandas()

def build_X(frame):
    X = frame[feat_cols].astype(float).copy()
    for c in cats:
        X[f"cat_{c}"] = (frame["category"] == c).astype(float)
    return X

X = build_X(pdf)
y = pdf["is_fraud"].astype(int).values
Xtr, Xval, ytr, yval = train_test_split(X, y, test_size=0.25, random_state=42, stratify=y)

mlflow.set_registry_uri("databricks-uc")
mlflow.set_experiment(EXP)
with mlflow.start_run(run_name="ccfraud_hgb") as run:
    clf = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08,
                                         max_depth=6, l2_regularization=1.0,
                                         random_state=42)
    clf.fit(Xtr, ytr)
    pval = clf.predict_proba(Xval)[:, 1]
    val_auc = float(roc_auc_score(yval, pval))
    val_ap = float(average_precision_score(yval, pval))
    val_f1 = float(f1_score(yval, (pval >= 0.5).astype(int)))
    mlflow.log_metric("val_roc_auc", val_auc)
    mlflow.log_metric("val_pr_auc", val_ap)
    mlflow.log_metric("val_f1", val_f1)
    mlflow.log_param("model", "HistGradientBoostingClassifier")
    mlflow.log_param("n_features", X.shape[1])
    print("VAL ROC AUC:", val_auc, "PR AUC:", val_ap, "F1:", val_f1)

    # refit on all training data for final scoring
    final = HistGradientBoostingClassifier(max_iter=400, learning_rate=0.08,
                                            max_depth=6, l2_regularization=1.0,
                                            random_state=42)
    final.fit(X, y)
    mlflow.log_metric("train_roc_auc", float(roc_auc_score(y, final.predict_proba(X)[:, 1])))

    from mlflow.models.signature import infer_signature
    sig = infer_signature(X, final.predict_proba(X)[:, 1])
    mlflow.sklearn.log_model(final, artifact_path="model", signature=sig,
                             registered_model_name=MODEL,
                             input_example=X.head(3))
    run_id = run.info.run_id
print("registered model:", MODEL, "run:", run_id)

# COMMAND ----------
# ---- Score score_transactions -> predictions feature table ----
Xsc = build_X(score_pdf)
probs = final.predict_proba(Xsc)[:, 1]
pred_pdf = pd.DataFrame({
    "transaction_id": score_pdf["transaction_id"].values,
    "fraud_probability": np.clip(probs.astype(float), 0.0, 1.0),
})
pred_sdf = spark.createDataFrame(pred_pdf)

spark.sql(f"DROP TABLE IF EXISTS {PRED}")
fe.create_table(
    name=PRED,
    primary_keys=["transaction_id"],
    df=pred_sdf,
    description="Per-transaction fraud probability predictions.",
)
# enable change data feed so an online table can sync from it
spark.sql(f"ALTER TABLE {PRED} SET TBLPROPERTIES (delta.enableChangeDataFeed = true)")
print("predictions table created:", PRED, pred_sdf.count())
print("PIPELINE_DONE")
