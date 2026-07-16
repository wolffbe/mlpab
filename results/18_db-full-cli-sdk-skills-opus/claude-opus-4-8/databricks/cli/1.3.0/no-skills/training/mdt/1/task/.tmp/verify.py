# Databricks notebook source
import json
from pyspark.sql import functions as F
T = "workspace.mlpabaa7d89.scaledd437a3"
df = spark.table(T)
res = {}
res["count"] = df.count()
res["count_train"] = df.filter("split='train'").count()
res["count_serve"] = df.filter("split='serve'").count()
# train means of standardized features should be ~0
tr = df.filter("split='train'")
means = tr.agg(*[F.round(F.mean(c), 6).alias(c) for c in ["f1","f2","f3","f4"]]).collect()[0]
res["train_means"] = {c: means[c] for c in ["f1","f2","f3","f4"]}
stds = tr.agg(*[F.round(F.stddev_pop(c), 6).alias(c) for c in ["f1","f2","f3","f4"]]).collect()[0]
res["train_pop_std"] = {c: stds[c] for c in ["f1","f2","f3","f4"]}
sample = df.orderBy("row_id").limit(2).collect()
res["sample"] = [r.asDict() for r in sample]
dbutils.notebook.exit(json.dumps(res))
