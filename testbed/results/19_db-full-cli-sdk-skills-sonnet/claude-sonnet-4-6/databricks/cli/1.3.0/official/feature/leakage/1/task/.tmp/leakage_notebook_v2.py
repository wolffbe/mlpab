# Databricks notebook source

import pandas as pd
import numpy as np
from scipy import stats
import json

# COMMAND ----------
# Load data from volume
df = spark.read.csv(
    "/Volumes/workspace/mlpab6007c0/mlpab6007c0_data/training_data.csv",
    header=True, inferSchema=True
)
print(f"Row count: {df.count()}")
df.show(5)

# COMMAND ----------
# Convert to pandas for correlation analysis
pdf = df.toPandas()

features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']

print("=== Pearson Correlation with label (absolute) ===")
corr_results = {}
for col in features:
    corr = abs(pdf[col].corr(pdf['label']))
    corr_results[col] = corr
    print(f"  {col}: {corr:.4f}")

# COMMAND ----------
print("\n=== Point-biserial correlation ===")
pb_results = {}
for col in features:
    corr, pval = stats.pointbiserialr(pdf['label'], pdf[col])
    pb_results[col] = {'corr': abs(corr), 'pval': pval}
    print(f"  {col}: |corr|={abs(corr):.4f}, p-value={pval:.2e}")

print("\nSorted by absolute correlation:")
for col, vals in sorted(pb_results.items(), key=lambda x: -x[1]['corr']):
    print(f"  {col}: |corr|={vals['corr']:.4f}")

# COMMAND ----------
# AUC for each feature alone using Decision Tree
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import DecisionTreeClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator

evaluator = BinaryClassificationEvaluator(labelCol='label', metricName='areaUnderROC')

print("\n=== AUC for each feature alone (Decision Tree depth=5) ===")
auc_results = {}
for col in features:
    assembler = VectorAssembler(inputCols=[col], outputCol='features')
    df_feat = assembler.transform(df)

    dt = DecisionTreeClassifier(featuresCol='features', labelCol='label', maxDepth=5)
    model = dt.fit(df_feat)
    preds = model.transform(df_feat)
    auc = evaluator.evaluate(preds)
    auc_results[col] = float(auc)
    print(f"  {col}: AUC={auc:.4f}")

print("\nSorted by AUC:")
for col, auc in sorted(auc_results.items(), key=lambda x: -x[1]):
    print(f"  {col}: AUC={auc:.4f}")

best_feature = max(auc_results, key=lambda x: auc_results[x])
sorted_aucs = sorted(auc_results.values(), reverse=True)
print(f"\nLeaking feature: {best_feature} (AUC={auc_results[best_feature]:.4f})")

# COMMAND ----------
# Save result to Delta table in the schema
result_data = {
    "feature": best_feature,
    "evidence": f"AUC={auc_results[best_feature]:.4f} for {best_feature} alone predicting label (next best: {sorted_aucs[1]:.4f}); correlation={corr_results[best_feature]:.4f}",
    "all_aucs": str(auc_results),
    "all_corrs": str(corr_results)
}

result_df = spark.createDataFrame([result_data])
result_df.write.mode("overwrite").saveAsTable("workspace.mlpab6007c0.leakage_result")
print("Result saved to table workspace.mlpab6007c0.leakage_result")
print(json.dumps(result_data, indent=2))
dbutils.notebook.exit(json.dumps(result_data))
