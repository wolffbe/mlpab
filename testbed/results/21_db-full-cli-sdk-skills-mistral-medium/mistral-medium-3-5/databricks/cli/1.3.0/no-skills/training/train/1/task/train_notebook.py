# Databricks notebook source
# MAGIC %python
# MAGIC import numpy as np
# MAGIC import pandas as pd
# MAGIC 
# MAGIC FEATURES = ["f1", "f2", "f3", "f4", "f5"]
# MAGIC LEARNING_RATE = 0.1
# MAGIC ITERATIONS = 300
# MAGIC 
# MAGIC def main():
# MAGIC     train = pd.read_csv("/dbfs/FileStore/train.csv")
# MAGIC     score = pd.read_csv("/dbfs/FileStore/score.csv")
# MAGIC     X = train[FEATURES].to_numpy(dtype=float)
# MAGIC     y = train["label"].to_numpy(dtype=float)
# MAGIC     w = np.zeros(X.shape[1], dtype=float)
# MAGIC     b = 0.0
# MAGIC     for _ in range(ITERATIONS):
# MAGIC         p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
# MAGIC         g = p - y
# MAGIC         w = w - LEARNING_RATE * (X.T @ g) / len(y)
# MAGIC         b = b - LEARNING_RATE * g.mean()
# MAGIC     Xs = score[FEATURES].to_numpy(dtype=float)
# MAGIC     preds = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
# MAGIC     out = pd.DataFrame({"row_id": score["row_id"], "score": np.round(preds, 6)})
# MAGIC     out.to_csv("/dbfs/FileStore/predictions.csv", index=False)
# MAGIC 
# MAGIC if __name__ == "__main__":
# MAGIC     main()