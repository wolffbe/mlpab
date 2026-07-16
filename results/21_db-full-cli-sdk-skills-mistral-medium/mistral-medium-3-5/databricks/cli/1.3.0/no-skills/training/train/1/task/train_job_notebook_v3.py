# Databricks notebook source
# MAGIC %python
import numpy as np
import pandas as pd

FEATURES = ["f1", "f2", "f3", "f4", "f5"]
LEARNING_RATE = 0.1
ITERATIONS = 300

def main():
    train = pd.read_csv("/Workspace/mlpab3f14e0_train.csv")
    score = pd.read_csv("/Workspace/mlpab3f14e0_score.csv")
    X = train[FEATURES].to_numpy(dtype=float)
    y = train["label"].to_numpy(dtype=float)
    w = np.zeros(X.shape[1], dtype=float)
    b = 0.0
    for _ in range(ITERATIONS):
        p = 1.0 / (1.0 + np.exp(-(X @ w + b)))
        g = p - y
        w = w - LEARNING_RATE * (X.T @ g) / len(y)
        b = b - LEARNING_RATE * g.mean()
    Xs = score[FEATURES].to_numpy(dtype=float)
    preds = 1.0 / (1.0 + np.exp(-(Xs @ w + b)))
    out = pd.DataFrame({"row_id": score["row_id"], "score": np.round(preds, 6)})
    out.to_csv("/Workspace/mlpab3f14e0_predictions.csv", index=False)

if __name__ == "__main__":
    main()