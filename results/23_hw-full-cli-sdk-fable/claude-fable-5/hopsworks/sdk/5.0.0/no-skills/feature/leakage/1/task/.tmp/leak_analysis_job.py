"""Runs on the Hopsworks cluster: ingest training data into a feature group,
then measure how strongly each feature predicts the label (correlation + AUC)
to find the leaky feature. Writes leak_report.json back to Resources."""

import json

import hopsworks
import numpy as np
import pandas as pd

project = hopsworks.login()
fs = project.get_feature_store()

df = pd.read_csv("/hopsfs/Resources/training_data.csv")

# Ingest into a feature group so the data lives in the feature store
try:
    fg = fs.get_or_create_feature_group(
        name="leakage_training_data",
        version=1,
        primary_key=["row_id"],
        description="Training data for leakage detection",
        statistics_config={"enabled": True, "correlations": True, "histograms": True},
    )
    fg.insert(df, wait=True)
    data = fg.read()
except Exception as e:
    print("feature group path failed, using CSV directly:", e)
    data = df

y = data["label"].to_numpy()
features = ["f1", "f2", "f3", "f4", "f5", "f6"]
report = {}
for f in features:
    x = data[f].to_numpy(dtype=float)
    corr = float(np.corrcoef(x, y)[0, 1])
    # single-feature AUC via Mann-Whitney rank statistic
    ranks = pd.Series(x).rank().to_numpy()
    n1 = int(y.sum())
    n0 = len(y) - n1
    auc = float((ranks[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))
    auc = max(auc, 1 - auc)
    # separation of class-conditional means in std units
    d = float(abs(x[y == 1].mean() - x[y == 0].mean()) / x.std())
    report[f] = {"corr_with_label": corr, "abs_corr": abs(corr), "auc": auc, "effect_size": d}

leaky = max(report, key=lambda f: report[f]["auc"])
result = {"per_feature": report, "leaky_feature": leaky}
print(json.dumps(result, indent=2))

with open("/hopsfs/Resources/leak_report.json", "w") as fh:
    json.dump(result, fh, indent=2)
print("report written to /hopsfs/Resources/leak_report.json")
