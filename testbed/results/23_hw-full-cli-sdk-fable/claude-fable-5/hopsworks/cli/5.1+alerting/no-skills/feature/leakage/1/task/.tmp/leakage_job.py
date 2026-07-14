"""Platform job: detect the label-leaking feature in training_data.csv.

Reads the uploaded CSV from HopsFS, computes per-feature association with
the binary label (Pearson correlation and single-feature ROC AUC), and
uploads the results JSON back to HopsFS.
"""

import json

import numpy as np
import pandas as pd

import hopsworks

FEATURES = ["f1", "f2", "f3", "f4", "f5", "f6"]


def single_feature_auc(x, y):
    """ROC AUC of raw feature values as scores, via the rank-sum identity."""
    ranks = pd.Series(x).rank(method="average").to_numpy()
    n_pos = int(y.sum())
    n_neg = int(len(y) - n_pos)
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def main():
    project = hopsworks.login()
    dataset_api = project.get_dataset_api()
    local_csv = dataset_api.download(
        "Resources/leakage/training_data.csv", overwrite=True
    )
    df = pd.read_csv(local_csv)
    y = df["label"].to_numpy()

    results = {}
    for feat in FEATURES:
        x = df[feat].to_numpy()
        corr = float(np.corrcoef(x, y)[0, 1])
        auc = single_feature_auc(x, y)
        # separation: distance between class means in pooled-std units
        m1, m0 = x[y == 1].mean(), x[y == 0].mean()
        pooled_std = float(np.sqrt((x[y == 1].var() + x[y == 0].var()) / 2))
        results[feat] = {
            "pearson_corr_with_label": corr,
            "single_feature_auc": auc,
            "auc_effective": max(auc, 1 - auc),
            "class_mean_separation_stddevs": abs(m1 - m0) / pooled_std,
        }

    leaky = max(results, key=lambda f: results[f]["auc_effective"])
    out = {"feature": leaky, "per_feature_results": results}
    with open("leakage_analysis.json", "w") as fh:
        json.dump(out, fh, indent=2)
    dataset_api.upload("leakage_analysis.json", "Resources/leakage", overwrite=True)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
