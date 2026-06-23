#!/usr/bin/env python3

import pandas as pd
import numpy as np
import json

# Read the training data
df = pd.read_csv("data/training_data.csv")

# Exclude non-numeric columns (row_id)
features = ["f1", "f2", "f3", "f4", "f5", "f6"]

# Compute correlation of each feature with the label
correlations = df[features].corrwith(df["label"]).abs().sort_values(ascending=False)
leaking_feature = correlations.index[0]

# Write the result to submission/answers.json
result = {
    "feature": leaking_feature,
    "evidence": f"Feature '{leaking_feature}' has the highest correlation with the label (|correlation| = {correlations[leaking_feature]:.4f})."
}

with open("submission/answers.json", "w") as f:
    json.dump(result, f, indent=4)

print(f"Identified leaking feature: {leaking_feature}")