#!/usr/bin/env python3
"""
Identify the feature that leaks the outcome in the dataset using local analysis.
"""

import pandas as pd
import json

# Load the dataset
data = pd.read_csv("data/training_data.csv")

# Exclude non-feature columns (row_id)
features = data.drop(columns=["row_id", "label"])

# Compute correlation between each feature and the label
correlations = features.corrwith(data["label"])

# Identify the leaking feature (highest absolute correlation)
leaking_feature = correlations.abs().idxmax()

# Write the result to submission/answers.json
result = {
    "feature": leaking_feature,
    "evidence": f"Correlation values: {correlations.to_dict()}"
}

with open("submission/answers.json", "w") as f:
    json.dump(result, f, indent=2)

print(f"Leaking feature: {leaking_feature}")