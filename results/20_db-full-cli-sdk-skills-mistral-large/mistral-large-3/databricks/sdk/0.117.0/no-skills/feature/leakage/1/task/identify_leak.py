#!/usr/bin/env python3
"""
Identify the feature that leaks the outcome in the training data.
Uses local correlation analysis to find the feature with the highest correlation to the label.
"""

import json
import pandas as pd

# Read the data
df = pd.read_csv("data/training_data.csv")

# Select only numeric columns (features and label)
numeric_cols = [col for col in df.columns if col not in ["row_id"]]
df_numeric = df[numeric_cols]

# Calculate correlation with label
corr_matrix = df_numeric.corr()
corr_with_label = corr_matrix["label"].abs().sort_values(ascending=False)

# The feature with the highest correlation (excluding label itself) is the leaking feature
leaking_feature = corr_with_label.index[1]  # Skip label itself
correlation_value = corr_with_label.iloc[1]

# Prepare the answer
answer = {
    "feature": leaking_feature,
    "evidence": f"Feature {leaking_feature} has the highest absolute correlation ({correlation_value:.4f}) with the label, suggesting it leaks outcome information."
}

# Write the answer to submission/answers.json
with open("submission/answers.json", "w") as f:
    json.dump(answer, f, indent=2)

print(f"Identified leaking feature: {leaking_feature}")
print(f"Answer written to submission/answers.json")