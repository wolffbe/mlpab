#!/usr/bin/env python3
import pandas as pd
import numpy as np

# Read data
train = pd.read_csv("data/features_train.csv")
serve = pd.read_csv("data/features_serve.csv")

# Compute mean and std for training data only
means = train[['f1', 'f2', 'f3', 'f4']].mean()
stds = train[['f1', 'f2', 'f3', 'f4']].std(ddof=0)  # Population std

# Standardize both splits
for df, split_name in [(train, "train"), (serve, "serve")]:
    for col in ['f1', 'f2', 'f3', 'f4']:
        df[f"{col}"] = ((df[col] - means[col]) / stds[col]).round(6)
    df['split'] = split_name

# Combine and save
combined = pd.concat([train, serve])
combined['row_id'] = combined['row_id']  # Ensure row_id is preserved
combined = combined[['row_id', 'split', 'f1', 'f2', 'f3', 'f4']]
combined.to_csv("./tmp/standardized.csv", index=False)

# Output schema and stats for verification
print("Means:")
print(means)
print("\nStds:")
print(stds)