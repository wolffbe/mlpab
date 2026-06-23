"""
Register a feature table named 'scaled21081b', version 1, on Hopsworks with standardized features.
"""
import hopsworks
import pandas as pd
import numpy as np

# Load data
features_train = pd.read_csv("data/features_train.csv")
features_serve = pd.read_csv("data/features_serve.csv")

# Compute mean and std for each feature using training data only
means = features_train[['f1', 'f2', 'f3', 'f4']].mean()
stds = features_train[['f1', 'f2', 'f3', 'f4']].std(ddof=0)  # Population std

# Standardize both splits
def standardize(df, means, stds):
    df_std = df.copy()
    for col in ['f1', 'f2', 'f3', 'f4']:
        df_std[col] = (df[col] - means[col]) / stds[col]
        df_std[col] = df_std[col].round(6)
    return df_std

features_train_std = standardize(features_train, means, stds)
features_serve_std = standardize(features_serve, means, stds)

# Add split column
features_train_std['split'] = "train"
features_serve_std['split'] = "serve"

# Combine both splits
combined = pd.concat([features_train_std, features_serve_std], axis=0)

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Register feature table
feature_table = fs.create_feature_group(
    name="scaled21081b",
    version=1,
    description="Standardized features (f1, f2, f3, f4) for training and serving splits.",
    primary_key=["row_id"],
    online_enabled=True,
)

# Insert data
feature_table.insert(combined, write_options={"wait_for_job": True})

print("Feature table 'scaled21081b', version 1, has been registered successfully.")