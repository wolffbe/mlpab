import hopsworks
import pandas as pd

# Read data
train_df = pd.read_csv("data/features_train.csv")
serve_df = pd.read_csv("data/features_serve.csv")

features = ["f1", "f2", "f3", "f4"]

# Compute mean and population std from training data only
means = train_df[features].mean()
stds = train_df[features].std(ddof=0)  # population std (no Bessel's correction)

print("Training means:", means.to_dict())
print("Training stds:", stds.to_dict())

# Standardize both splits
def standardize(df):
    result = df.copy()
    for f in features:
        result[f] = ((result[f] - means[f]) / stds[f]).round(6)
    return result

train_scaled = standardize(train_df)
serve_scaled = standardize(serve_df)

# Add split column
train_scaled["split"] = "train"
serve_scaled["split"] = "serve"

# Combine and reorder columns
combined = pd.concat([train_scaled, serve_scaled], ignore_index=True)
combined = combined[["row_id", "split", "f1", "f2", "f3", "f4"]]

print(f"Combined shape: {combined.shape}")
print(combined.head(3))
print(combined.tail(3))

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Create feature group with online storage enabled
fg = fs.get_or_create_feature_group(
    name="scaled560eee",
    version=1,
    primary_key=["row_id"],
    online_enabled=True,
    description="Standardized features from train and serve splits",
)

print("Feature group created/retrieved:", fg)

# Insert data
fg.insert(combined)

print("Data inserted successfully.")
print("Feature group name:", fg.name)
print("Feature group version:", fg.version)
print("Online enabled:", fg.online_enabled)
