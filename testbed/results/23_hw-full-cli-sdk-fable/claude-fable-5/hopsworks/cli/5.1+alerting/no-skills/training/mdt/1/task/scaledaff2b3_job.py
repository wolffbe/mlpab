"""Standardize train+serve splits with train-only stats and write FG scaledaff2b3 v1."""
import pandas as pd
import hopsworks

FEATS = ["f1", "f2", "f3", "f4"]

project = hopsworks.login()
fs = project.get_feature_store()
ds_api = project.get_dataset_api()

train_path = ds_api.download("Resources/aff2b3/features_train.csv", overwrite=True)
serve_path = ds_api.download("Resources/aff2b3/features_serve.csv", overwrite=True)

train = pd.read_csv(train_path)
serve = pd.read_csv(serve_path)

means = train[FEATS].mean()
stds = train[FEATS].std(ddof=0)  # population std over training rows only

train_std = train.copy()
serve_std = serve.copy()
for c in FEATS:
    train_std[c] = ((train[c] - means[c]) / stds[c]).round(6)
    serve_std[c] = ((serve[c] - means[c]) / stds[c]).round(6)

train_std.insert(1, "split", "train")
serve_std.insert(1, "split", "serve")

df = pd.concat([train_std, serve_std], ignore_index=True)
df = df[["row_id", "split"] + FEATS]

fg = fs.get_or_create_feature_group(
    name="scaledaff2b3",
    version=1,
    primary_key=["row_id"],
    online_enabled=True,
    description="Train+serve splits standardized with train-only mean and population std, rounded to 6 decimals",
)
fg.insert(df, wait=True)
print("Inserted", len(df), "rows into scaledaff2b3 v1")
