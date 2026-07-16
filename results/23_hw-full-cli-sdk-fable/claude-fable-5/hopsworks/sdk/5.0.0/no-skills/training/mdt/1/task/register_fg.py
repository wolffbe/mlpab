import os

# The sandbox only allows outbound traffic via the localhost proxy; NO_PROXY
# would bypass it for the 10.0.0.0/8 Hopsworks host, so drop it.
os.environ.pop("NO_PROXY", None)
os.environ.pop("no_proxy", None)

import pandas as pd
import hopsworks

train = pd.read_csv("data/features_train.csv")
serve = pd.read_csv("data/features_serve.csv")
train["split"] = "train"
serve["split"] = "serve"

feats = ["f1", "f2", "f3", "f4"]
mean = train[feats].mean()
std = train[feats].std(ddof=0)  # population std over training rows only

df = pd.concat([train, serve], ignore_index=True)
df[feats] = ((df[feats] - mean) / std).round(6)
df = df[["row_id", "split", "f1", "f2", "f3", "f4"]]
print(df.head())
print(df.shape)

project = hopsworks.login()
fs = project.get_feature_store()

fg = fs.get_or_create_feature_group(
    name="scaledaff2b3",
    version=1,
    description="Standardized features (train-stats z-score, rounded to 6 decimals)",
    primary_key=["row_id"],
    online_enabled=True,
)
fg.insert(df, wait=True)
print("Inserted", len(df), "rows into", fg.name, "v", fg.version)
