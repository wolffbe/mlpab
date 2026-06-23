import pandas as pd
import hopsworks
from hsfs.feature import Feature

project = hopsworks.login()
fs = project.get_feature_store()

# Load raw splits (transport only: label the split, no feature math here)
train = pd.read_csv("data/features_train.csv")
train["split"] = "train"
serve = pd.read_csv("data/features_serve.csv")
serve["split"] = "serve"

print("train rows:", len(train), "serve rows:", len(serve))
print(train.dtypes)

stg = fs.get_or_create_feature_group(
    name="scaled1f3dc5_stg",
    version=1,
    description="Raw staging for scaled1f3dc5: unscaled f1-f4 with split label, for platform-side standardization",
    primary_key=["row_id"],
    online_enabled=False,
    features=[
        Feature("row_id", "string", description="Unique row id across both splits"),
        Feature("split", "string", description="train or serve"),
        Feature("f1", "double", description="raw feature f1"),
        Feature("f2", "double", description="raw feature f2"),
        Feature("f3", "double", description="raw feature f3"),
        Feature("f4", "double", description="raw feature f4"),
    ],
    statistics_config=False,
)

cols = ["row_id", "split", "f1", "f2", "f3", "f4"]
stg.insert(train[cols], wait=True)
stg.insert(serve[cols], wait=True)
print("STAGING DONE; fg.id=", stg.id)
print(project.name)
