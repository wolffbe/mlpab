import hopsworks
import pandas as pd

proj = hopsworks.login()
print("project id", proj.id)
fs = proj.get_feature_store()

df = pd.read_csv("data/features.csv")
for c in ["f1", "f2", "f3", "f4"]:
    df[c] = df[c].astype(float)
print(df.dtypes)
print(df.head())

fg = fs.get_or_create_feature_group(
    name="profiles27ba29",
    version=1,
    primary_key=["account_id"],
    online_enabled=True,
    description="Account feature profiles",
)
print("fg", fg.name, fg.version)
fg.insert(df)
print("insert done")
