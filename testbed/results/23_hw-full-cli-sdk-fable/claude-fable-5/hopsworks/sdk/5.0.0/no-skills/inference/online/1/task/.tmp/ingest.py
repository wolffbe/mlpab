import pandas as pd
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
print("Project:", project.name)

df = pd.read_csv("data/features.csv")
df["account_id"] = df["account_id"].astype(str)
for c in ["f1", "f2", "f3", "f4"]:
    df[c] = df[c].astype(float)
print(df.head(), df.dtypes, sep="\n")

fg = fs.get_or_create_feature_group(
    name="profiles926b2c",
    version=1,
    primary_key=["account_id"],
    online_enabled=True,
    description="Account feature profiles",
)
fg.insert(df, wait=True)
print("Insert done")
