"""Verify online store contents of accountsd00439 v1."""
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("accountsd00439", version=1)

df = fg.read(online=True)
print("ONLINE row count:", len(df))
print("ONLINE distinct row_ids:", df["row_id"].nunique())
for rid in ("R00086", "R00161", "R00150"):
    print(df[df["row_id"] == rid].to_dict(orient="records"))
