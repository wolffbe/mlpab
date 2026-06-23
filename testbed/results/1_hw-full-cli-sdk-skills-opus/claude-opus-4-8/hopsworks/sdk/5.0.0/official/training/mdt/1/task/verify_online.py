import hopsworks
project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("scaled1f3dc5", version=1)

on = fg.read(online=True)
print("ONLINE rows:", len(on))
print("ONLINE split counts:\n", on["split"].value_counts())
print(on[on.row_id.isin(["R00000", "R00400"])].to_string())
