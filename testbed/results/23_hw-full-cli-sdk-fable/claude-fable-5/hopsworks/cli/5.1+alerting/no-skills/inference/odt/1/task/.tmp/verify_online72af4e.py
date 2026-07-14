"""Verify scored72af4e online store contents from inside the cluster."""

import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("scored72af4e", version=1)

df = fg.read(online=True)
print("online rows:", len(df))
print("columns:", list(df.columns))
print(df.sort_values("request_id").head(3).to_string(index=False))
