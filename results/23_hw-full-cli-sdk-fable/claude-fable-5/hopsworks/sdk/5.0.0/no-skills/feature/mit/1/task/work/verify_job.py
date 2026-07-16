"""Platform job: read featureseb4964 back through offline and online paths."""

import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("featureseb4964", version=1)
print("online_enabled:", fg.online_enabled)

df = fg.read()
print("OFFLINE rows:", len(df))
print(df.sort_values("row_id").head(3).to_string())

odf = fg.read(online=True)
print("ONLINE rows:", len(odf))
print(odf.sort_values("row_id").head(3).to_string())
