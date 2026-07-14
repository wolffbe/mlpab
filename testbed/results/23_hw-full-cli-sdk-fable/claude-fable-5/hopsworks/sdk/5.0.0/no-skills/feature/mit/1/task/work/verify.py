import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("featureseb4964", version=1)
print("name:", fg.name, "version:", fg.version)
print("online_enabled:", fg.online_enabled)
print("primary_key:", fg.primary_key)
print("event_time:", fg.event_time)
print("features:", [(f.name, f.type) for f in fg.features])

df = fg.read()
print("offline rows:", len(df))
print(df.sort_values("row_id").head(5).to_string())

odf = fg.read(online=True)
print("online rows:", len(odf))
print(odf.sort_values("row_id").head(5).to_string())
