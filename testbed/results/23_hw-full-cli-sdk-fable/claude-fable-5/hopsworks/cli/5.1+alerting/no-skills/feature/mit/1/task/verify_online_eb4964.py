import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("featureseb4964", version=1)
print("online_enabled:", fg.online_enabled)
df = fg.read(online=True)
print("online row count:", len(df))
print(df.head(3).to_string())
