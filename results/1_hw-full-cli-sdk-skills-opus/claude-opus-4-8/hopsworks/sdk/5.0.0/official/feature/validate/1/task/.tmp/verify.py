import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("events5b591e", version=1)
print("id:", fg.id)
print("online_enabled:", fg.online_enabled)
print("primary_key:", fg.primary_key)
print("event_time:", fg.event_time)

# offline read count
df = fg.read(dataframe_type="pandas")
print("offline rows:", len(df))

# online lookup check
sample_key = df.iloc[0]["row_id"]
vec = fg.get_feature_vector({"row_id": sample_key})
print("online feature vector for", sample_key, "->", vec)
