import hopsworks

proj = hopsworks.login()
fs = proj.get_feature_store()
fg = fs.get_feature_group("transactions3cd0a6", version=1)

print("online_enabled:", fg.online_enabled)
print("primary_key:", fg.primary_key)
print("event_time:", fg.event_time)

# Offline read-back
df = fg.read()
print("OFFLINE rows:", len(df))
print("OFFLINE unique row_id:", df["row_id"].nunique())

# Online read-back for a sample key
try:
    sample_key = str(df["row_id"].iloc[0])
    ov = fg.get_feature_vector({"row_id": sample_key})
    print("ONLINE lookup for", sample_key, "->", ov)
except Exception as e:
    print("Online lookup error:", e)
