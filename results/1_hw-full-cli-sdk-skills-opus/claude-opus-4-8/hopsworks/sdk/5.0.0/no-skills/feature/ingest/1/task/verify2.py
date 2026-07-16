import hopsworks

proj = hopsworks.login()
fs = proj.get_feature_store()
fg = fs.get_feature_group("transactions3cd0a6", version=1)

print("online_enabled:", fg.online_enabled)
print("primary_key:", fg.primary_key)
print("event_time:", fg.event_time)

df = fg.read()
print("OFFLINE rows:", len(df))
print("OFFLINE unique row_id:", df["row_id"].nunique())

# Online / low-latency read-back via the FeatureView serving path.
sample_key = str(df["row_id"].iloc[0])
try:
    online_df = fg.read(online=True)
    print("ONLINE rows:", len(online_df))
except Exception as e:
    print("Online bulk read error:", e)
