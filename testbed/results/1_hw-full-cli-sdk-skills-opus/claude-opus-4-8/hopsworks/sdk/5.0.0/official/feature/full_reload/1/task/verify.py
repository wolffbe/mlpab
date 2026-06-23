import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

fg2 = fs.get_feature_group("customersa8deb9", version=2)
print("online_enabled:", fg2.online_enabled)
print("primary_key:", fg2.primary_key)
print("event_time:", fg2.event_time)
print("schema:", [(f.name, f.type) for f in fg2.features])

off = fg2.read(dataframe_type="pandas")
print("OFFLINE rows:", len(off), "cols:", list(off.columns))
print("OFFLINE unique row_ids:", off["row_id"].nunique())

on = fg2.read(online=True, dataframe_type="pandas")
print("ONLINE rows:", len(on), "cols:", list(on.columns))
print(off.sort_values("row_id").head(3).to_string())
