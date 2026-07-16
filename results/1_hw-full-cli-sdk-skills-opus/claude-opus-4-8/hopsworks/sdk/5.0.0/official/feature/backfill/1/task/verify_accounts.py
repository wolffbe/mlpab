import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("accounts65d53c", version=1)

print("name:", fg.name, "version:", fg.version, "id:", fg.id)
print("online_enabled:", fg.online_enabled)
print("primary_key:", fg.primary_key)
print("event_time:", fg.event_time)
print("schema:", [(f.name, f.type, f.primary) for f in fg.features])

# Offline contents
off = fg.read()
print("offline rows:", len(off), "unique row_id:", off["row_id"].nunique())

# Online lookup sanity check (low-latency path)
sample = off.iloc[0]["row_id"]
vec = fg.read(online=True)
print("online rows:", len(vec), "unique row_id:", vec["row_id"].nunique())
