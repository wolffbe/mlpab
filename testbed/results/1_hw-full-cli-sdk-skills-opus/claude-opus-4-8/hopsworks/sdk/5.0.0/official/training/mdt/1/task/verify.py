import hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

fg = fs.get_feature_group("scaled1f3dc5", version=1)
print("online_enabled:", fg.online_enabled, "| pk:", fg.primary_key)
print("schema:", [(f.name, f.type) for f in fg.features])

off = fg.read()
print("OFFLINE rows:", len(off), "| cols:", list(off.columns))
print("split counts:\n", off["split"].value_counts())
print(off.sort_values("row_id").head(3).to_string())

# Online low-latency lookup
v_train = fg.get_feature_vector({"row_id": "R00000"})
v_serve = fg.get_feature_vector({"row_id": "R00400"})
print("ONLINE R00000:", v_train)
print("ONLINE R00400:", v_serve)
