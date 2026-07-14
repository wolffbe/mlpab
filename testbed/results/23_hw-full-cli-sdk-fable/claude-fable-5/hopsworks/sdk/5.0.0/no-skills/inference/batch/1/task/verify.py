import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("scores4fa858", version=1)
print("name:", fg.name, "version:", fg.version)
print("online_enabled:", fg.online_enabled)
print("primary_key:", fg.primary_key)
print("features:", [(f.name, f.type) for f in fg.features])

try:
    vec = fg.select_all().read(online=True)
    print("online read rows:", len(vec))
    print(vec.head())
except Exception as e:
    print("online read from external client failed (expected if port blocked):", type(e).__name__, str(e)[:300])
