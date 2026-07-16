import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("transactions82e347", version=1)
print("name:", fg.name)
print("version:", fg.version)
print("primary_key:", fg.primary_key)
print("event_time:", fg.event_time)
print("online_enabled:", fg.online_enabled)
print("features:", [(f.name, f.type) for f in fg.features])
