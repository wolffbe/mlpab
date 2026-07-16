import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("eventsee881b", 1)
print("name:", fg.name, "version:", fg.version, "id:", fg.id)
print("online_enabled:", fg.online_enabled)
print("primary_key:", fg.primary_key)
print("event_time:", fg.event_time)
print("features:", [(f.name, f.type) for f in fg.features])
