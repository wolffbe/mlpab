import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("eventsee881b", version=1)

offline = fg.read()
print(f"offline rows: {len(offline)}")
print(offline.head(3).to_string())

online = fg.read(online=True)
print(f"online rows: {len(online)}")
print(online.head(3).to_string())
