import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("recs48963e", 1)
print("online_enabled:", fg.online_enabled)
online = fg.select_all().read(online=True)
print("online rows:", len(online))
print("unique users:", online["user_id"].nunique())
print("ranks:", sorted(online["rank"].unique()))
print(online[online["user_id"] == "U0003"].sort_values("rank").to_string())
print(online[online["user_id"] == "U0000"].sort_values("rank").to_string())
