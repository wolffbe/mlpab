"""Verify job: read predictions646af0 v1 from offline and online stores."""
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("predictions646af0", version=1)

offline_df = fg.read()
print("OFFLINE rows:", len(offline_df))
print(offline_df.sort_values("row_id").head(3).to_string(index=False))

online_df = fg.read(online=True)
print("ONLINE rows:", len(online_df))
print(online_df.sort_values("row_id").head(3).to_string(index=False))
