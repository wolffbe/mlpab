import warnings
warnings.filterwarnings("ignore")
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("scores30c485", version=1)
print("online_enabled:", fg.online_enabled)
print("primary_key:", fg.primary_key)
print("features:", [f.name for f in fg.features])

off = fg.read(dataframe_type="pandas")
print("OFFLINE rows:", len(off), "cols:", list(off.columns), "distinct:", off.account_id.nunique())
print(off.sort_values("account_id").head(5).to_string())

on = fg.read(online=True, dataframe_type="pandas")
print("ONLINE rows:", len(on), "cols:", list(on.columns))
print(on.sort_values("account_id").head(5).to_string())
