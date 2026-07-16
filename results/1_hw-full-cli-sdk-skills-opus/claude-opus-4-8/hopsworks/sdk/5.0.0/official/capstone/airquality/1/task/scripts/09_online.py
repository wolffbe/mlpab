import warnings
warnings.filterwarnings("ignore")
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("airqpred963ee7", version=1)
online_df = fg.read(online=True)
print("ONLINE rows:", len(online_df))
print(online_df.head())
