import warnings
warnings.filterwarnings("ignore")
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

req_fg = fs.get_feature_group("requests_raw", version=1)
prof_fg = fs.get_feature_group("profiles_raw", version=1)

query = req_fg.select(["request_id", "account_id", "request_lat", "request_lon"]).join(
    prof_fg.select(["account_id", "home_lat", "home_lon", "base_score"]),
    on=["account_id"],
)
joined = query.read()
print("columns:", list(joined.columns))
print("shape:", joined.shape)
print(joined.head(3).to_dict("records"))
