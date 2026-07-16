import hopsworks

proj = hopsworks.login()
fs = proj.get_feature_store()

fg = fs.get_feature_group("recsfd473b", version=1)
print("online_enabled:", fg.online_enabled)
print("features:", [f.name for f in fg.features])
print("primary_key:", [f.name for f in fg.features if f.primary])

# Offline read-back
df = fg.read()
print("OFFLINE rows:", len(df), "users:", df["user_id"].nunique())
print("ranks per user (min/max):", df["rank"].min(), df["rank"].max())
print("sample:")
print(df.sort_values(["user_id", "rank"]).head(7).to_string())

# Online / low-latency read-back via a feature view + feature vector lookup
fv = fs.get_or_create_feature_view(
    name="recsfd473b_fv", version=1, query=fg.select_all()
)
vec = fv.get_feature_vector({"rec_id": "U0003#1"}, return_type="pandas")
print("ONLINE feature vector for U0003#1:")
print(vec.to_string())
