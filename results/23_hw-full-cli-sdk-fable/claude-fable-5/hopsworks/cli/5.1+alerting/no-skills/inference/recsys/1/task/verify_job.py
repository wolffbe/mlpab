"""Read back recs48963e v1 through platform read paths (offline + online)."""
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("recs48963e", version=1)

df = fg.read()
df = df.sort_values("rec_id").reset_index(drop=True)
print("OFFLINE rows:", len(df))
print("columns:", list(df.columns))
print("distinct users:", df["user_id"].nunique())
print("rank dtype:", df["rank"].dtype, "min:", df["rank"].min(), "max:", df["rank"].max())
print(df.head(10).to_string())

online = fg.read(online=True)
print("ONLINE rows:", len(online))
sample = online[online["rec_id"] == "U0003#1"]
print("online lookup U0003#1:")
print(sample.to_string())
