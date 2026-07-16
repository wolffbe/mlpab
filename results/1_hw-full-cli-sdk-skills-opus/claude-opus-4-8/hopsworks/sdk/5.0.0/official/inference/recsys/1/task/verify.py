import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("recsfd473b", version=1)

print("online_enabled:", fg.online_enabled)
print("primary_key:", fg.primary_key)
print("schema:", [(f.name, f.type, f.primary) for f in fg.features])

# Offline read via platform query engine (ArrowFlight)
df = fg.read()
print("offline rows:", len(df))
print("distinct users:", df["user_id"].nunique())
print("rank set:", sorted(df["rank"].unique().tolist()))
g = df.groupby("user_id")["rank"].count()
print("rows-per-user min/max:", int(g.min()), int(g.max()))
samp = df[df.user_id == "U0000"].sort_values("rank")
print(samp[["rec_id", "user_id", "rank", "item_id"]].to_string(index=False))

# Online / low-latency lookup
fg.init_serving() if hasattr(fg, "init_serving") else None
try:
    vec = fg.get_feature_vector({"rec_id": "U0003#1"})
    print("ONLINE get_feature_vector U0003#1:", vec)
except Exception as e:
    print("get_feature_vector err:", repr(e)[:200])
    odf = fg.read(online=True)
    print("ONLINE read rows:", len(odf))
    print(odf[odf.rec_id == "U0003#1"].to_string(index=False))
