"""

project = hopsworks.login()
fs = project.get_feature_store()

df = pd.read_csv(io.StringIO(CSV_DATA))
df["account_id"] = df["account_id"].astype(str)
df["event_time"] = df["event_time"].astype("int64")
print("total rows:", len(df), "accounts:", df["account_id"].nunique())

valid = df[df["event_time"] <= T]
print("rows at/before T:", len(valid), "accounts at/before T:", valid["account_id"].nunique())

latest = (
    valid.sort_values(["account_id", "event_time"])
    .groupby("account_id", as_index=False)
    .tail(1)
)

z = (
    WEIGHTS["f1"] * latest["f1"]
    + WEIGHTS["f2"] * latest["f2"]
    + WEIGHTS["f3"] * latest["f3"]
    + BIAS
)
scores = [round(1.0 / (1.0 + math.exp(-v)), 6) for v in z]

out = (
    pd.DataFrame({"account_id": latest["account_id"].values, "score": scores})
    .sort_values("account_id")
    .reset_index(drop=True)
)
print(out.head(10))
print("output rows:", len(out))

fg = fs.get_or_create_feature_group(
    name="scores4fa858",
    version=1,
    primary_key=["account_id"],
    online_enabled=True,
    description="Batch scores as of T=1773565200000 (point-in-time correct)",
)
fg.insert(out, wait=True)
print("INSERT DONE:", len(out), "rows into scores4fa858 v1 (online_enabled=True)")

readback = fg.read()
print("offline readback rows:", len(readback))
print(readback.sort_values("account_id").head(5))
