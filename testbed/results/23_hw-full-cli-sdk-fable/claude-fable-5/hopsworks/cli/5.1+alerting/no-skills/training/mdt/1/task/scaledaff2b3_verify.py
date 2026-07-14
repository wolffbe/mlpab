"""Verify FG scaledaff2b3 v1: offline (hive path) and online reads."""
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("scaledaff2b3", 1)

odf = fg.read(online=True)
print("online rows:", len(odf))
print("columns:", list(odf.columns))
print(odf.sort_values("row_id").head(3).to_string())
print(odf.groupby("split").size().to_string())

try:
    df = fg.read(read_options={"use_hive": True})
    print("offline shape:", df.shape)
    tr = df[df["split"] == "train"]
    for c in ["f1", "f2", "f3", "f4"]:
        print(c, "train mean=%.6f std(pop)=%.6f" % (tr[c].mean(), tr[c].std(ddof=0)))
    print(df.sort_values("row_id").head(3).to_string())
except Exception as e:
    print("offline hive read failed:", e)
