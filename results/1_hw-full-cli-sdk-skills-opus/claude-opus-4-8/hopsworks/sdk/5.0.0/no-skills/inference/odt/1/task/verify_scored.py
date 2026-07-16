import hopsworks

proj = hopsworks.login()
fs = proj.get_feature_store()
scored = fs.get_feature_group("scored26cb88", version=1)

print("online_enabled:", scored.online_enabled)
print("primary_key:", scored.primary_key)
print("columns:", [f.name for f in scored.features])

# offline read
df = scored.read()
print("offline rows:", df.shape, "unique request_id:", df["request_id"].nunique())
print(df.sort_values("request_id").head().to_string())

# expected check from raw sources (compute via formula to validate correctness)
import math
req = {r["request_id"]: r for _, r in __import__("pandas").read_csv("data/requests.csv").iterrows()}
prof = {p["account_id"]: p for _, p in __import__("pandas").read_csv("data/profiles.csv").iterrows()}
bad = 0
for _, row in df.iterrows():
    r = req[row["request_id"]]
    p = prof[r["account_id"]]
    exp_d = round(math.sqrt((r["request_lat"] - p["home_lat"]) ** 2 + (r["request_lon"] - p["home_lon"]) ** 2), 6)
    exp_s = round(p["base_score"] - 0.1 * exp_d, 6)
    if abs(row["distance_deg"] - exp_d) > 1e-6 or abs(row["score"] - exp_s) > 1e-6:
        bad += 1
        if bad <= 5:
            print("MISMATCH", row["request_id"], row["distance_deg"], exp_d, row["score"], exp_s)
print("mismatches:", bad, "of", len(df))

# online read path
fv = scored.get_feature_vector({"request_id": "Q00000"})
print("online vector Q00000:", fv)
