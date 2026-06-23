import hopsworks
import pandas as pd

proj = hopsworks.login()
fs = proj.get_feature_store()

fg_train = fs.get_feature_group("skew_train", version=1)
fg_serve = fs.get_feature_group("skew_serve", version=1)

# Join training vs serving on entity_id ON THE PLATFORM (query service)
q = fg_train.select_all().join(
    fg_serve.select_all(), on=["entity_id"], prefix="srv_")
df = q.read()
print("joined shape:", df.shape)
print("columns:", list(df.columns))

feats = ["f1", "f2", "f3", "f4", "f5"]
print("\nmatched rows:", len(df))
print(f"{'feat':>6} {'mean|diff|':>12} {'max|diff|':>12} {'corr':>8} {'mean_t':>10} {'mean_s':>10}")
results = []
for f in feats:
    t = df[f].astype(float)
    s = df["srv_" + f].astype(float)
    d = (t - s).abs()
    corr = t.corr(s)
    results.append((f, d.mean(), d.max(), corr))
    print(f"{f:>6} {d.mean():12.6f} {d.max():12.6f} {corr:8.4f} {t.mean():10.4f} {s.mean():10.4f}")

# Identify the diverging feature: largest mean abs diff / lowest correlation
worst = max(results, key=lambda r: r[1])
print("\nDIVERGING FEATURE (max mean abs diff):", worst[0])
worst_corr = min(results, key=lambda r: (r[3] if r[3]==r[3] else -1))
print("DIVERGING FEATURE (min correlation):", worst_corr[0])

# Print a few example rows for the suspect feature
suspect = worst[0]
print("\nSample rows for", suspect)
print(df[["entity_id", suspect, "srv_"+suspect]].head(12).to_string(index=False))
