import json
import hopsworks

project = hopsworks.login()
schema = f"{project.name.lower()}_featurestore"
api = project.get_trino_api()

feats = ["f1", "f2", "f3", "f4", "f5", "f6"]

def run(catalog):
    conn = api.connect(catalog=catalog, schema=schema, verify=False)
    cur = conn.cursor()
    table = "leakage_probe_1"
    # Pearson correlation of each feature with the binary label, computed in Trino.
    sel = ", ".join(
        [f"corr(CAST({f} AS double), CAST(label AS double)) AS c_{f}" for f in feats]
    )
    sql = f"SELECT count(*) AS n, {sel} FROM {catalog}.{schema}.{table}"
    cur.execute(sql)
    cols = [d[0] for d in cur.description]
    row = cur.fetchone()
    return dict(zip(cols, row))

res = None
for cat in ["delta", "hudi", "iceberg"]:
    try:
        res = run(cat)
        print("OK catalog:", cat)
        break
    except Exception as e:
        print(f"catalog {cat} failed:", repr(e)[:200])

if res is None:
    raise SystemExit("all catalogs failed")

print("n =", res["n"])
corrs = {f: res[f"c_{f}"] for f in feats}
for f in feats:
    print(f"  corr({f}, label) = {corrs[f]:.6f}   |abs|={abs(corrs[f]):.6f}")

leaker = max(feats, key=lambda f: abs(corrs[f]))
print("LEAKER:", leaker)

with open(".tmp/corr_result.json", "w") as fp:
    json.dump({"n": res["n"], "correlations": {f: corrs[f] for f in feats}, "leaker": leaker}, fp, indent=2, default=str)
print("wrote .tmp/corr_result.json")
