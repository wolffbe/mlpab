import json, hopsworks
project = hopsworks.login()
schema = f"{project.name.lower()}_featurestore"
conn = project.get_trino_api().connect(catalog="delta", schema=schema, verify=False)
cur = conn.cursor()
cur.execute(
    "SELECT label, count(*) n, avg(f3) mean_f3, stddev(f3) sd_f3, "
    "min(f3) min_f3, max(f3) max_f3 "
    f"FROM delta.{schema}.leakage_probe_1 GROUP BY label ORDER BY label"
)
cols = [d[0] for d in cur.description]
rows = [dict(zip(cols, r)) for r in cur.fetchall()]
for r in rows:
    print(r)
json.dump(rows, open(".tmp/sep.json", "w"), indent=2, default=str)
