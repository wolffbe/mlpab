import hopsworks
proj = hopsworks.login()
fs = proj.get_feature_store()
t = proj.get_trino_api()
schema = fs.offline_featurestore_name if hasattr(fs, 'offline_featurestore_name') else fs.name
print('schema:', schema)
conn = t.connect(catalog='delta', schema=schema, verify=False)
cur = conn.cursor()
q = """
SELECT cast(ts AS date) AS d,
       count(*) AS n,
       avg(prediction) AS mean_pred,
       stddev(prediction) AS std_pred,
       min(prediction) AS min_pred,
       max(prediction) AS max_pred
FROM prediction_log_1
GROUP BY cast(ts AS date)
ORDER BY 1
"""
cur.execute(q)
rows = cur.fetchall()
print('cols:', [c[0] for c in cur.description])
import csv
with open('.tmp/daily.csv', 'w', newline='') as f:
    w = csv.writer(f)
    w.writerow(['d', 'n', 'mean', 'std', 'min', 'max'])
    for r in rows:
        w.writerow([str(r[0]), r[1], round(float(r[2]), 4), round(float(r[3]), 4), round(float(r[4]), 4), round(float(r[5]), 4)])
        print(str(r[0]), r[1], round(float(r[2]), 4), round(float(r[3]), 4))
print('DAYS:', len(rows))
