import warnings
warnings.filterwarnings("ignore")
import json
import hopsworks

T = 1773410400000
project = hopsworks.login()
trino = project.get_trino_api()
conn = trino.connect(catalog="delta", schema="mlpabb8bef2_featurestore", verify=False)
cur = conn.cursor()

sql = """
SELECT account_id,
       round(1.0/(1.0+exp(-(1.1161*f1 + 0.6773*f2 + 0.155*f3 + 0.2799))), 6) AS score
FROM (
  SELECT account_id, f1, f2, f3,
         row_number() OVER (PARTITION BY account_id ORDER BY event_time DESC) AS rn
  FROM delta.mlpabb8bef2_featurestore.account_feature_history_1
  WHERE event_time <= %d
) t
WHERE rn = 1
ORDER BY account_id
""" % T

cur.execute(sql)
rows = cur.fetchall()
print("rows:", len(rows))
for r in rows[:8]:
    print(r)
print("distinct:", len(set(r[0] for r in rows)))
with open(".tmp_trino_scores.json", "w") as f:
    json.dump(rows, f)
print("saved")
