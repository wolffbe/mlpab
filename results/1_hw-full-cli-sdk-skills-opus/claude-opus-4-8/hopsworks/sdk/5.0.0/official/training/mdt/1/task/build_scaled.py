import pandas as pd
import hopsworks
from hsfs.feature import Feature

project = hopsworks.login()
fs = project.get_feature_store()
schema = fs.offline_featurestore_name
tbl = f"delta.{schema}.scaled1f3dc5_stg_1"

# --- Standardization runs PLATFORM-SIDE in Trino ---
# mean & population std (stddev_pop, no Bessel) over TRAIN rows only, applied to BOTH splits.
sql = f"""
WITH stats AS (
  SELECT avg(f1) m1, stddev_pop(f1) s1,
         avg(f2) m2, stddev_pop(f2) s2,
         avg(f3) m3, stddev_pop(f3) s3,
         avg(f4) m4, stddev_pop(f4) s4
  FROM {tbl}
  WHERE split = 'train'
)
SELECT t.row_id AS row_id,
       t.split  AS split,
       round((t.f1 - s.m1) / s.s1, 6) AS f1,
       round((t.f2 - s.m2) / s.s2, 6) AS f2,
       round((t.f3 - s.m3) / s.s3, 6) AS f3,
       round((t.f4 - s.m4) / s.s4, 6) AS f4
FROM {tbl} t CROSS JOIN stats s
"""

trino_api = project.get_trino_api()
conn = trino_api.connect(catalog="delta", schema=schema, verify=False)
cur = conn.cursor()
cur.execute(sql)
rows = cur.fetchall()
cols = [d[0] for d in cur.description]
df = pd.DataFrame(rows, columns=cols)
df = df[["row_id", "split", "f1", "f2", "f3", "f4"]]
for c in ["f1", "f2", "f3", "f4"]:
    df[c] = df[c].astype("float64")
df["row_id"] = df["row_id"].astype(str)
df["split"] = df["split"].astype(str)
print("standardized rows:", len(df))
print(df.head())
print("split counts:\n", df["split"].value_counts())

stg = fs.get_feature_group("scaled1f3dc5_stg", version=1)

scaled = fs.get_or_create_feature_group(
    name="scaled1f3dc5",
    version=1,
    description="f1-f4 standardized (x-mean)/std using TRAIN-only mean & population std, both splits; online-enabled.",
    primary_key=["row_id"],
    online_enabled=True,
    stream=True,
    parents=[stg],
    features=[
        Feature("row_id", "string", description="Unique row id across both splits"),
        Feature("split", "string", description="'train' or 'serve' source split"),
        Feature("f1", "double", description="standardized f1 (train mean/pop-std)"),
        Feature("f2", "double", description="standardized f2 (train mean/pop-std)"),
        Feature("f3", "double", description="standardized f3 (train mean/pop-std)"),
        Feature("f4", "double", description="standardized f4 (train mean/pop-std)"),
    ],
    statistics_config=False,
)

scaled.insert(df, wait=True)
print("INSERT DONE; fg.id =", scaled.id, "online_enabled =", scaled.online_enabled)
