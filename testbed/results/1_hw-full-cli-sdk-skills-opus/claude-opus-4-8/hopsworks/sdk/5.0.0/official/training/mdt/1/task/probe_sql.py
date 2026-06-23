import hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

print("HAS fs.sql:", hasattr(fs, "sql"))
print([m for m in dir(fs) if not m.startswith("_")])

stg = fs.get_feature_group("scaled1f3dc5_stg", version=1)
print("FG methods:", [m for m in dir(stg) if not m.startswith("_")])

# Try aggregate via fs.sql against offline store
q = ("SELECT count(*) AS n, "
     "avg(CASE WHEN split='train' THEN f1 END) AS m1, "
     "stddev_pop(CASE WHEN split='train' THEN f1 END) AS s1 "
     "FROM scaled1f3dc5_stg_1")
try:
    df = fs.sql(q, dataframe_type="pandas")
    print("fs.sql OK:")
    print(df)
except Exception as e:
    print("fs.sql ERROR:", repr(e))
