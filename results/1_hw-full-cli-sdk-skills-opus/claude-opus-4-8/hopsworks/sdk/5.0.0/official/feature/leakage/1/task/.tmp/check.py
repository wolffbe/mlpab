import hopsworks
project = hopsworks.login()
fs = project.get_feature_store()
fg = fs.get_feature_group("leakage_probe", version=1)
print("FG found, id=", fg.id)
try:
    df = fg.read()
    print("read shape", df.shape)
    print(df.head().to_string())
except Exception as e:
    print("read err:", repr(e)[:400])
