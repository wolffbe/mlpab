import hopsworks

proj = hopsworks.login()
fs = proj.get_feature_store()
scored = fs.get_feature_group("scored26cb88", version=1)

# online read path on the feature group
try:
    odf = scored.read(online=True)
    print("online read rows:", odf.shape)
    print(odf[odf["request_id"] == "Q00000"].to_string())
except Exception as e:
    print("online read err:", type(e).__name__, e)

# init serving and fetch a single low-latency vector via a feature view
try:
    fv = fs.get_or_create_feature_view(
        name="scored26cb88_fv", version=1, query=scored.select_all()
    )
    fv.init_serving()
    vec = fv.get_feature_vector({"request_id": "Q00000"})
    print("online feature vector Q00000:", vec)
except Exception as e:
    print("fv serving err:", type(e).__name__, e)
