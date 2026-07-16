import google.cloud.aiplatform as aiplatform
print("version", aiplatform.__version__)

# Try to find the BigQuery-backed feature store (FeatureGroup / FeatureOnlineStore / FeatureView)
candidates = []
for modpath in [
    "google.cloud.aiplatform.preview",
    "google.cloud.aiplatform.preview.feature_store",
    "google.cloud.aiplatform.featurestore",
]:
    try:
        mod = __import__(modpath, fromlist=["*"])
        names = [x for x in dir(mod) if not x.startswith("_")]
        print(modpath, "->", names)
    except Exception as e:
        print(modpath, "ERR", repr(e))
