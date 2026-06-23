import hopsworks, warnings
warnings.filterwarnings("ignore")
project = hopsworks.login()
fs = project.get_feature_store()
print("Project:", project.name)
try:
    import hsfs.engine as e
    print("engine:", e.get_type())
except Exception as ex:
    print("engine err", ex)
for m in ["sql", "get_or_create_feature_group", "get_feature_group", "create_feature_view"]:
    print("fs has", m, hasattr(fs, m))
try:
    from hopsworks import udf
    print("hopsworks.udf OK")
except Exception as ex:
    print("udf import err", ex)
