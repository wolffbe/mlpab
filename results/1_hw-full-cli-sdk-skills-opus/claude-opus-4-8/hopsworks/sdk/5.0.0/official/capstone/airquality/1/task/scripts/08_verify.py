import warnings
warnings.filterwarnings("ignore")
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

fg = fs.get_feature_group("airqpred963ee7", version=1)
print("FG:", fg.name, "online_enabled:", fg.online_enabled)
df = fg.read()
print("offline rows:", len(df))
print(df.sort_values("date").head())
print(df["pm25_pred"].describe())

# online low-latency lookup check
sample_date = str(df.sort_values("date").iloc[0]["date"])
try:
    vec = fg.get_feature_vector({"date": sample_date})
    print("online lookup for", sample_date, "->", vec)
except Exception as e:
    print("online lookup err:", repr(e))

# model + metrics
mr = project.get_model_registry()
m = mr.get_model("airqmodel963ee7", version=1)
print("MODEL metrics:", m.training_metrics)
