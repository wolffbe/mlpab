import warnings, os, json
warnings.filterwarnings("ignore")
import numpy as np
import pandas as pd
import joblib
import hopsworks
from hsfs.feature import Feature

FEATURES = ["pm25_lag1", "temperature", "humidity", "wind_speed", "pressure", "precipitation"]

project = hopsworks.login()
fs = project.get_feature_store()

# load registered model
mr = project.get_model_registry()
model = mr.get_model("airqmodel963ee7", version=1)
mdir = model.download()
mdl = joblib.load(os.path.join(mdir, "model.pkl"))
print("loaded model from", mdir)

# read forecast days from the forecast-input feature group (platform-native)
fc_fg = fs.get_feature_group("airqforecast_in963ee7", version=1)
fc = fc_fg.read()
print("forecast rows:", len(fc))

fc_dates = fc["date"].astype(str).values
Xf = fc[FEATURES].astype("float64")
pm25_pred = mdl.predict(Xf).astype("float64")
pred_df = pd.DataFrame({"date": fc_dates, "pm25_pred": pm25_pred})
print(pred_df.head())

pred_fg = fs.get_or_create_feature_group(
    name="airqpred963ee7",
    version=1,
    description="Predicted PM2.5 for forecast days",
    primary_key=["date"],
    online_enabled=True,
    stream=True,
    features=[
        Feature("date", "string", description="Forecast day (record key)"),
        Feature("pm25_pred", "double", description="Predicted PM2.5"),
    ],
)
pred_fg.insert(pred_df, wait=True)
print("wrote", len(pred_df), "predictions into airqpred963ee7")
print("DONE")
