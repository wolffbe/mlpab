import os
import joblib
import pandas as pd
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

# Download model
model_meta = mr.get_model("airqmodel3c8c0c", version=1)
model_dir = model_meta.download()
clf = joblib.load(os.path.join(model_dir, "model.pkl"))
print("Model loaded")

# Download forecast data from HopsFS
dataset_api = project.get_dataset_api()
dataset_api.download("Resources/forecast_days.csv", local_path="forecast_days.csv", overwrite=True)

df_forecast = pd.read_csv("forecast_days.csv")
print(f"Forecast rows: {len(df_forecast)}")
print(df_forecast.head())

feature_cols = ["pm25_lag1", "temperature", "humidity", "wind_speed", "pressure", "precipitation"]
X_forecast = df_forecast[feature_cols]

predictions = clf.predict(X_forecast)
print(f"Predictions: {predictions}")

df_pred = pd.DataFrame({
    "date": df_forecast["date"].astype(str),
    "pm25_pred": predictions.tolist(),
})
print(df_pred)

# Create predictions feature group with online store enabled
pred_fg = fs.get_or_create_feature_group(
    name="airqpred3c8c0c",
    version=1,
    primary_key=["date"],
    online_enabled=True,
    description="PM2.5 predictions",
)
print("Got/created airqpred3c8c0c FG")

pred_fg.insert(df_pred)
print("Predictions inserted into airqpred3c8c0c")
