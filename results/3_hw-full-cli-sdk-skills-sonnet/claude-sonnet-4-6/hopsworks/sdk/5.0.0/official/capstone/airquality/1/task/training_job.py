"""Training job — runs on Hopsworks platform."""
import hopsworks
import pandas as pd
import numpy as np
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error
import joblib
import json
import os

project = hopsworks.login()
fs = project.get_feature_store()

fv = fs.get_feature_view(name="airqtd3c8c0c", version=1)

X_train, X_test, y_train, y_test = fv.train_test_split(test_size=0.2)

# Drop primary key — string date is not a numeric feature
X_train = X_train.drop(columns=["date"], errors="ignore")
X_test  = X_test.drop(columns=["date"], errors="ignore")
feature_cols = list(X_train.columns)
print("Feature cols:", feature_cols)
print("Train:", X_train.shape, "Test:", X_test.shape)

model = GradientBoostingRegressor(
    n_estimators=400,
    learning_rate=0.05,
    max_depth=4,
    subsample=0.8,
    random_state=42,
)
model.fit(X_train, y_train)

y_pred = model.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, y_pred)))
mae  = float(mean_absolute_error(y_test, y_pred))
print(f"RMSE={rmse:.4f}  MAE={mae:.4f}")

os.makedirs("model_dir", exist_ok=True)
joblib.dump(model, "model_dir/model.pkl")
json.dump(feature_cols, open("model_dir/feature_cols.json", "w"))

mr = project.get_model_registry()
hw_model = mr.sklearn.create_model(
    name="airqmodel3c8c0c",
    version=1,
    metrics={"rmse": rmse, "mae": mae},
    description="PM2.5 GradientBoosting regressor",
    input_example=X_train.head(3),
    feature_view=fv,
)
hw_model.save("model_dir")
print("Model registered:", hw_model.name, "v", hw_model.version)
