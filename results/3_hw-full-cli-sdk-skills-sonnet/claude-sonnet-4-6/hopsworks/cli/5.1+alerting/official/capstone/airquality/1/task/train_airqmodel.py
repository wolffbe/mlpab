import os
import math
import joblib
import numpy as np
import hopsworks
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

project = hopsworks.login()
fs = project.get_feature_store()
mr = project.get_model_registry()

fv = fs.get_feature_view("airqtd3c8c0c", version=1)

print("Creating training dataset...")
X, y = fv.training_data(
    description="PM2.5 air quality training data",
    data_format="csv",
    write_options={"wait_for_job": True},
    coalesce=True,
)
print(f"Total samples: {len(X)}, Features: {X.columns.tolist()}")

# Drop non-numeric columns (e.g. date) before training
drop_cols = [c for c in X.columns if X[c].dtype == object]
print(f"Dropping columns: {drop_cols}")
X = X.drop(columns=drop_cols)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Train: {len(X_train)}, Test: {len(X_test)}")

model = GradientBoostingRegressor(
    n_estimators=300,
    max_depth=4,
    learning_rate=0.05,
    subsample=0.8,
    min_samples_leaf=3,
    random_state=42,
)
model.fit(X_train, y_train.values.ravel())

y_pred = model.predict(X_test)
rmse = math.sqrt(mean_squared_error(y_test.values.ravel(), y_pred))
mae = float(np.mean(np.abs(y_test.values.ravel() - y_pred)))
r2 = float(model.score(X_test, y_test.values.ravel()))

print(f"RMSE: {rmse:.4f}")
print(f"MAE:  {mae:.4f}")
print(f"R2:   {r2:.4f}")

model_dir = "airqmodel3c8c0c"
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, f"{model_dir}/model.pkl")

sklearn_model = mr.sklearn.create_model(
    name="airqmodel3c8c0c",
    metrics={"rmse": rmse, "mae": mae, "r2": r2},
    description="PM2.5 air quality regressor (GradientBoosting)",
    feature_view=fv,
)
sklearn_model.save(model_dir)
print("Model registered successfully as airqmodel3c8c0c")
