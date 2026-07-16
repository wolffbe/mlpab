"""FTI training + inference job. Runs ON the Hopsworks platform (pandas-training-pipeline env)."""
import os, warnings
warnings.filterwarnings('ignore')
import numpy as np
import pandas as pd
import hopsworks

proj = hopsworks.login()
fs = proj.get_feature_store()
mr = proj.get_model_registry()

FEATURES = ['pm25_lag1', 'temperature', 'humidity', 'wind_speed', 'pressure', 'precipitation']

# ---- 1. Read training data from the feature view ----
fv = fs.get_feature_view('airqtd963ee7', version=1)
X_train, X_test, y_train, y_test = fv.train_test_split(test_size=0.2)
for df in (X_train, X_test):
    if 'date' in df.columns:
        df.drop(columns=['date'], inplace=True)
X_train = X_train[FEATURES]
X_test = X_test[FEATURES]
y_train = np.ravel(y_train.values)
y_test = np.ravel(y_test.values)
print('TRAIN', X_train.shape, 'TEST', X_test.shape)

# ---- 2. Train regressor ----
from sklearn.ensemble import GradientBoostingRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error

model = GradientBoostingRegressor(n_estimators=500, max_depth=3,
                                  learning_rate=0.05, subsample=0.9,
                                  random_state=42)
model.fit(X_train, y_train)

pred_test = model.predict(X_test)
rmse = float(np.sqrt(mean_squared_error(y_test, pred_test)))
mae = float(mean_absolute_error(y_test, pred_test))
r2 = float(r2_score(y_test, pred_test))
print('HELDOUT_RMSE', rmse, 'MAE', mae, 'R2', r2)

# ---- 3. Register model with metrics ----
import joblib
model_dir = 'airqmodel963ee7_dir'
os.makedirs(model_dir, exist_ok=True)
joblib.dump(model, os.path.join(model_dir, 'model.pkl'))

metrics = {'rmse': rmse, 'mae': mae, 'r2': r2}
input_example = X_train.head(3)
hops_model = mr.python.create_model(
    name='airqmodel963ee7',
    metrics=metrics,
    description='PM2.5 daily regressor (GradientBoosting)',
    input_example=input_example,
    feature_view=fv,
)
hops_model.save(model_dir)
print('MODEL_REGISTERED', hops_model.name, hops_model.version, metrics)

# ---- 4. Predict forecast days into airqpred963ee7 ----
ds = proj.get_dataset_api()
local_csv = 'forecast_days.csv'
if not os.path.exists(local_csv):
    ds.download('Resources/airq_job/forecast_days.csv', local_csv, overwrite=True)
fdf = pd.read_csv(local_csv)
Xf = fdf[FEATURES]
preds = model.predict(Xf)
out = pd.DataFrame({'date': pd.to_datetime(fdf['date']), 'pm25_pred': preds.astype(float)})
print('FORECAST_ROWS', out.shape)

pred_fg = fs.get_or_create_feature_group(
    name='airqpred963ee7', version=1,
    description='PM2.5 predictions for forecast days',
    primary_key=['date'], event_time='date', online_enabled=True)
pred_fg.insert(out, write_options={'wait_for_job': True})
print('PRED_FG_INSERTED rows', len(out))
print('JOB_DONE rmse=%.4f' % rmse)
