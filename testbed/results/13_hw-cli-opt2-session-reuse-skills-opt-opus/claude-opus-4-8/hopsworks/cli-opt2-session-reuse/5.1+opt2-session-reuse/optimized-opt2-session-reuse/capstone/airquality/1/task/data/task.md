# Capstone — air-quality PM2.5 forecasting (regression)

`data/airquality_history.csv` is a daily history (`date, pm25_lag1, temperature, humidity, wind_speed, pressure, precipitation, pm25`) where `pm25` is the measured air quality and `pm25_lag1` is the previous day's value. `data/forecast_days.csv` holds later days WITHOUT `pm25` — predict it for each.

Build the full pipeline on the platform:
1. Engineer features (weather + lag/rolling air-quality signals) into a feature group `airq06009f`.
2. Assemble a training dataset `airqtd06009f`.
3. Train a PM2.5 regressor and register it as `airqmodel06009f` WITH its evaluation metrics.
4. Predict every row of `forecast_days.csv` into a feature table `airqpred06009f` (record key `date`, column `pm25_pred`).

Target: RMSE ≤ 1.803 µg/m³ on the held-out days (train-mean baseline ≈ 2.366).
