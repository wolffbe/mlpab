# Capstone — air-quality PM2.5 forecasting (regression)

`data/airquality_history.csv` is a daily history (`date, pm25_lag1, temperature, humidity, wind_speed, pressure, precipitation, pm25`) where `pm25` is the measured air quality and `pm25_lag1` is the previous day's value. `data/forecast_days.csv` holds later days WITHOUT `pm25` — predict it for each.

Build the full pipeline on the platform:
1. Engineer features (weather + lag/rolling air-quality signals) into a feature group `airqfcd91b`.
2. Assemble a training dataset `airqtdfcd91b`.
3. Train a PM2.5 regressor and register it as `airqmodelfcd91b` WITH its evaluation metrics.
4. Predict every row of `forecast_days.csv` into a feature table `airqpredfcd91b` (record key `date`, column `pm25_pred`).

Target: RMSE ≤ 1.998 µg/m³ on the held-out days (train-mean baseline ≈ 2.393).
