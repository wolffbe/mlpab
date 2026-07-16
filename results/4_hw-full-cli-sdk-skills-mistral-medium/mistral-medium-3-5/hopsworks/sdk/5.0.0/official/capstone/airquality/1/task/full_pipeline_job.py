import hopsworks
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import json
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    print("=" * 60)
    print("Full Air Quality FTI Pipeline - Running on Platform")
    print("=" * 60)

    # Connect to Hopsworks
    print("\n1. Connecting to Hopsworks...")
    project = hopsworks.login()
    fs = project.get_feature_store()

    # Read the data files from Resources
    print("\n2. Reading data files...")
    history_df = pd.read_csv('Resources/airquality_history.csv')
    forecast_df = pd.read_csv('Resources/forecast_days.csv')

    print(f"   History: {history_df.shape[0]} rows, {history_df.shape[1]} columns")
    print(f"   Forecast: {forecast_df.shape[0]} rows, {forecast_df.shape[1]} columns")

    # Feature Engineering
    print("\n3. Feature Engineering...")

    # Convert date to datetime
    history_df['date'] = pd.to_datetime(history_df['date'])
    forecast_df['date'] = pd.to_datetime(forecast_df['date'])

    # Sort by date
    history_df = history_df.sort_values('date')

    # Create rolling statistics
    history_df['pm25_rolling_7d_mean'] = history_df['pm25'].rolling(window=7, min_periods=1).mean()
    history_df['pm25_rolling_7d_std'] = history_df['pm25'].rolling(window=7, min_periods=1).std()
    history_df['temp_rolling_7d_mean'] = history_df['temperature'].rolling(window=7, min_periods=1).mean()
    history_df['humidity_rolling_7d_mean'] = history_df['humidity'].rolling(window=7, min_periods=1).mean()

    # Create lag features
    history_df['pm25_lag2'] = history_df['pm25'].shift(1)
    history_df['pm25_lag3'] = history_df['pm25'].shift(2)

    # Add time-based features
    history_df['day_of_week'] = history_df['date'].dt.dayofweek
    history_df['day_of_month'] = history_df['date'].dt.day
    history_df['month'] = history_df['date'].dt.month
    history_df['year'] = history_df['date'].dt.year

    # Interaction features
    history_df['temp_humidity'] = history_df['temperature'] * history_df['humidity']
    history_df['pressure_temp'] = history_df['pressure'] * history_df['temperature']

    # Fill NaN values
    history_df = history_df.ffill().bfill()

    print(f"   Engineered features: {history_df.shape[1]} columns")

    # Create feature group
    fg_name = 'airq2ce555'
    print(f"\n4. Creating feature group: {fg_name}")

    # Get or create feature group
    fg = fs.get_or_create_feature_group(
        name=fg_name,
        version=1,
        description='Air quality and weather features for PM2.5 forecasting',
        primary_key=['date'],
        event_time='date'
    )
    print(f"   Feature group {fg_name} ready")

    # Select columns to insert
    feature_columns = [
        'date', 'pm25_lag1', 'temperature', 'humidity', 'wind_speed', 
        'pressure', 'precipitation', 'pm25',
        'pm25_rolling_7d_mean', 'pm25_rolling_7d_std', 
        'temp_rolling_7d_mean', 'humidity_rolling_7d_mean',
        'pm25_lag2', 'pm25_lag3',
        'day_of_week', 'day_of_month', 'month', 'year',
        'temp_humidity', 'pressure_temp'
    ]

    history_to_insert = history_df[feature_columns].copy()

    print(f"   Inserting {len(history_to_insert)} rows...")
    fg.insert(history_to_insert, write_options={'wait_for_job': True})
    print(f"   Data inserted into feature group {fg_name}")

    # Create training dataset (feature view)
    dataset_name = 'airqtd2ce555'
    print(f"\n5. Creating training dataset: {dataset_name}")

    # Get or create feature view
    fv = fs.get_or_create_feature_view(
        name=dataset_name,
        version=1,
        query=fg.select_all(),
        description='Training dataset for PM2.5 forecasting',
        labels=['pm25']
    )
    print(f"   Feature view {dataset_name} ready")

    # Get all data from feature view for training
    print("\n6. Training model...")
    all_data, _ = fv.get_batch_data()

    # Sort by date
    all_data = all_data.sort_values('date')

    # Split into train and validation (last 30 days for validation)
    train_end_date = all_data['date'].max() - pd.Timedelta(days=30)
    train_data = all_data[all_data['date'] <= train_end_date]
    val_data = all_data[all_data['date'] > train_end_date]

    print(f"   Train size: {len(train_data)}, Validation size: {len(val_data)}")

    # Prepare features and labels
    feature_columns_list = [col for col in train_data.columns if col not in ['date', 'pm25']]
    X_train = train_data[feature_columns_list]
    y_train = train_data['pm25']
    X_val = val_data[feature_columns_list]
    y_val = val_data['pm25']

    print(f"   Features: {len(feature_columns_list)}")

    # Train model
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )

    model.fit(X_train, y_train)

    # Predict and calculate metrics
    print("\n7. Evaluating model...")
    y_pred_train = model.predict(X_train)
    y_pred_val = model.predict(X_val)

    rmse_train = np.sqrt(mean_squared_error(y_train, y_pred_train))
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    r2_train = r2_score(y_train, y_pred_train)
    r2_val = r2_score(y_val, y_pred_val)

    print(f'   Training RMSE: {rmse_train:.4f}')
    print(f'   Validation RMSE: {rmse_val:.4f}')
    print(f'   Training R2: {r2_train:.4f}')
    print(f'   Validation R2: {r2_val:.4f}')

    # Register model
    print("\n8. Registering model...")
    mr = project.get_model_registry()

    input_example = X_train.iloc[0:1].to_dict('records')[0]
    output_example = {'pm25_pred': float(y_pred_val[0])}

    metrics = {
        'rmse': float(rmse_val),
        'r2': float(r2_val),
        'mae': float(np.mean(np.abs(y_pred_val - y_val))),
        'rmse_train': float(rmse_train),
        'r2_train': float(r2_train)
    }

    model_reg = mr.python.create_model(
        name='airqmodel2ce555',
        metrics=metrics,
        description='PM2.5 forecasting model trained on air quality and weather data',
        input_example=input_example,
        output_example=output_example,
        model_file='/tmp/model.pkl',
        framework='scikit-learn'
    )

    joblib.dump(model, '/tmp/model.pkl')
    model_reg.save('/tmp/model.pkl')
    print("   Model airqmodel2ce555 registered successfully")
    print(f"   Validation RMSE: {rmse_val:.4f}")

    # Now predict on forecast days
    print("\n9. Predicting on forecast days...")

    # Add features to forecast data
    # Get last history row for rolling stats
    last_history = all_data.iloc[-1].copy()

    # Add time features
    forecast_df['day_of_week'] = forecast_df['date'].dt.dayofweek
    forecast_df['day_of_month'] = forecast_df['date'].dt.day
    forecast_df['month'] = forecast_df['date'].dt.month
    forecast_df['year'] = forecast_df['date'].dt.year

    # Add interaction features
    forecast_df['temp_humidity'] = forecast_df['temperature'] * forecast_df['humidity']
    forecast_df['pressure_temp'] = forecast_df['pressure'] * forecast_df['temperature']

    # Add rolling features (use last history values)
    forecast_df['pm25_rolling_7d_mean'] = last_history['pm25_rolling_7d_mean']
    forecast_df['pm25_rolling_7d_std'] = last_history['pm25_rolling_7d_std']
    forecast_df['temp_rolling_7d_mean'] = last_history['temp_rolling_7d_mean']
    forecast_df['humidity_rolling_7d_mean'] = last_history['humidity_rolling_7d_mean']

    # Add lag features
    last_pm25 = all_data['pm25'].iloc[-1]
    second_last_pm25 = all_data['pm25'].iloc[-2] if len(all_data) >= 2 else last_pm25
    third_last_pm25 = all_data['pm25'].iloc[-3] if len(all_data) >= 3 else second_last_pm25

    forecast_df['pm25_lag2'] = second_last_pm25
    forecast_df['pm25_lag3'] = third_last_pm25

    # Prepare features for prediction
    forecast_features = forecast_df[feature_columns_list]

    # Make predictions
    predictions = model.predict(forecast_features)
    forecast_df['pm25_pred'] = predictions

    print(f"   Generated {len(predictions)} predictions")

    # Create predictions feature table
    print("\n10. Creating predictions feature table...")
    pred_fg_name = 'airqpred2ce555'

    # Get or create predictions feature group
    pred_fg = fs.get_or_create_feature_group(
        name=pred_fg_name,
        version=1,
        description='PM2.5 predictions for forecast days',
        primary_key=['date'],
        event_time='date',
        online_enabled=True
    )
    print(f"   Predictions feature group {pred_fg_name} ready")

    # Insert predictions
    pred_data = forecast_df[['date', 'pm25_pred']].copy()
    pred_fg.insert(pred_data, write_options={'wait_for_job': True})
    print(f"   Predictions inserted into feature table {pred_fg_name}")

    print("\n" + "=" * 60)
    print("Pipeline execution complete!")
    print("=" * 60)
    print(f"Feature group: {fg_name}")
    print(f"Training dataset: {dataset_name}")
    print(f"Model: airqmodel2ce555")
    print(f"Predictions table: {pred_fg_name}")
    print(f"Validation RMSE: {rmse_val:.4f}")
    print(f"Target RMSE: <= 2.124")
