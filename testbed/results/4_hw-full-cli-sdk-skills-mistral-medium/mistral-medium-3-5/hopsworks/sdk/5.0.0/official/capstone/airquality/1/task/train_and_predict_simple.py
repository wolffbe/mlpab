import hopsworks
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import warnings
warnings.filterwarnings('ignore')

if __name__ == '__main__':
    print("Training and predicting on platform...")
    
    project = hopsworks.login()
    fs = project.get_feature_store()
    mr = project.get_model_registry()
    
    # Get feature view
    fv = fs.get_feature_view('airqtd2ce555', version=1)
    
    # Get all data
    all_data, _ = fv.get_batch_data()
    all_data = all_data.sort_values('date')
    
    # Split
    train_end_date = all_data['date'].max() - pd.Timedelta(days=30)
    train_data = all_data[all_data['date'] <= train_end_date]
    val_data = all_data[all_data['date'] > train_end_date]
    
    # Prepare features
    feature_columns_list = [col for col in train_data.columns if col not in ['date', 'pm25']]
    X_train = train_data[feature_columns_list]
    y_train = train_data['pm25']
    X_val = val_data[feature_columns_list]
    y_val = val_data['pm25']
    
    # Train
    model = RandomForestRegressor(
        n_estimators=200,
        max_depth=10,
        min_samples_split=5,
        min_samples_leaf=2,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred_val = model.predict(X_val)
    rmse_val = np.sqrt(mean_squared_error(y_val, y_pred_val))
    print(f"Validation RMSE: {rmse_val:.4f}")
    
    # Register model
    input_example = X_train.iloc[0:1].to_dict('records')[0]
    output_example = {'pm25_pred': float(y_pred_val[0])}
    metrics = {'rmse': float(rmse_val), 'r2': float(r2_score(y_val, y_pred_val))}
    
    model_reg = mr.python.create_model(
        name='airqmodel2ce555',
        metrics=metrics,
        description='PM2.5 forecasting model',
        input_example=input_example,
        output_example=output_example,
        model_file='/tmp/model.pkl',
        framework='scikit-learn'
    )
    joblib.dump(model, '/tmp/model.pkl')
    model_reg.save('/tmp/model.pkl')
    print("Model registered")
    
    # Predict on forecast
    forecast_df = pd.read_csv('Resources/forecast_days.csv')
    forecast_df['date'] = pd.to_datetime(forecast_df['date'])
    
    # Add features
    last_history = all_data.iloc[-1].copy()
    forecast_df['day_of_week'] = forecast_df['date'].dt.dayofweek
    forecast_df['day_of_month'] = forecast_df['date'].dt.day
    forecast_df['month'] = forecast_df['date'].dt.month
    forecast_df['year'] = forecast_df['date'].dt.year
    forecast_df['temp_humidity'] = forecast_df['temperature'] * forecast_df['humidity']
    forecast_df['pressure_temp'] = forecast_df['pressure'] * forecast_df['temperature']
    forecast_df['pm25_rolling_7d_mean'] = last_history['pm25_rolling_7d_mean']
    forecast_df['pm25_rolling_7d_std'] = last_history['pm25_rolling_7d_std']
    forecast_df['temp_rolling_7d_mean'] = last_history['temp_rolling_7d_mean']
    forecast_df['humidity_rolling_7d_mean'] = last_history['humidity_rolling_7d_mean']
    
    last_pm25 = all_data['pm25'].iloc[-1]
    second_last_pm25 = all_data['pm25'].iloc[-2] if len(all_data) >= 2 else last_pm25
    third_last_pm25 = all_data['pm25'].iloc[-3] if len(all_data) >= 3 else second_last_pm25
    forecast_df['pm25_lag2'] = second_last_pm25
    forecast_df['pm25_lag3'] = third_last_pm25
    
    # Predict
    forecast_features = forecast_df[feature_columns_list]
    predictions = model.predict(forecast_features)
    forecast_df['pm25_pred'] = predictions
    
    # Save predictions to feature group
    pred_fg = fs.get_or_create_feature_group(
        name='airqpred2ce555',
        version=1,
        description='PM2.5 predictions',
        primary_key=['date'],
        event_time='date',
        online_enabled=True
    )
    pred_data = forecast_df[['date', 'pm25_pred']].copy()
    pred_fg.insert(pred_data, write_options={'wait_for_job': True})
    print(f"Predictions saved, RMSE: {rmse_val:.4f}")
