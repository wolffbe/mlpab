#!/usr/bin/env python3
"""
Air Quality PM2.5 Forecasting Pipeline
Creates feature group airq2ce555, training dataset airqtd2ce555,
trains and registers model airqmodel2ce555, and creates predictions airqpred2ce555
"""
import hopsworks
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error
from sklearn.model_selection import train_test_split
import joblib
import os

def main():
    # Initialize Hopsworks
    hopsworks.login()
    fs = hopsworks.feature_store.get_feature_store()
    
    # Read the history data
    history_df = pd.read_csv("Resources/airquality_history.csv")
    forecast_df = pd.read_csv("Resources/forecast_days.csv")
    
    print(f"History data shape: {history_df.shape}")
    print(f"Forecast data shape: {forecast_df.shape}")
    print(f"History columns: {history_df.columns.tolist()}")
    print(f"Forecast columns: {forecast_df.columns.tolist()}")
    
    # Create feature group airq2ce555
    # The feature group should contain all features including pm25 for training
    fg_name = "airq2ce555"
    
    # Check if feature group exists, if not create it
    try:
        fg = fs.get_feature_group(fg_name, version=1)
        print(f"Feature group {fg_name} already exists")
    except:
        # Create the feature group
        fg = fs.create_feature_group(
            name=fg_name,
            version=1,
            primary_key=["date"],
            event_time=None,
            online_enabled=True,
            description="Air quality feature group with weather and PM2.5 data"
        )
        
        # Define schema - all features as double except date as string
        from hopsworks.feature_store.api.util import create_schema
        
        schema = create_schema(
            date="string",
            pm25_lag1="double",
            temperature="double",
            humidity="double", 
            wind_speed="double",
            pressure="double",
            precipitation="double",
            pm25="double"
        )
        
        fg.save(schema)
        print(f"Created feature group {fg_name}")
    
    # Insert training data into feature group
    # Convert date to string to match schema
    history_df["date"] = history_df["date"].astype(str)
    
    try:
        fg.insert(history_df, write_options={"wait_for_job": True})
        print(f"Inserted {len(history_df)} rows into feature group")
    except Exception as e:
        print(f"Error inserting data: {e}")
        # Try to get existing FG and insert
        fg = fs.get_feature_group(fg_name, version=1)
        fg.insert(history_df, write_options={"wait_for_job": True})
        print(f"Inserted {len(history_df)} rows into feature group (retry)")
    
    # Create feature view for training
    fv_name = "airq_fv_2ce555"
    try:
        fv = fs.get_feature_view(fv_name, version=1)
        print(f"Feature view {fv_name} already exists")
    except:
        fv = fs.create_feature_view(
            name=fv_name,
            version=1,
            query=fg.select_all(),
            labels=["pm25"]
        )
        print(f"Created feature view {fv_name}")
    
    # Create training dataset airqtd2ce555
    td_name = "airqtd2ce555"
    try:
        td = fs.get_training_dataset(td_name)
        print(f"Training dataset {td_name} already exists")
    except:
        td = fv.create_training_dataset(
            name=td_name,
            description="Training dataset for air quality PM2.5 forecasting",
            data_format="csv"
        )
        print(f"Created training dataset {td_name}")
    
    # Get the training data
    X, y = fv.get_training_data()
    print(f"Training data shape: X={X.shape}, y={y.shape}")
    
    # Split into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    # Train model
    model = RandomForestRegressor(
        n_estimators=100,
        max_depth=10,
        random_state=42,
        n_jobs=-1
    )
    model.fit(X_train, y_train)
    
    # Evaluate
    y_pred = model.predict(X_val)
    rmse = np.sqrt(mean_squared_error(y_val, y_pred))
    print(f"Validation RMSE: {rmse:.4f}")
    
    # Save model
    model_dir = "Models/airqmodel2ce555"
    os.makedirs(model_dir, exist_ok=True)
    joblib.dump(model, os.path.join(model_dir, "model.pkl"))
    
    # Register model
    mr = hopsworks.model_registry.get_registry()
    
    model_metadata = {
        "name": "airqmodel2ce555",
        "description": "Random Forest regressor for PM2.5 forecasting",
        "metrics": {"rmse": float(rmse)},
        "framework": "scikit-learn"
    }
    
    try:
        registered_model = mr.register(
            name="airqmodel2ce555",
            path=model_dir,
            description="Random Forest regressor for PM2.5 forecasting",
            metrics={"rmse": float(rmse)},
            framework="scikit-learn",
            input_example=X_train.iloc[0:1].to_dict(orient="records")[0],
            feature_view=fv_name,
            training_dataset=td_name
        )
        print(f"Registered model: {registered_model}")
    except Exception as e:
        print(f"Error registering model: {e}")
        # Try without optional params
        registered_model = mr.register(
            name="airqmodel2ce555",
            path=model_dir,
            description="Random Forest regressor for PM2.5 forecasting",
            metrics={"rmse": float(rmse)},
            framework="scikit-learn"
        )
        print(f"Registered model (retry): {registered_model}")
    
    # Create predictions for forecast_days
    # Prepare forecast data - it has the same features except pm25
    forecast_df["date"] = forecast_df["date"].astype(str)
    
    # Get feature view to use for batch scoring
    # We need to create a temporary feature group for forecast data or use the existing one
    # For now, let's just predict directly
    
    # Get the feature names from the training data
    feature_columns = [col for col in X.columns if col != "pm25"]
    print(f"Feature columns: {feature_columns}")
    
    # Prepare forecast features
    X_forecast = forecast_df[feature_columns]
    
    # Make predictions
    forecast_predictions = model.predict(X_forecast)
    
    # Create predictions dataframe
    predictions_df = forecast_df["date"].copy()
    predictions_df = pd.DataFrame({
        "date": forecast_df["date"],
        "pm25_pred": forecast_predictions
    })
    
    print(f"Predictions shape: {predictions_df.shape}")
    print(f"Sample predictions:\n{predictions_df.head()}")
    
    # Create predictions feature group airqpred2ce555
    pred_fg_name = "airqpred2ce555"
    try:
        pred_fg = fs.get_feature_group(pred_fg_name, version=1)
        print(f"Predictions feature group {pred_fg_name} already exists")
    except:
        pred_fg = fs.create_feature_group(
            name=pred_fg_name,
            version=1,
            primary_key=["date"],
            event_time=None,
            online_enabled=True,
            description="PM2.5 predictions for forecast days"
        )
        
        from hopsworks.feature_store.api.util import create_schema
        pred_schema = create_schema(
            date="string",
            pm25_pred="double"
        )
        pred_fg.save(pred_schema)
        print(f"Created predictions feature group {pred_fg_name}")
    
    # Insert predictions
    try:
        pred_fg.insert(predictions_df, write_options={"wait_for_job": True})
        print(f"Inserted {len(predictions_df)} predictions")
    except Exception as e:
        print(f"Error inserting predictions: {e}")
        # Try to get existing FG and insert
        pred_fg = fs.get_feature_group(pred_fg_name, version=1)
        pred_fg.insert(predictions_df, write_options={"wait_for_job": True})
        print(f"Inserted {len(predictions_df)} predictions (retry)")
    
    print("Pipeline completed successfully!")

if __name__ == "__main__":
    main()
