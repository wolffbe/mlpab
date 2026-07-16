#!/usr/bin/env python3
"""Training script for air quality PM2.5 prediction."""

import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error
import joblib
import os

def main():
    # Initialize Hopsworks
    import hopsworks
    project = hopsworks.login()
    fs = project.get_feature_store()
    
    # Get the feature view
    fv = fs.get_feature_view("airq2ce555_fv", version=1)
    
    # Get batch data directly from the feature view
    td_df = fv.get_batch_data()
    
    print(f"Training dataset shape: {td_df.shape}")
    print(f"Columns: {td_df.columns.tolist()}")
    print(f"\nFirst few rows:\n{td_df.head()}")
    
    # Separate features and label
    # The label is pm25, features are everything else except date
    if 'pm25' in td_df.columns:
        X = td_df.drop(['pm25', 'date'], axis=1, errors='ignore')
        y = td_df['pm25']
        
        print(f"\nFeatures: {X.columns.tolist()}")
        print(f"Target shape: {y.shape}")
        
        # Split data
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )
        
        # Train model
        model = GradientBoostingRegressor(
            n_estimators=200,
            learning_rate=0.05,
            max_depth=5,
            random_state=42
        )
        
        model.fit(X_train, y_train)
        
        # Evaluate
        y_pred = model.predict(X_test)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        print(f"\nTest RMSE: {rmse:.4f}")
        
        # Save model
        model_dir = "/tmp/airqmodel2ce555"
        os.makedirs(model_dir, exist_ok=True)
        joblib.dump(model, os.path.join(model_dir, "model.joblib"))
        
        # Save metrics
        with open(os.path.join(model_dir, "metrics.txt"), "w") as f:
            f.write(f"rmse={rmse:.4f}\n")
        
        print(f"\nModel saved to {model_dir}")
        
        # Register model
        mr = project.get_model_registry()
        mr.register(
            name="airqmodel2ce555",
            path=model_dir,
            description=f"Gradient Boosting Regressor for PM2.5 prediction, RMSE={rmse:.4f}",
            framework="sklearn",
            metrics={"rmse": rmse}
        )
        
        print("Model registered successfully!")
    else:
        print("ERROR: pm25 column not found in training dataset")
        print(f"Available columns: {td_df.columns.tolist()}")

if __name__ == "__main__":
    main()
