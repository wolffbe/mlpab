#!/usr/bin/env python3
"""
Full FTI pipeline for air-quality PM2.5 forecasting (final version).

Steps:
1. Create feature group `airqfcd91b` from `airquality_history.csv`.
2. Assemble training dataset `airqtdfcd91b`.
3. Train and register a regressor `airqmodelfcd91b` with metrics using Databricks AutoML.
4. Predict `pm25` for `forecast_days.csv` and store in `airqpredfcd91b`.
5. Enable low-latency lookup for the predictions table.
"""

import os
import pandas as pd
import mlflow
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog, jobs, ml
from databricks.sdk.service.catalog import TableType, ColumnTypeName
from databricks.sdk.service.serving import EndpointStateReady

# Environment variables
SCHEMA = os.getenv("MLPAB_DATABRICKS_SCHEMA")
PREFIX = os.getenv("MLPAB_DATABRICKS_PREFIX")
CATALOG, SCHEMA_NAME = SCHEMA.split(".")

# Initialize Databricks client
w = WorkspaceClient()

# Read input data
history_df = pd.read_csv("data/airquality_history.csv")
forecast_df = pd.read_csv("data/forecast_days.csv")

# --- Step 1: Create Feature Group (`airqfcd91b`) ---
def create_feature_group():
    """Create a feature group in Unity Catalog."""
    feature_group_name = f"{PREFIX}_airqfcd91b"
    
    # Define schema for the feature group
    columns = [
        catalog.Column(name="date", type_name=ColumnTypeName.DATE, nullable=False, comment="Record date"),
        catalog.Column(name="pm25_lag1", type_name=ColumnTypeName.FLOAT, nullable=True, comment="Previous day's PM2.5"),
        catalog.Column(name="temperature", type_name=ColumnTypeName.FLOAT, nullable=True, comment="Temperature"),
        catalog.Column(name="humidity", type_name=ColumnTypeName.FLOAT, nullable=True, comment="Humidity"),
        catalog.Column(name="wind_speed", type_name=ColumnTypeName.FLOAT, nullable=True, comment="Wind speed"),
        catalog.Column(name="pressure", type_name=ColumnTypeName.FLOAT, nullable=True, comment="Pressure"),
        catalog.Column(name="precipitation", type_name=ColumnTypeName.FLOAT, nullable=True, comment="Precipitation"),
        catalog.Column(name="pm25", type_name=ColumnTypeName.FLOAT, nullable=True, comment="Target: PM2.5"),
    ]
    
    # Create the feature table
    try:
        w.tables.create(
            name=feature_group_name,
            catalog_name=CATALOG,
            schema_name=SCHEMA_NAME,
            table_type=TableType.MANAGED,
            columns=columns,
            comment="Feature group for air-quality PM2.5 forecasting",
        )
        print(f"Created feature group: {feature_group_name}")
    except Exception as e:
        print(f"Feature group may already exist: {e}")
    
    # Ingest data into the feature group
    spark_df = w.create_dataframe(history_df)
    spark_df.write.save_as_table(
        name=f"{CATALOG}.{SCHEMA_NAME}.{feature_group_name}",
        mode="overwrite",
    )
    print(f"Ingested data into feature group: {feature_group_name}")

# --- Step 2: Assemble Training Dataset (`airqtdfcd91b`) ---
def assemble_training_dataset():
    """Assemble training dataset from the feature group."""
    training_dataset_name = f"{PREFIX}_airqtdfcd91b"
    
    # Create a Delta table for the training dataset
    try:
        w.tables.create(
            name=training_dataset_name,
            catalog_name=CATALOG,
            schema_name=SCHEMA_NAME,
            table_type=TableType.MANAGED,
            columns=[
                catalog.Column(name="date", type_name=ColumnTypeName.DATE, nullable=False),
                catalog.Column(name="pm25_lag1", type_name=ColumnTypeName.FLOAT, nullable=True),
                catalog.Column(name="temperature", type_name=ColumnTypeName.FLOAT, nullable=True),
                catalog.Column(name="humidity", type_name=ColumnTypeName.FLOAT, nullable=True),
                catalog.Column(name="wind_speed", type_name=ColumnTypeName.FLOAT, nullable=True),
                catalog.Column(name="pressure", type_name=ColumnTypeName.FLOAT, nullable=True),
                catalog.Column(name="precipitation", type_name=ColumnTypeName.FLOAT, nullable=True),
                catalog.Column(name="pm25", type_name=ColumnTypeName.FLOAT, nullable=False),
            ],
            comment="Training dataset for air-quality PM2.5 forecasting",
        )
        print(f"Created training dataset table: {training_dataset_name}")
    except Exception as e:
        print(f"Training dataset table may already exist: {e}")
    
    # Ingest data into the training dataset
    spark_df = w.create_dataframe(history_df)
    spark_df.write.save_as_table(
        name=f"{CATALOG}.{SCHEMA_NAME}.{training_dataset_name}",
        mode="overwrite",
    )
    print(f"Ingested data into training dataset: {training_dataset_name}")

# --- Step 3: Train and Register Regressor (`airqmodelfcd91b`) ---
def train_and_register_model():
    """Train a regressor using Databricks AutoML and register it with MLflow."""
    model_name = f"{PREFIX}_airqmodelfcd91b"
    training_dataset_name = f"{CATALOG}.{SCHEMA_NAME}.{PREFIX}_airqtdfcd91b"
    
    # Run AutoML experiment
    experiment = w.automl.regressions.create(
        table_name=training_dataset_name,
        target_col="pm25",
        timeout_minutes=30,
        experiment_name=f"{PREFIX}_automl_pm25",
    )
    
    # Wait for AutoML to complete
    run = experiment.result()
    print(f"AutoML run completed: {run.run_id}")
    
    # Register the best model
    best_model = run.best_trial.model_path
    registered_model = w.model_registry.register_model(
        name=model_name,
        model_path=best_model,
        catalog_name=CATALOG,
        schema_name=SCHEMA_NAME,
    )
    print(f"Registered model: {registered_model.full_name}")

# --- Step 4: Predict `pm25` for `forecast_days.csv` ---
def predict_and_store():
    """Predict `pm25` for `forecast_days.csv` and store in `airqpredfcd91b`."""
    predictions_table_name = f"{PREFIX}_airqpredfcd91b"
    model_name = f"{PREFIX}_airqmodelfcd91b"
    
    # Load the registered model
    model_uri = f"models:/{CATALOG}.{SCHEMA_NAME}.{model_name}/latest"
    model = mlflow.pyfunc.load_model(model_uri)
    
    # Predict
    X_forecast = forecast_df.drop(columns=["date"])
    forecast_df["pm25_pred"] = model.predict(X_forecast)
    
    # Create predictions table
    try:
        w.tables.create(
            name=predictions_table_name,
            catalog_name=CATALOG,
            schema_name=SCHEMA_NAME,
            table_type=TableType.MANAGED,
            columns=[
                catalog.Column(name="date", type_name=ColumnTypeName.DATE, nullable=False, comment="Record date"),
                catalog.Column(name="pm25_pred", type_name=ColumnTypeName.FLOAT, nullable=False, comment="Predicted PM2.5"),
            ],
            comment="Predictions for air-quality PM2.5 forecasting",
        )
        print(f"Created predictions table: {predictions_table_name}")
    except Exception as e:
        print(f"Predictions table may already exist: {e}")
    
    # Ingest predictions
    spark_df = w.create_dataframe(forecast_df[["date", "pm25_pred"]])
    spark_df.write.save_as_table(
        name=f"{CATALOG}.{SCHEMA_NAME}.{predictions_table_name}",
        mode="overwrite",
    )
    print(f"Ingested predictions into table: {predictions_table_name}")
    
    # Enable low-latency lookup
    try:
        w.online_tables.create(
            name=predictions_table_name,
            catalog_name=CATALOG,
            schema_name=SCHEMA_NAME,
            primary_key_columns=["date"],
        )
        print(f"Enabled low-latency lookup for table: {predictions_table_name}")
    except Exception as e:
        print(f"Failed to enable low-latency lookup: {e}")

# --- Run Pipeline ---
if __name__ == "__main__":
    create_feature_group()
    assemble_training_dataset()
    train_and_register_model()
    predict_and_store()
    print("Pipeline completed successfully.")