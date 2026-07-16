#!/usr/bin/env python3
import mlflow
import json
import os

# Load metrics
with open("data/metrics.json", "r") as f:
    metrics = json.load(f)

# Set MLflow tracking URI
mlflow.set_tracking_uri("databricks")
mlflow.set_registry_uri("databricks-uc")

# Set experiment
experiment_name = f"/Users/{os.getenv('USER')}/{os.getenv('MLPAB_DATABRICKS_PREFIX')}/churnmodelc12d6a_experiment"
try:
    experiment = mlflow.get_experiment_by_name(experiment_name)
    if experiment is None:
        experiment_id = mlflow.create_experiment(experiment_name)
    else:
        experiment_id = experiment.experiment_id
except Exception as e:
    print(f"Failed to create experiment: {e}")
    raise

# Start run
with mlflow.start_run(experiment_id=experiment_id) as run:
    # Log metrics
    mlflow.log_metrics(metrics)
    
    # Log model
    with open("data/model.json", "r") as f:
        model = json.load(f)
    mlflow.log_dict(model, "model.json")
    
    # Register model
    model_uri = f"runs:/{run.info.run_id}/model.json"
    registered_model = mlflow.register_model(
        model_uri=model_uri,
        name=f"{os.getenv('MLPAB_DATABRICKS_SCHEMA')}.churnmodelc12d6a",
        tags={"task": "registration"}
    )
    
    print(f"Model registered: {registered_model.name}, version: {registered_model.version}")
    
    # Write submission file
    submission = {
        "model_name": "churnmodelc12d6a",
        "version": int(registered_model.version),
        "metrics": metrics
    }
    
    os.makedirs("submission", exist_ok=True)
    with open("submission/answers.json", "w") as f:
        json.dump(submission, f, indent=4)