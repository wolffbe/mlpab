import json
import mlflow
from mlflow.tracking import MlflowClient

def main():
    # Set up MLflow to use Unity Catalog
    mlflow.set_registry_uri('databricks-uc')
    
    # Load metrics from the job output
    with open('/dbfs/Users/benedict@logicalclocks.com/mlpabb86f4f/output/metrics.json', 'r') as f:
        metrics = json.load(f)
    
    # Register the model in Unity Catalog
    model_name = 'workspace.mlpabb86f4f.ftmodel65e929'
    model_uri = 'dbfs:/Users/benedict@logicalclocks.com/mlpabb86f4f/output/finetuned_model.npz'
    
    # Log the model with metrics
    with mlflow.start_run() as run:
        # Log metrics
        mlflow.log_metrics(metrics)
        
        # Register the model
        model_version = mlflow.register_model(
            model_uri=model_uri,
            name=model_name
        )
        
        print(f'Registered model version: {model_version.version}')
        
        # Set tags with metrics
        client = MlflowClient()
        client.set_model_version_tag(
            name=model_name,
            version=model_version.version,
            key='eval_loss',
            value=str(metrics['eval_loss'])
        )
        client.set_model_version_tag(
            name=model_name,
            version=model_version.version,
            key='base_eval_loss',
            value=str(metrics['base_eval_loss'])
        )

if __name__ == '__main__':
    main()