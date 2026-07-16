#!/usr/bin/env python3
"""
Script to register the scorer model in Unity Catalog.
"""
import argparse
import mlflow
import os
import sys
import shutil

# Add the data directory to Python path
sys.path.append(os.path.join(os.path.dirname(__file__), "data"))
from scorer import score

# Set the tracking URI to use Databricks
mlflow.set_registry_uri("databricks")

# Create a wrapper class for the scorer function
class ScorerModel(mlflow.pyfunc.PythonModel):
    def load_context(self, context):
        pass
    
    def predict(self, context, model_input):
        return score(model_input)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model-name", type=str, required=True, 
                       help="The name of the model to register")
    args = parser.parse_args()
    
    # Create a temporary directory for the model
    model_dir = os.path.join(os.path.dirname(__file__), "model")
    os.makedirs(model_dir, exist_ok=True)
    
    # Copy the scorer.py file to the model directory
    shutil.copy(os.path.join(os.path.dirname(__file__), "data/scorer.py"), 
                os.path.join(model_dir, "scorer.py"))
    
    # Log the model
    with mlflow.start_run():
        mlflow.pyfunc.log_model(
            artifact_path="scorer_model",
            python_model=ScorerModel(),
            code_path=[os.path.join(model_dir, "scorer.py")],
            registered_model_name=args.model_name
        )
    
    print(f"Model registered as: {args.model_name}")

if __name__ == "__main__":
    main()