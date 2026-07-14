#!/usr/bin/env python3

import os
import json
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput, 
    ServedModelInput, 
    ServedModelInputWorkloadType
)

def main():
    # Initialize workspace client
    wc = WorkspaceClient()
    
    # Environment variables
    prefix = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpabc71b12')
    schema = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpabc71b12')
    endpoint_name = "scorer40bb09"
    
    print(f"Prefix: {prefix}, Schema: {schema}, Endpoint: {endpoint_name}")
    
    # Step 1: Upload the MLflow model to workspace
    workspace_model_dir = f"/Users/benedict@hopsworks.ai/{prefix}/scorer_mlflow_model"
    
    # Create model artifacts directory in workspace
    try:
        wc.workspace.mkdirs(workspace_model_dir)
        print(f"✓ Created directory: {workspace_model_dir}")
    except Exception as e:
        print(f"Directory may already exist: {e}")
    
    # Upload all model files to workspace
    model_files = [
        ('MLmodel', 'mlflow_model/MLmodel'),
        ('model.py', 'mlflow_model/model.py'),
        ('requirements.txt', 'mlflow_model/requirements.txt')
    ]
    
    for target_name, local_path in model_files:
        with open(local_path, 'r') as f:
            content = f.read()
        model_file_path = f"{workspace_model_dir}/{target_name}"
        wc.workspace.upload(model_file_path, content.encode(), overwrite=True)
        print(f"✓ Uploaded {target_name} to {model_file_path}")
    
    # Step 2: Create a registered model
    model_name = f"{schema}.scorer_model"
    print(f"Creating registered model: {model_name}")
    
    try:
        model = wc.model_registry.create_model(
            name=model_name,
            description="Trigram log-likelihood scorer model"
        )
        print(f"✓ Created model: {model.name}")
    except Exception as e:
        if "RESOURCE_ALREADY_EXISTS" in str(e):
            print(f"Model {model_name} already exists")
            model = wc.model_registry.get_model(model_name)
        else:
            raise
    
    # Step 3: Create a model version
    # Convert workspace path to dbfs path for model registry
    dbfs_model_path = workspace_model_dir.replace("/Users/", "dbfs:/userfile/")
    print(f"Using model source path: {dbfs_model_path}")
    
    try:
        model_version = wc.model_registry.create_model_version(
            name=model_name,
            source=dbfs_model_path,
            description="Initial version of trigram scorer"
        )
        print(f"✓ Created model version: {model_version.version}")
    except Exception as e:
        if "RESOURCE_ALREADY_EXISTS" in str(e):
            print(f"Model version already exists, getting latest")
            versions = wc.model_registry.get_latest_versions(model_name)
            model_version = versions.model_latest_versions[0]
            print(f"✓ Using existing model version: {model_version.version}")
        else:
            raise
    
    # Step 4: Create serving endpoint
    print(f"Creating serving endpoint: {endpoint_name}")
    
    config = EndpointCoreConfigInput(
        name=f"{prefix}_{endpoint_name}",
        served_models=[
            ServedModelInput(
                name=f"{prefix}_scorer_model",
                model_name=model_name,
                model_version=model_version.version,
                workload_size="Small",
                workload_type=ServedModelInputWorkloadType.CPU,
                scale_to_zero_enabled=True,
                min_provisioned_concurrency=0,
                max_provisioned_concurrency=1
            )
        ]
    )
    
    try:
        # Check if endpoint already exists
        try:
            existing = wc.serving_endpoints.get(endpoint_name)
            print(f"Endpoint {endpoint_name} already exists, updating...")
            # Update the endpoint
            wait = wc.serving_endpoints.update_config(
                name=endpoint_name,
                config=config
            )
            wait.result(timeout=300)  # 5 minute timeout
        except Exception as e:
            if "NOT_FOUND" in str(e) or "does not exist" in str(e):
                # Create new endpoint
                print(f"Creating new endpoint...")
                wait = wc.serving_endpoints.create(
                    name=endpoint_name,
                    config=config,
                    description="Trigram log-likelihood scorer endpoint"
                )
                wait.result(timeout=300)  # 5 minute timeout
            else:
                raise
        
        print(f"✓ Created/updated serving endpoint: {endpoint_name}")
        
        # Wait for endpoint to be ready
        print("Waiting for endpoint to be ready...")
        endpoint = wc.serving_endpoints.wait_get_serving_endpoint_not_updating(endpoint_name, timeout=600)
        print(f"✓ Endpoint is ready: {endpoint.state}")
        
    except Exception as e:
        print(f"Error creating endpoint: {e}")
        raise
    
    # Step 5: Load payloads and invoke endpoint
    print("Loading payloads...")
    with open('data/payloads.json', 'r') as f:
        payloads = json.load(f)
    
    print(f"Loaded {len(payloads)} payloads")
    
    # Step 6: Invoke endpoint on each payload
    print("Invoking endpoint on payloads...")
    responses = []
    
    for i, payload in enumerate(payloads):
        print(f"Processing payload {i+1}/{len(payloads)}: {payload[:50]}...")
        
        try:
            # Try different input formats
            response = wc.serving_endpoints.query(
                name=endpoint_name,
                data={"data": payload}
            )
            
            # Parse the response
            if hasattr(response, 'result') and response.result:
                result_data = response.result.data
                if isinstance(result_data, list) and len(result_data) > 0:
                    result = result_data[0]
                else:
                    result = result_data
            else:
                result = response
            
            print(f"  Response: {result}")
            responses.append(result)
            
        except Exception as e:
            print(f"  Error invoking endpoint: {e}")
            # Try with different input format
            try:
                response = wc.serving_endpoints.query(
                    name=endpoint_name,
                    data=payload  # Direct string
                )
                if hasattr(response, 'result') and response.result:
                    result_data = response.result.data
                    if isinstance(result_data, list) and len(result_data) > 0:
                        result = result_data[0]
                    else:
                        result = result_data
                else:
                    result = response
                print(f"  Response (direct): {result}")
                responses.append(result)
            except Exception as e2:
                print(f"  Error with direct input: {e2}")
                responses.append({"error": str(e2)})
    
    # Step 7: Write results to submission/answers.json
    print("Writing results to submission/answers.json...")
    
    # Create submission directory if it doesn't exist
    os.makedirs('submission', exist_ok=True)
    
    result_data = {
        "endpoint_name": endpoint_name,
        "responses": responses
    }
    
    with open('submission/answers.json', 'w') as f:
        json.dump(result_data, f, indent=2)
    
    print("✓ Written results to submission/answers.json")
    print(f"Final result: {json.dumps(result_data, indent=2)}")

if __name__ == "__main__":
    main()