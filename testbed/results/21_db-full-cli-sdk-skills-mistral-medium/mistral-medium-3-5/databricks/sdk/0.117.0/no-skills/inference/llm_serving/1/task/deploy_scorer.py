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
    
    # Step 1: Upload the scorer.py to workspace
    workspace_path = f"/Users/benedict@hopsworks.ai/{prefix}/scorer.py"
    print(f"Uploading scorer.py to {workspace_path}")
    
    with open('data/scorer.py', 'r') as f:
        scorer_content = f.read()
    
    # Upload the scorer file
    wc.workspace.upload(workspace_path, scorer_content.encode(), overwrite=True)
    print("✓ Uploaded scorer.py")
    
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
    # For this, we need to upload the model artifacts somewhere
    # Let's use dbfs to store the model
    dbfs_model_path = f"dbfs:/Users/benedict@hopsworks.ai/{prefix}/scorer_model"
    
    # Create model artifacts directory
    try:
        wc.dbfs.mkdirs(dbfs_model_path)
    except:
        pass
    
    # Upload scorer.py as model artifact
    model_file_path = f"{dbfs_model_path}/scorer.py"
    with open('data/scorer.py', 'r') as f:
        model_content = f.read()
    
    wc.dbfs.put_file(model_file_path, model_content.encode(), overwrite=True)
    print(f"✓ Uploaded model artifacts to {model_file_path}")
    
    # Create model version
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
            wait.result()
        except:
            # Create new endpoint
            wait = wc.serving_endpoints.create(
                name=endpoint_name,
                config=config,
                description="Trigram log-likelihood scorer endpoint"
            )
            wait.result()
        
        print(f"✓ Created/updated serving endpoint: {endpoint_name}")
        
        # Wait for endpoint to be ready
        print("Waiting for endpoint to be ready...")
        endpoint = wc.serving_endpoints.wait_get_serving_endpoint_not_updating(endpoint_name)
        print(f"✓ Endpoint is ready: {endpoint.state}")
        
    except Exception as e:
        print(f"Error creating endpoint: {e}")
        raise

if __name__ == "__main__":
    main()