#!/usr/bin/env python3
"""Simple test to try to create a serving endpoint."""

import os
import json
from datetime import timedelta
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.serving import (
    EndpointCoreConfigInput,
    ServedModelInput,
    ServedModelInputWorkloadType
)

# Environment variables
MLPAB_DATABRICKS_SCHEMA = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpab222a08')
MLPAB_DATABRICKS_PREFIX = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpab222a08')

# Parse schema
catalog_name, schema_name = MLPAB_DATABRICKS_SCHEMA.split('.')

# Initialize workspace client
wc = WorkspaceClient()

# Try to create a serving endpoint with a simple configuration
endpoint_name = "scorer40bb09"

# Try using the existing model from another schema
full_model_name = "workspace.mlpabaf8386.scorer40bb09"

config = EndpointCoreConfigInput(
    name=endpoint_name,
    served_models=[
        ServedModelInput(
            model_name=full_model_name,
            model_version="1",
            workload_size="Small",
            workload_type=ServedModelInputWorkloadType.CPU,
            scale_to_zero_enabled=True,
            min_provisioned_concurrency=0
        )
    ]
)

try:
    print(f"Creating serving endpoint: {endpoint_name}")
    endpoint = wc.serving_endpoints.create_and_wait(
        name=endpoint_name,
        config=config,
        timeout=timedelta(seconds=1200)
    )
    print(f"Serving endpoint created: {endpoint}")
    
    # Test the endpoint
    with open('data/payloads.json', 'r') as f:
        payloads = json.load(f)
    
    for i, payload in enumerate(payloads):
        print(f"Testing payload {i+1}")
        response = wc.serving_endpoints.query(
            name=endpoint_name,
            inputs={"text": payload}
        )
        print(f"Response: {response}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
