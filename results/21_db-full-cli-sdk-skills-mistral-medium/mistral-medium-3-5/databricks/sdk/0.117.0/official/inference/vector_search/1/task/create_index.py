#!/usr/bin/env python3
import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    VectorIndexType,
    IndexSubtype,
    DeltaSyncVectorIndexSpecRequest,
    EmbeddingSourceColumn,
)

wc = WorkspaceClient()
prefix = os.environ.get('MLPAB_DATABRICKS_PREFIX', 'mlpabde8d0a')
schema = os.environ.get('MLPAB_DATABRICKS_SCHEMA', 'workspace.mlpabde8d0a')

endpoint_name = prefix + '_itemsffc8a7'
index_name = schema + '.itemsffc8a7'
table_name = schema + '.itemsffc8a7'

print('Endpoint:', endpoint_name)
print('Index:', index_name)
print('Table:', table_name)

# Check if endpoint exists
endpoints = list(wc.vector_search_endpoints.list_endpoints())
our_endpoint = None
for ep in endpoints:
    if ep.name == endpoint_name:
        our_endpoint = ep
        break

if not our_endpoint:
    print('Creating endpoint...')
    wc.vector_search_endpoints.create_endpoint(
        name=endpoint_name,
        endpoint_type='STANDARD',
    )
    print('Waiting for endpoint to be online...')
    wc.vector_search_endpoints.wait_get_endpoint_vector_search_endpoint_online(
        endpoint_name=endpoint_name
    )
    print('Endpoint is online')
else:
    print('Endpoint already exists')

# Check if index exists
indexes = list(wc.vector_search_indexes.list_indexes(endpoint_name=endpoint_name))
our_index = None
for idx in indexes:
    if idx.name == index_name:
        our_index = idx
        break

if not our_index:
    print('Creating index...')
    wc.vector_search_indexes.create_index(
        name=index_name,
        endpoint_name=endpoint_name,
        primary_key='item_id',
        index_type=VectorIndexType.DELTA_SYNC,
        delta_sync_index_spec=DeltaSyncVectorIndexSpecRequest(
            source_table=table_name,
            embedding_source_columns=[EmbeddingSourceColumn(name='embedding')],
        ),
        index_subtype=IndexSubtype.HYBRID,
    )
    print('Index created')
    
    # Wait for index to sync
    print('Waiting for index to sync (60 seconds)...')
    time.sleep(60)
    print('Index should be ready')
else:
    print('Index already exists')

print('Done')
