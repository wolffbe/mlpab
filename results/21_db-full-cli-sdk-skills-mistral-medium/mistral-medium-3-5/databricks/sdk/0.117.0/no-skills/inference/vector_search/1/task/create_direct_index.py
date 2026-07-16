import databricks.sdk
from databricks.sdk.service.vectorsearch import (
    VectorIndexType,
    IndexSubtype,
    DirectAccessVectorIndexSpec,
    EmbeddingVectorColumn,
)
import json

client = databricks.sdk.WorkspaceClient()

# Create a direct access index with schema_json
INDEX_NAME = 'workspace.mlpabd7bcb5.itemsffc8a7_vec_idx'
ENDPOINT_NAME = 'mlpabd7bcb5_itemsffc8a7'

# Schema JSON for the table
schema = {
    "columns": [
        {"name": "item_id", "type": "string"},
        {"name": "embedding", "type": {"type": "array", "elementType": "float", "containsNull": True}},
        {"name": "label", "type": "string"}
    ]
}
schema_json = json.dumps(schema)

index_spec = DirectAccessVectorIndexSpec(
    embedding_vector_columns=[
        EmbeddingVectorColumn(
            name='embedding',
            embedding_dimension=16
        )
    ],
    schema_json=schema_json
)

try:
    index = client.vector_search_indexes.create_index(
        name=INDEX_NAME,
        endpoint_name=ENDPOINT_NAME,
        primary_key='item_id',
        index_type=VectorIndexType.DIRECT_ACCESS,
        index_subtype=IndexSubtype.HYBRID,
        direct_access_index_spec=index_spec
    )
    print(f'Created direct access index: {index.name}')
except Exception as e:
    print(f'Error: {e}')
