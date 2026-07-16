import csv, json, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    VectorIndexType, DirectAccessVectorIndexSpec, EmbeddingVectorColumn)

w = WorkspaceClient()
EP = 'mlpab072c27_itemsf3ae88_endpoint'
INDEX = 'workspace.mlpab072c27.itemsf3ae88'
DIM = 16

# --- load inputs ---
items = []
with open('data/items.csv') as f:
    for row in csv.DictReader(f):
        items.append({
            'item_id': row['item_id'],
            'embedding': [float(x) for x in json.loads(row['embedding'])],
            'label': row['label'],
        })
queries = []
with open('data/queries.csv') as f:
    for row in csv.DictReader(f):
        queries.append({'query_id': row['query_id'],
                        'embedding': [float(x) for x in json.loads(row['embedding'])]})
print('items', len(items), 'queries', len(queries))

# --- create index ---
existing = [i.name for i in w.vector_search_indexes.list_indexes(EP)]
print('existing indexes', existing)
if INDEX not in existing:
    spec = DirectAccessVectorIndexSpec(
        embedding_vector_columns=[EmbeddingVectorColumn(name='embedding', embedding_dimension=DIM)],
        schema_json=json.dumps({'item_id': 'string', 'embedding': 'array<float>', 'label': 'string'}),
    )
    w.vector_search_indexes.create_index(
        name=INDEX, endpoint_name=EP, primary_key='item_id',
        index_type=VectorIndexType.DIRECT_ACCESS, direct_access_index_spec=spec)
    print('created index', INDEX)

# wait until index ready
for i in range(40):
    idx = w.vector_search_indexes.get_index(INDEX)
    st = idx.status
    ready = getattr(st, 'ready', None)
    print('ready?', ready, getattr(st,'detailed_state',None), flush=True)
    if ready:
        break
    time.sleep(10)

# --- upsert items ---
w.vector_search_indexes.upsert_data_vector_index(
    index_name=INDEX, inputs_json=json.dumps(items))
print('upserted', len(items))

# wait for rows to be indexed
for i in range(40):
    idx = w.vector_search_indexes.get_index(INDEX)
    cnt = getattr(idx.status, 'indexed_row_count', None)
    print('indexed_row_count', cnt, flush=True)
    if cnt and cnt >= len(items):
        break
    time.sleep(10)
