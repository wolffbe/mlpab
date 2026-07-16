import csv, json, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    VectorIndexType, DirectAccessVectorIndexSpec, EmbeddingVectorColumn)

w = WorkspaceClient()
ep_name = 'mlpab08ae05_itemsf3ae88'
index_name = 'workspace.mlpab08ae05.itemsf3ae88'
DIM = 16

items = []
with open('data/items.csv') as f:
    for r in csv.DictReader(f):
        items.append({'item_id': r['item_id'],
                      'embedding': json.loads(r['embedding']),
                      'label': r['label']})
print('items', len(items))

existing = [i.name for i in w.vector_search_indexes.list_indexes(endpoint_name=ep_name)]
print('existing idx', existing)
schema_json = json.dumps({"item_id": "string", "embedding": "array<float>", "label": "string"})
if index_name not in existing:
    idx = w.vector_search_indexes.create_index(
        name=index_name,
        endpoint_name=ep_name,
        primary_key='item_id',
        index_type=VectorIndexType.DIRECT_ACCESS,
        direct_access_index_spec=DirectAccessVectorIndexSpec(
            embedding_vector_columns=[EmbeddingVectorColumn(name='embedding', embedding_dimension=DIM)],
            schema_json=schema_json,
        ),
    )
    print('created index', idx.name)
else:
    print('index exists')

# wait until ready
for _ in range(60):
    info = w.vector_search_indexes.get_index(index_name)
    ready = getattr(info.status, 'ready', None)
    print('ready?', ready)
    if ready:
        break
    time.sleep(5)

# upsert all items
resp = w.vector_search_indexes.upsert_data_vector_index(index_name, json.dumps(items))
print('upsert status:', resp.status)
PY_DONE = True
print('DONE_BUILD')
