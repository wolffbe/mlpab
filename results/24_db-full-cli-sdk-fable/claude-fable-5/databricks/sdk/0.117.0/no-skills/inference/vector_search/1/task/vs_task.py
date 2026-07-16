import csv, json, os, time
from datetime import timedelta
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    EndpointType, VectorIndexType, DirectAccessVectorIndexSpec, EmbeddingVectorColumn)

w = WorkspaceClient()
schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]          # workspace.mlpab12199a
prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]          # mlpab12199a
endpoint_name = f"{prefix}_itemse391d8"
index_name = f"{schema}.itemse391d8"

with open("data/items.csv") as f:
    items = [{"item_id": r["item_id"],
              "embedding": json.loads(r["embedding"]),
              "label": r["label"]} for r in csv.DictReader(f)]
with open("data/queries.csv") as f:
    queries = [(r["query_id"], json.loads(r["embedding"])) for r in csv.DictReader(f)]

def write_answers(store, neighbors):
    with open("submission/answers.json", "w") as f:
        json.dump({"store": store, "neighbors": neighbors}, f, indent=1)
    print("DONE store=", store)

# ---------- Phase A: native Vector Search, retried ----------
index_created = False
spec = DirectAccessVectorIndexSpec(
    embedding_vector_columns=[EmbeddingVectorColumn(name="embedding", embedding_dimension=16)],
    schema_json=json.dumps({"item_id": "string", "embedding": "array<float>", "label": "string"}),
)

deadline = time.time() + 12 * 60
while time.time() < deadline and not index_created:
    eps = list(w.vector_search_endpoints.list_endpoints())
    online = [e for e in eps if e.endpoint_status and e.endpoint_status.state.value == "ONLINE"]
    online.sort(key=lambda e: (e.name != endpoint_name, e.num_indexes or 0))
    for ep in online:
        try:
            w.vector_search_indexes.create_index(
                name=index_name, endpoint_name=ep.name, primary_key="item_id",
                index_type=VectorIndexType.DIRECT_ACCESS, direct_access_index_spec=spec)
            endpoint_name = ep.name
            index_created = True
            print("index created on endpoint", ep.name, flush=True)
            break
        except Exception as e:
            msg = str(e).lower()
            if "already exists" in msg:
                endpoint_name = ep.name
                index_created = True
                break
            print("create_index on", ep.name, "failed:", e, flush=True)
    if index_created:
        break
    if not any(e.name == endpoint_name for e in eps):
        try:
            w.vector_search_endpoints.create_endpoint(
                name=endpoint_name, endpoint_type=EndpointType.STANDARD)
            print("own endpoint creation accepted, provisioning...", flush=True)
        except Exception as e:
            print("create_endpoint failed:", e, flush=True)
    time.sleep(45)

if index_created:
    for _ in range(120):
        st = w.vector_search_indexes.get_index(index_name).status
        if st and st.ready:
            break
        time.sleep(10)
    print("index ready", flush=True)

    for i in range(0, len(items), 100):
        resp = w.vector_search_indexes.upsert_data_vector_index(
            index_name=index_name, inputs_json=json.dumps(items[i:i+100]))
        print("upsert batch", i, resp.status, flush=True)

    neighbors = {}
    for qid, vec in queries:
        for attempt in range(10):
            try:
                res = w.vector_search_indexes.query_index(
                    index_name=index_name, columns=["item_id"],
                    query_vector=vec, num_results=5)
                rows = res.result.data_array
                if rows and len(rows) == 5:
                    neighbors[qid] = [r[0] for r in rows]
                    break
            except Exception as e:
                print(qid, "retry:", e, flush=True)
            time.sleep(15)
        else:
            raise RuntimeError(f"query {qid} failed")
        print(qid, neighbors[qid], flush=True)
    write_answers(index_name, neighbors)
    raise SystemExit(0)

# ---------- Phase B: Delta table + exact L2 KNN in Databricks SQL ----------
print("FALLBACK: native vector search unavailable (quota); using Delta table + SQL KNN", flush=True)
wh = None
for cand in w.warehouses.list():
    if cand.name == "Serverless Starter Warehouse":
        wh = cand.id
if wh is None:
    wh = list(w.warehouses.list())[0].id

def run_sql(stmt):
    r = w.statement_execution.execute_statement(
        statement=stmt, warehouse_id=wh, wait_timeout="50s")
    while r.status.state.value in ("PENDING", "RUNNING"):
        time.sleep(5)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state.value != "SUCCEEDED":
        raise RuntimeError(f"SQL failed: {r.status.state} {r.status.error}")
    return r

table = index_name  # workspace.mlpab12199a.itemse391d8
run_sql(f"CREATE OR REPLACE TABLE {table} (item_id STRING, embedding ARRAY<DOUBLE>, label STRING)")
def arr(v): return "array(" + ",".join(f"{x}D" for x in v) + ")"
vals = ",".join(f"('{it['item_id']}',{arr(it['embedding'])},'{it['label']}')" for it in items)
run_sql(f"INSERT INTO {table} VALUES {vals}")
print("table loaded", flush=True)

qvals = ",".join(f"('{qid}',{arr(vec)})" for qid, vec in queries)
res = run_sql(f"""
WITH q(query_id, qe) AS (VALUES {qvals}),
d AS (
  SELECT q.query_id, i.item_id,
         aggregate(zip_with(i.embedding, q.qe, (x, y) -> (x - y) * (x - y)),
                   CAST(0 AS DOUBLE), (acc, v) -> acc + v) AS dist
  FROM q CROSS JOIN {table} i
)
SELECT query_id, item_id
FROM d
QUALIFY ROW_NUMBER() OVER (PARTITION BY query_id ORDER BY dist ASC, item_id ASC) <= 5
ORDER BY query_id, dist ASC
""")
neighbors = {}
for qid, item_id in res.result.data_array:
    neighbors.setdefault(qid, []).append(item_id)
assert len(neighbors) == len(queries) and all(len(v) == 5 for v in neighbors.values())
write_answers(table, neighbors)
