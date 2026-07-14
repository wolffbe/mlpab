import csv, json, os, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    VectorIndexType, DirectAccessVectorIndexSpec, EmbeddingVectorColumn)

w = WorkspaceClient()
schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]
index_name = f"{schema}.itemse391d8"

with open("data/items.csv") as f:
    items = [{"item_id": r["item_id"],
              "embedding": json.loads(r["embedding"]),
              "label": r["label"]} for r in csv.DictReader(f)]
with open("data/queries.csv") as f:
    queries = [(r["query_id"], json.loads(r["embedding"])) for r in csv.DictReader(f)]

# wait (cheaply) for a warehouse to come up
wh = None
t0 = time.time()
while time.time() - t0 < 15 * 60:
    ws = list(w.warehouses.list())
    states = {x.name: x.state.value for x in ws}
    running = [x for x in ws if x.state.value == "RUNNING"]
    print(int(time.time() - t0), states, flush=True)
    if running:
        wh = running[0].id
        break
    time.sleep(15)
if wh is None:
    raise SystemExit("EXHAUSTED: no warehouse reached RUNNING")

def run_sql(stmt, deadline_s=600):
    r = w.statement_execution.execute_statement(
        statement=stmt, warehouse_id=wh, wait_timeout="50s")
    t = time.time()
    while r.status.state.value in ("PENDING", "RUNNING"):
        if time.time() - t > deadline_s:
            raise TimeoutError("statement stuck")
        time.sleep(5)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state.value != "SUCCEEDED":
        raise RuntimeError(f"SQL {r.status.state}: {r.status.error}")
    return r

# one more shot at the native index now that some capacity freed up
native_ep = None
try:
    idx = w.vector_search_indexes.get_index(index_name)
    native_ep = idx.endpoint_name
except Exception:
    spec = DirectAccessVectorIndexSpec(
        embedding_vector_columns=[EmbeddingVectorColumn(name="embedding", embedding_dimension=16)],
        schema_json=json.dumps({"item_id": "string", "embedding": "array<float>", "label": "string"}),
    )
    for ep in w.vector_search_endpoints.list_endpoints():
        if ep.endpoint_status and ep.endpoint_status.state.value == "ONLINE":
            try:
                w.vector_search_indexes.create_index(
                    name=index_name, endpoint_name=ep.name, primary_key="item_id",
                    index_type=VectorIndexType.DIRECT_ACCESS, direct_access_index_spec=spec)
                native_ep = ep.name
                break
            except Exception as e:
                print("native still blocked:", e, flush=True)
                break

neighbors = {}
if native_ep:
    print("native index available on", native_ep, flush=True)
    for _ in range(180):
        st = w.vector_search_indexes.get_index(index_name).status
        if st and st.ready:
            break
        time.sleep(10)
    for i in range(0, len(items), 100):
        w.vector_search_indexes.upsert_data_vector_index(
            index_name=index_name, inputs_json=json.dumps(items[i:i+100]))
    for qid, vec in queries:
        for _ in range(12):
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
            time.sleep(10)
        else:
            raise RuntimeError(f"query {qid} failed")
        print(qid, neighbors[qid], flush=True)
    store = index_name
else:
    print("using SQL KNN fallback on warehouse", wh, flush=True)
    table = index_name
    def arr(v): return "array(" + ",".join(f"{x}D" for x in v) + ")"
    run_sql(f"CREATE OR REPLACE TABLE {table} (item_id STRING, embedding ARRAY<DOUBLE>, label STRING)")
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
    for qid, item_id in res.result.data_array:
        neighbors.setdefault(qid, []).append(item_id)
    store = table

assert len(neighbors) == len(queries) and all(len(v) == 5 for v in neighbors.values())
with open("submission/answers.json", "w") as f:
    json.dump({"store": store, "neighbors": neighbors}, f, indent=1)
print("DONE store=", store)
