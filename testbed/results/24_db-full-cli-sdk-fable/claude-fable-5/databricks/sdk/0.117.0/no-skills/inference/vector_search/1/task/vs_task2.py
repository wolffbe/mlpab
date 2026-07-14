import csv, json, os, time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.vectorsearch import (
    EndpointType, VectorIndexType, DirectAccessVectorIndexSpec, EmbeddingVectorColumn)

w = WorkspaceClient()
schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]
prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]
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
    print("DONE store=", store, flush=True)

# ---------- native vector search path ----------
spec = DirectAccessVectorIndexSpec(
    embedding_vector_columns=[EmbeddingVectorColumn(name="embedding", embedding_dimension=16)],
    schema_json=json.dumps({"item_id": "string", "embedding": "array<float>", "label": "string"}),
)

def try_native():
    """Returns endpoint name if index exists/created, else None."""
    try:
        idx = w.vector_search_indexes.get_index(index_name)
        return idx.endpoint_name
    except Exception:
        pass
    eps = list(w.vector_search_endpoints.list_endpoints())
    online = [e for e in eps if e.endpoint_status and e.endpoint_status.state.value == "ONLINE"]
    online.sort(key=lambda e: (e.name != endpoint_name, e.num_indexes or 0))
    for ep in online:
        try:
            w.vector_search_indexes.create_index(
                name=index_name, endpoint_name=ep.name, primary_key="item_id",
                index_type=VectorIndexType.DIRECT_ACCESS, direct_access_index_spec=spec)
            print("index created on", ep.name, flush=True)
            return ep.name
        except Exception as e:
            if "already exists" in str(e).lower():
                return ep.name
            print("create_index", ep.name, "->", e, flush=True)
    if not any(e.name == endpoint_name for e in eps):
        try:
            w.vector_search_endpoints.create_endpoint(
                name=endpoint_name, endpoint_type=EndpointType.STANDARD)
            print("own endpoint accepted, provisioning", flush=True)
        except Exception as e:
            print("create_endpoint ->", e, flush=True)
    return None

def finish_native():
    for _ in range(180):
        st = w.vector_search_indexes.get_index(index_name).status
        if st and st.ready:
            break
        time.sleep(10)
    print("index ready", flush=True)
    for i in range(0, len(items), 100):
        r = w.vector_search_indexes.upsert_data_vector_index(
            index_name=index_name, inputs_json=json.dumps(items[i:i+100]))
        print("upsert", i, r.status, flush=True)
    neighbors = {}
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
    write_answers(index_name, neighbors)

# ---------- SQL fallback path ----------
def run_sql(stmt, wh, deadline_s=600):
    r = w.statement_execution.execute_statement(
        statement=stmt, warehouse_id=wh, wait_timeout="50s")
    t0 = time.time()
    while r.status.state.value in ("PENDING", "RUNNING"):
        if time.time() - t0 > deadline_s:
            try:
                w.statement_execution.cancel_execution(r.statement_id)
            except Exception:
                pass
            raise TimeoutError("statement stuck")
        time.sleep(5)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state.value != "SUCCEEDED":
        raise RuntimeError(f"SQL {r.status.state}: {r.status.error}")
    return r

def try_sql_warehouse():
    """Returns a warehouse id that can run a statement, else None."""
    for cand in w.warehouses.list():
        try:
            run_sql("SELECT 1", cand.id, deadline_s=420)
            print("warehouse usable:", cand.name, flush=True)
            return cand.id
        except Exception as e:
            print("warehouse", cand.name, "->", e, flush=True)
    return None

def finish_sql(wh):
    table = index_name
    def arr(v): return "array(" + ",".join(f"{x}D" for x in v) + ")"
    run_sql(f"CREATE OR REPLACE TABLE {table} (item_id STRING, embedding ARRAY<DOUBLE>, label STRING)", wh)
    vals = ",".join(f"('{it['item_id']}',{arr(it['embedding'])},'{it['label']}')" for it in items)
    run_sql(f"INSERT INTO {table} VALUES {vals}", wh)
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
""", wh)
    neighbors = {}
    for qid, item_id in res.result.data_array:
        neighbors.setdefault(qid, []).append(item_id)
    assert len(neighbors) == len(queries) and all(len(v) == 5 for v in neighbors.values())
    write_answers(table, neighbors)

# ---------- main loop: whichever path opens first ----------
deadline = time.time() + 40 * 60
while True:
    ep = try_native()
    if ep:
        finish_native()
        break
    wh = try_sql_warehouse()
    if wh:
        # give native one last immediate chance, else SQL
        ep = try_native()
        if ep:
            finish_native()
        else:
            finish_sql(wh)
        break
    if time.time() > deadline:
        raise SystemExit("EXHAUSTED: neither vector search nor SQL warehouse available")
    print("both paths unavailable; sleeping 60s", flush=True)
    time.sleep(60)
