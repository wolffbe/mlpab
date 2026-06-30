import json, os, time
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
WH = "4dfab06c923fe3cc"
ONLINE = "workspace.mlpab6cf45f.profilesf45007_online"

def run(sql):
    r = w.statement_execution.execute_statement(warehouse_id=WH, statement=sql, wait_timeout="50s")
    st = r.status.state.value if r.status else None
    while st in ("PENDING", "RUNNING"):
        time.sleep(1)
        r = w.statement_execution.get_statement(r.statement_id)
        st = r.status.state.value
    if st != "SUCCEEDED":
        raise RuntimeError(f"FAILED {st}: {r.status.error}")
    return r

with open("data/lookup_keys.txt") as f:
    keys = [ln.strip() for ln in f if ln.strip()]
print("keys:", len(keys))

vectors = {}
for k in keys:
    # online point lookup through the synced (Lakebase) online store
    r = run(f"SELECT f1, f2, f3, f4 FROM {ONLINE} WHERE account_id = '{k}'")
    rows = r.result.data_array if r.result else None
    if not rows:
        raise RuntimeError(f"no online row for {k}")
    f1, f2, f3, f4 = rows[0]
    vectors[k] = [float(f1), float(f2), float(f3), float(f4)]
    print(k, vectors[k])

os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump({"vectors": vectors}, f, indent=2)
print("wrote submission/answers.json with", len(vectors), "vectors")
