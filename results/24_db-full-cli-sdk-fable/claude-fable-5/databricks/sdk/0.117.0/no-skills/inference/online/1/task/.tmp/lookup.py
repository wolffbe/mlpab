import json
import os

import databricks.sdk
import databricks.sdk.service.sql as sql

w = databricks.sdk.WorkspaceClient()
schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]
online = f"{schema}.profiles44b75e_online"

wh_id = None
for wh in w.warehouses.list():
    if wh.name == "mlpab-grader":
        wh_id = wh.id
if wh_id is None:
    wh_id = next(iter(w.warehouses.list())).id

keys = [line.strip() for line in open("data/lookup_keys.txt") if line.strip()]
print(len(keys), "keys")

vectors = {}
for k in keys:
    resp = w.statement_execution.execute_statement(
        statement=f"SELECT f1, f2, f3, f4 FROM {online} WHERE account_id = :k",
        warehouse_id=wh_id,
        parameters=[sql.StatementParameterListItem(name="k", value=k, type="STRING")],
        wait_timeout="50s",
    )
    state = resp.status.state.value if resp.status and resp.status.state else None
    if state != "SUCCEEDED":
        raise RuntimeError(f"lookup failed for {k}: {state} {resp.status.error}")
    rows = resp.result.data_array
    if not rows or len(rows) != 1:
        raise RuntimeError(f"unexpected rows for {k}: {rows}")
    vectors[k] = [float(v) for v in rows[0]]
    print(k, vectors[k])

os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump({"vectors": vectors}, f, indent=2)
print("wrote submission/answers.json with", len(vectors), "entries")
