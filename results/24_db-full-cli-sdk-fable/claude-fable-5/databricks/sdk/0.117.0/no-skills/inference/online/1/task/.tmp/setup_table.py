import csv
import os

import databricks.sdk

w = databricks.sdk.WorkspaceClient()

schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # workspace.<run-id>
table = f"{schema}.profiles44b75e"

whs = list(w.warehouses.list())
for wh in whs:
    print(wh.id, wh.name, wh.state.value if wh.state else None)
wh_id = whs[0].id

rows = []
with open("data/features.csv") as f:
    for r in csv.DictReader(f):
        rows.append((r["account_id"], r["f1"], r["f2"], r["f3"], r["f4"]))
print("rows:", len(rows))


def run(sql, **kw):
    resp = w.statement_execution.execute_statement(
        statement=sql, warehouse_id=wh_id, wait_timeout="50s", **kw
    )
    st = resp.status.state.value if resp.status and resp.status.state else None
    if st != "SUCCEEDED":
        raise RuntimeError(f"SQL failed ({st}): {resp.status.error}\n{sql[:200]}")
    return resp


run(f"""
CREATE TABLE IF NOT EXISTS {table} (
  account_id STRING NOT NULL,
  f1 DOUBLE, f2 DOUBLE, f3 DOUBLE, f4 DOUBLE,
  CONSTRAINT profiles44b75e_pk PRIMARY KEY (account_id)
) TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")

values = ",\n".join(
    f"('{a}', {f1}, {f2}, {f3}, {f4})" for a, f1, f2, f3, f4 in rows
)
run(f"INSERT OVERWRITE {table} VALUES\n{values}")

resp = run(f"SELECT COUNT(*) FROM {table}")
print("count in table:", resp.result.data_array)
