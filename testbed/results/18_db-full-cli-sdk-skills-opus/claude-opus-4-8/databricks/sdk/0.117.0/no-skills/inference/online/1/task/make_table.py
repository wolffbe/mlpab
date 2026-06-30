import csv, time
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
WH = "4dfab06c923fe3cc"
SCHEMA = "workspace.mlpab6cf45f"
TABLE = f"{SCHEMA}.profilesf45007"

def run(sql):
    r = w.statement_execution.execute_statement(warehouse_id=WH, statement=sql, wait_timeout="50s")
    st = r.status.state.value if r.status else None
    while st in ("PENDING", "RUNNING"):
        time.sleep(2)
        r = w.statement_execution.get_statement(r.statement_id)
        st = r.status.state.value
    if st != "SUCCEEDED":
        raise RuntimeError(f"FAILED {st}: {r.status.error}\nSQL: {sql[:200]}")
    return r

run(f"DROP TABLE IF EXISTS {TABLE}")
run(f"""CREATE TABLE {TABLE} (
  account_id STRING NOT NULL,
  f1 DOUBLE, f2 DOUBLE, f3 DOUBLE, f4 DOUBLE,
  CONSTRAINT profilesf45007_pk PRIMARY KEY(account_id)
) TBLPROPERTIES (delta.enableChangeDataFeed = true)""")
print("table created")

rows = []
with open("data/features.csv") as f:
    for r in csv.DictReader(f):
        rows.append((r["account_id"], r["f1"], r["f2"], r["f3"], r["f4"]))
print("rows:", len(rows))

B = 60
for i in range(0, len(rows), B):
    chunk = rows[i:i + B]
    vals = ",".join(f"('{a}',{f1},{f2},{f3},{f4})" for (a, f1, f2, f3, f4) in chunk)
    run(f"INSERT INTO {TABLE} (account_id,f1,f2,f3,f4) VALUES {vals}")
    print("inserted batch", i)

r = run(f"SELECT count(*) c FROM {TABLE}")
print("count:", r.result.data_array)
r = run(f"SELECT * FROM {TABLE} WHERE account_id='A0004'")
print("sample A0004:", r.result.data_array)
