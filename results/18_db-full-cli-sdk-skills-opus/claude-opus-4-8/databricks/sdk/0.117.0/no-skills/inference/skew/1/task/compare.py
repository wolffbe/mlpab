import databricks.sdk

w = databricks.sdk.WorkspaceClient()
SCHEMA = "workspace.mlpab98b0c2"
WH = "4dfab06c923fe3cc"
base = "/Volumes/workspace/mlpab98b0c2/compare_vol"


def run(sql):
    r = w.statement_execution.execute_statement(
        warehouse_id=WH, statement=sql, wait_timeout="50s"
    )
    # poll if needed
    import time
    while r.status.state.value in ("PENDING", "RUNNING"):
        time.sleep(2)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state.value != "SUCCEEDED":
        raise RuntimeError(f"{r.status.state} {r.status.error}")
    if r.result and r.result.data_array:
        return r.result.data_array
    return []


# Create tables from CSVs
run(f"""CREATE OR REPLACE TABLE {SCHEMA}.train AS
SELECT * FROM read_files('{base}/training_sample.csv', format=>'csv', header=>true,
  schema=>'entity_id string, f1 double, f2 double, f3 double, f4 double, f5 double')""")
run(f"""CREATE OR REPLACE TABLE {SCHEMA}.serve AS
SELECT * FROM read_files('{base}/serving_log.csv', format=>'csv', header=>true,
  schema=>'entity_id string, f1 double, f2 double, f3 double, f4 double, f5 double')""")
print("tables created")

# Join on entity_id, compute per-feature divergence stats over matching entities
feats = ["f1", "f2", "f3", "f4", "f5"]
sel = ",\n".join(
    [f"avg(abs(t.{f}-s.{f})) as mad_{f}, "
     f"max(abs(t.{f}-s.{f})) as maxd_{f}, "
     f"avg(abs(t.{f}-s.{f})/(abs(t.{f})+1e-9)) as relmad_{f}, "
     f"corr(t.{f}, s.{f}) as corr_{f}"
     for f in feats]
)
sql = f"""SELECT count(*) as n, {sel}
FROM {SCHEMA}.train t JOIN {SCHEMA}.serve s ON t.entity_id = s.entity_id"""
rows = run(sql)
cols = ["n"]
for f in feats:
    cols += [f"mad_{f}", f"maxd_{f}", f"relmad_{f}", f"corr_{f}"]
vals = rows[0]
d = dict(zip(cols, vals))
print("matched rows n =", d["n"])
print(f"{'feat':5} {'mean_abs_diff':>14} {'max_abs_diff':>14} {'rel_mad':>12} {'corr':>10}")
for f in feats:
    print(f"{f:5} {float(d['mad_'+f]):14.6f} {float(d['maxd_'+f]):14.6f} {float(d['relmad_'+f]):12.6f} {float(d['corr_'+f]):10.6f}")
