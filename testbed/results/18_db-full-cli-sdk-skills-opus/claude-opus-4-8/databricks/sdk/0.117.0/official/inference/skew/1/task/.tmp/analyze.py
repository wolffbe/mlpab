import databricks.sdk
from databricks.sdk.service.sql import StatementState
import time

w = databricks.sdk.WorkspaceClient()
SCHEMA = "workspace.mlpab9f3ba9"
CATALOG, SCHEMANAME = SCHEMA.split(".")
WH = "4dfab06c923fe3cc"
VOL = f"/Volumes/{CATALOG}/{SCHEMANAME}/skewvol"


def run_sql(sql):
    r = w.statement_execution.execute_statement(warehouse_id=WH, statement=sql, wait_timeout="50s")
    while r.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(2)
        r = w.statement_execution.get_statement(r.statement_id)
    if r.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"SQL failed: {r.status.state} {r.status.error}")
    return r


def rows(r):
    if r.result and r.result.data_array:
        return r.result.data_array
    return []


# create tables from CSV using read_files
for tbl, fn in [("train_t", "training_sample.csv"), ("serve_t", "serving_log.csv")]:
    run_sql(f"DROP TABLE IF EXISTS {SCHEMA}.{tbl}")
    run_sql(f"""
        CREATE TABLE {SCHEMA}.{tbl} AS
        SELECT * FROM read_files(
            '{VOL}/{fn}',
            format => 'csv',
            header => true,
            inferSchema => true
        )
    """)
    print("created", tbl)

# sanity counts
for tbl in ["train_t", "serve_t"]:
    r = run_sql(f"SELECT count(*) FROM {SCHEMA}.{tbl}")
    print(tbl, "rows:", rows(r))

# Join on entity_id (inner join -> only overlapping entities)
# For each feature compute correlation, mean abs diff, max abs diff between train and serve
feats = ["f1", "f2", "f3", "f4", "f5"]
sel = []
for f in feats:
    sel.append(f"corr(t.{f}, s.{f}) AS corr_{f}")
    sel.append(f"avg(abs(t.{f} - s.{f})) AS mad_{f}")
    sel.append(f"max(abs(t.{f} - s.{f})) AS maxd_{f}")
    sel.append(f"avg(t.{f}) AS tmean_{f}")
    sel.append(f"avg(s.{f}) AS smean_{f}")

q = f"""
SELECT count(*) AS n, {', '.join(sel)}
FROM {SCHEMA}.train_t t
JOIN {SCHEMA}.serve_t s ON t.entity_id = s.entity_id
"""
r = run_sql(q)
cols = [c.name for c in r.manifest.schema.columns]
vals = rows(r)[0]
res = dict(zip(cols, vals))
print("\n=== overlap n ===", res["n"])
print("\nfeature | corr | mean_abs_diff | max_abs_diff | train_mean | serve_mean")
for f in feats:
    print(f"{f}: corr={float(res['corr_'+f]):.6f}  mad={float(res['mad_'+f]):.6f}  "
          f"maxd={float(res['maxd_'+f]):.6f}  tmean={float(res['tmean_'+f]):.4f}  smean={float(res['smean_'+f]):.4f}")
