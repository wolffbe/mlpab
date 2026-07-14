import os
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]
wh_id = "a832b544eb7dc3fe"

def sql(stmt):
    r = w.statement_execution.execute_statement(statement=stmt, warehouse_id=wh_id, wait_timeout="50s")
    if r.status.state.value != "SUCCEEDED":
        raise RuntimeError(f"{r.status.state}: {r.status.error}")
    return r.result.data_array

feats = ["f1", "f2", "f3", "f4", "f5", "f6"]

# Pearson correlation with label
corr_expr = ", ".join(f"corr({f}, label) AS c_{f}" for f in feats)
print("corr with label:", dict(zip(feats, sql(f"SELECT {corr_expr} FROM {schema}.training_data")[0])))

# Per-feature AUC via rank-sum, plus class-conditional overlap
for f in feats:
    rows = sql(f"""
        WITH ranked AS (
          SELECT label, RANK() OVER (ORDER BY {f}) AS rk
          FROM {schema}.training_data
        )
        SELECT
          (SUM(CASE WHEN label = 1 THEN rk ELSE 0 END)
            - (SUM(label) * (SUM(label) + 1)) / 2.0)
          / (SUM(label) * (COUNT(*) - SUM(label))) AS auc
        FROM ranked
    """)
    ov = sql(f"""
        SELECT label, MIN({f}), MAX({f}), AVG({f}) FROM {schema}.training_data GROUP BY label ORDER BY label
    """)
    print(f, "AUC:", rows[0][0], "| per-class min/max/avg:", ov)
