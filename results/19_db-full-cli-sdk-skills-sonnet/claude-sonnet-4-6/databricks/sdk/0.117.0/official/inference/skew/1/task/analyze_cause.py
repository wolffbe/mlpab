import os
import time
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
schema = os.environ.get("MLPAB_DATABRICKS_SCHEMA", "workspace.default")
warehouses = list(w.warehouses.list())
warehouse_id = warehouses[0].id

def run_sql(sql, description=""):
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        wait_timeout="50s",
    )
    state = resp.status.state.value
    stmt_id = resp.statement_id
    while state not in ["SUCCEEDED", "FAILED", "CANCELED", "CLOSED"]:
        time.sleep(2)
        resp = w.statement_execution.get_statement(stmt_id)
        state = resp.status.state.value
    if state != "SUCCEEDED":
        raise Exception(f"SQL failed ({state}): {resp.status.error}")
    return resp

# Get sample of matched entities with f4 values
sample_sql = f"""
SELECT
    t.entity_id,
    t.f4 AS t_f4,
    s.f4 AS s_f4,
    s.f4 / NULLIF(t.f4, 0) AS ratio,
    s.f4 - t.f4 AS diff,
    t.f4 * t.f4 AS t_f4_squared,
    SQRT(s.f4) AS sqrt_s_f4,
    LN(s.f4) AS log_s_f4,
    LN(t.f4) AS log_t_f4
FROM {schema}.training_sample t
JOIN {schema}.serving_log s ON t.entity_id = s.entity_id
ORDER BY t.entity_id
LIMIT 20
"""

resp = run_sql(sample_sql, "Sample f4 comparison")
print("Sample f4 comparison (t_f4 vs s_f4):")
if resp.result and resp.result.data_array:
    cols = [c.name for c in resp.manifest.schema.columns]
    print("  " + "\t".join(cols))
    for row in resp.result.data_array:
        print("  " + "\t".join(str(v)[:10] for v in row))

# Check if s_f4 = t_f4^2
check_sql = f"""
SELECT
    AVG(ABS(s.f4 - t.f4 * t.f4)) AS diff_if_squared,
    AVG(ABS(s.f4 - t.f4 * 4)) AS diff_if_4x,
    AVG(ABS(s.f4 - t.f4 * 3)) AS diff_if_3x,
    AVG(ABS(LN(s.f4) - LN(t.f4))) AS diff_log,
    CORR(s.f4, t.f4) AS corr_linear,
    CORR(s.f4, t.f4 * t.f4) AS corr_squared
FROM {schema}.training_sample t
JOIN {schema}.serving_log s ON t.entity_id = s.entity_id
"""

resp2 = run_sql(check_sql, "Check transformation hypothesis")
print("\nHypothesis testing:")
if resp2.result and resp2.result.data_array:
    cols = [c.name for c in resp2.manifest.schema.columns]
    for row in resp2.result.data_array:
        for k, v in zip(cols, row):
            print(f"  {k}: {v}")
