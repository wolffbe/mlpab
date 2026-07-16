import os
import time
from databricks.sdk import WorkspaceClient

w = WorkspaceClient()
schema = os.environ.get("MLPAB_DATABRICKS_SCHEMA", "workspace.default")
warehouses = list(w.warehouses.list())
warehouse_id = warehouses[0].id

def run_sql(sql):
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
        raise Exception(f"SQL failed: {resp.status.error}")
    return resp

def print_result(resp, title=""):
    if title:
        print(f"\n{title}:")
    if resp.result and resp.result.data_array:
        cols = [c.name for c in resp.manifest.schema.columns]
        print("  " + " | ".join(cols))
        for row in resp.result.data_array:
            print("  " + " | ".join(str(v)[:12] for v in row))

# Test: is s_f4 = t_f4^2?
# If training computes f4 as sqrt(x) and serving uses x directly,
# then s_f4 = x and t_f4 = sqrt(x), so s_f4 = t_f4^2

# Or if training uses x and serving uses x^2:
# s_f4 = t_f4^2

# Check various power relationships
resp = run_sql(f"""
WITH joined AS (
    SELECT t.f4 AS t, s.f4 AS s
    FROM {schema}.training_sample t
    JOIN {schema}.serving_log s ON t.entity_id = s.entity_id
)
SELECT
    CORR(s, t) AS corr_1,
    CORR(s, t*t) AS corr_2,
    CORR(s*s, t) AS corr_sq_t,
    CORR(LN(s), LN(t)) AS corr_log,
    -- If s = t^2: slope of (s vs t^2) should be ~1
    -- regr_slope(y, x)
    REGR_SLOPE(s, t*t) AS slope_s_vs_t2,
    REGR_INTERCEPT(s, t*t) AS intercept_s_vs_t2,
    REGR_R2(s, t*t) AS r2_s_vs_t2,
    REGR_SLOPE(s, t) AS slope_s_vs_t,
    REGR_R2(s, t) AS r2_s_vs_t
FROM joined
""")
print_result(resp, "Regression analysis")

# Check if s = 2*t^2 or something
resp2 = run_sql(f"""
WITH joined AS (
    SELECT t.f4 AS t, s.f4 AS s
    FROM {schema}.training_sample t
    JOIN {schema}.serving_log s ON t.entity_id = s.entity_id
)
SELECT
    MIN(s/t) as min_ratio,
    MAX(s/t) as max_ratio,
    AVG(s/t) as avg_ratio,
    -- median
    PERCENTILE(s/t, 0.5) as median_ratio,
    MIN(LN(s) - LN(t)) as min_log_diff,
    MAX(LN(s) - LN(t)) as max_log_diff,
    AVG(LN(s) - LN(t)) as avg_log_diff,
    PERCENTILE(LN(s) - LN(t), 0.5) as median_log_diff
FROM joined
""")
print_result(resp2, "Ratio and log-ratio analysis")

# What if the skew is that serving uses t^2 / mean(t) or similar?
# Let me check t_mean for training
resp3 = run_sql(f"""
SELECT AVG(f4) as mean_f4, STDDEV(f4) as std_f4, MIN(f4) as min_f4, MAX(f4) as max_f4
FROM {schema}.training_sample
""")
print_result(resp3, "Training f4 stats")

resp4 = run_sql(f"""
SELECT AVG(f4) as mean_f4, STDDEV(f4) as std_f4, MIN(f4) as min_f4, MAX(f4) as max_f4
FROM {schema}.serving_log
""")
print_result(resp4, "Serving f4 stats")

# Check specific: if s_f4 / t_f4 = t_f4 (then s = t^2)
resp5 = run_sql(f"""
WITH joined AS (
    SELECT t.f4 AS t, s.f4 AS s
    FROM {schema}.training_sample t
    JOIN {schema}.serving_log s ON t.entity_id = s.entity_id
)
SELECT
    AVG(ABS(s - t*t)) as mean_diff_s_vs_t2,
    AVG(ABS(s/t - t)) as mean_diff_ratio_vs_t,
    -- maybe s = t^2 + noise?
    STDDEV(s - t*t) as std_diff_s_vs_t2
FROM joined
""")
print_result(resp5, "Check s = t^2")
