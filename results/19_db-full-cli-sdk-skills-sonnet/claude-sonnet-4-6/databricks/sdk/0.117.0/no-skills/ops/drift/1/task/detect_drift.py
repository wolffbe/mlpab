import os
import json
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()

schema = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # workspace.mlpaba2e00b
prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]  # mlpaba2e00b
user = w.current_user.me().user_name

catalog, schema_name = schema.split(".")
warehouse_id = "4dfab06c923fe3cc"

print(f"Schema: {schema}, Prefix: {prefix}")
print(f"User: {user}")

table_name = f"{schema}.features_drift"
volume_path = f"/Volumes/{catalog}/{schema_name}/data"


def run_sql(sql, timeout=120):
    """Execute SQL statement and wait for result."""
    stmt = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=warehouse_id,
        wait_timeout="50s",
    )
    start = time.time()
    while stmt.status.state in (StatementState.PENDING, StatementState.RUNNING):
        if time.time() - start > timeout:
            raise TimeoutError(f"Statement timed out: {sql[:100]}")
        time.sleep(2)
        stmt = w.statement_execution.get_statement(stmt.statement_id)

    if stmt.status.state != StatementState.SUCCEEDED:
        raise RuntimeError(f"Statement failed: {stmt.status.error} | SQL: {sql[:200]}")

    if stmt.result and stmt.result.data_array:
        cols = [c.name for c in stmt.manifest.schema.columns]
        return [dict(zip(cols, row)) for row in stmt.result.data_array]
    return []


# Create the features table
print("Creating features table...")
run_sql(f"DROP TABLE IF EXISTS {table_name}")
run_sql(f"""
CREATE TABLE {table_name}
AS SELECT * FROM read_files('{volume_path}/features.csv', format => 'csv', header => true)
""")
print("Table created")

# Verify row count
result = run_sql(f"SELECT COUNT(*) as cnt FROM {table_name}")
print(f"Row count: {result}")

# Compute daily statistics per feature
print("Computing daily statistics...")
stats_sql = f"""
SELECT
    DATE(event_time) as dt,
    AVG(f1) as f1_mean, STDDEV(f1) as f1_std,
    AVG(f2) as f2_mean, STDDEV(f2) as f2_std,
    AVG(f3) as f3_mean, STDDEV(f3) as f3_std,
    AVG(f4) as f4_mean, STDDEV(f4) as f4_std,
    AVG(f5) as f5_mean, STDDEV(f5) as f5_std,
    AVG(f6) as f6_mean, STDDEV(f6) as f6_std
FROM {table_name}
GROUP BY DATE(event_time)
ORDER BY dt
"""

daily_stats = run_sql(stats_sql)
print(f"Got {len(daily_stats)} days of stats")

features = ["f1", "f2", "f3", "f4", "f5", "f6"]

# For each feature, detect when its mean shifted using a baseline vs rolling comparison
drift_results = {}

for feat in features:
    means = [float(r[f"{feat}_mean"]) for r in daily_stats if r[f"{feat}_mean"] is not None]
    dates = [r["dt"] for r in daily_stats if r[f"{feat}_mean"] is not None]

    # Use first 30 days as baseline
    baseline = means[:30]
    baseline_mean = sum(baseline) / len(baseline)
    baseline_std = (sum((x - baseline_mean)**2 for x in baseline) / len(baseline))**0.5

    if baseline_std < 1e-10:
        continue

    # Scan with a 7-day rolling window to find first day of sustained shift
    window = 7
    max_zscore = 0
    onset_date = None

    for i in range(window, len(means)):
        window_mean = sum(means[i-window:i]) / window
        zscore = abs(window_mean - baseline_mean) / baseline_std
        if zscore > max_zscore:
            max_zscore = zscore
        if zscore > 2.0 and onset_date is None:
            onset_date = dates[i - window]

    drift_results[feat] = {
        "max_zscore": max_zscore,
        "onset_date": onset_date,
        "baseline_mean": baseline_mean,
        "baseline_std": baseline_std,
    }
    print(f"  {feat}: max_zscore={max_zscore:.3f}, baseline_mean={baseline_mean:.4f}, onset={onset_date}")

# Find feature with highest drift signal
best_feat = max(drift_results.items(), key=lambda x: x[1]["max_zscore"])
print(f"\nBest drift candidate: {best_feat[0]} with z-score {best_feat[1]['max_zscore']:.3f}")
print(f"Onset date: {best_feat[1]['onset_date']}")

# Print all for debugging
print("\nAll drift scores:")
for f, d in sorted(drift_results.items(), key=lambda x: -x[1]["max_zscore"]):
    print(f"  {f}: z={d['max_zscore']:.3f} onset={d['onset_date']}")

# Save answer
os.makedirs("submission", exist_ok=True)
answer = {
    "feature": best_feat[0],
    "onset": str(best_feat[1]["onset_date"])
}
with open("submission/answers.json", "w") as f:
    json.dump(answer, f)
print(f"\nAnswer written: {answer}")
