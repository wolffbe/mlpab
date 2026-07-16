# Databricks notebook source
# Try different approaches to create online table

# COMMAND ----------
from databricks.sdk import WorkspaceClient
import inspect

w = WorkspaceClient()
results = []

# Check the signature of online_tables.create
sig = inspect.signature(w.online_tables.create)
results.append(f"create signature: {sig}")

# Check other methods on online_tables
methods = [m for m in dir(w.online_tables) if not m.startswith('_')]
results.append(f"online_tables methods: {methods}")

print('\n'.join(results))

# COMMAND ----------
from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy

results2 = []

# Try creating with positional arg
try:
    spec = OnlineTableSpec(
        source_table_full_name="workspace.mlpab0442b8.accountse81ff1",
        primary_key_columns=["row_id", "updated_at"],
        run_triggered=OnlineTableSpecTriggeredSchedulingPolicy()
    )
    table = OnlineTable(
        name="workspace.mlpab0442b8.accountse81ff1_online",
        spec=spec
    )
    result = w.online_tables.create(table=table)
    results2.append(f"Success: {result}")
except Exception as e:
    results2.append(f"Error with table kwarg: {type(e).__name__}: {e}")

# Try with just the table object as positional arg
try:
    spec2 = OnlineTableSpec(
        source_table_full_name="workspace.mlpab0442b8.accountse81ff1",
        primary_key_columns=["row_id", "updated_at"],
        run_triggered=OnlineTableSpecTriggeredSchedulingPolicy()
    )
    table2 = OnlineTable(
        name="workspace.mlpab0442b8.accountse81ff1_online",
        spec=spec2
    )
    result2 = w.online_tables.create(table2)
    results2.append(f"Success positional: {result2}")
except Exception as e:
    results2.append(f"Error positional: {type(e).__name__}: {e}")

print('\n'.join(results2))
spark.createDataFrame([(r,) for r in results + results2], ["result"]).write.mode("overwrite").saveAsTable("workspace.mlpab0442b8.online_try_output")
