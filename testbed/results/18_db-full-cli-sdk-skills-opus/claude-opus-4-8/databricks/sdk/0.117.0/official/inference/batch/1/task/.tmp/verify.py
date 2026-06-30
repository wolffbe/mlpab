import databricks.sdk as s
import os
w = s.WorkspaceClient()
catalog, schema = os.environ["MLPAB_DATABRICKS_SCHEMA"].split(".")
wh = "4dfab06c923fe3cc"
table = f"{catalog}.{schema}.scores3380ed"

def q(sql):
    r = w.statement_execution.execute_statement(warehouse_id=wh, statement=sql, wait_timeout="50s")
    return r.result.data_array if r.result else None

# columns
ti = w.tables.get(table)
print("COLUMNS:", [(c.name, c.type_name.value) for c in ti.columns], flush=True)
print("PK constraints:", [(con.primary_key_constraint.name, con.primary_key_constraint.child_columns) for con in (ti.table_constraints or []) if con.primary_key_constraint], flush=True)
print("ROW/DISTINCT/NULLS:", q(f"SELECT COUNT(*), COUNT(DISTINCT account_id), SUM(CASE WHEN score IS NULL THEN 1 ELSE 0 END) FROM {table}"), flush=True)
print("SOURCE accounts:", q(f"SELECT COUNT(DISTINCT account_id) FROM read_files('/Volumes/{catalog}/{schema}/ingest/feature_history.csv', format=>'csv', header=>true)"), flush=True)
