import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState

w = WorkspaceClient()
schema_fqn = os.environ['MLPAB_DATABRICKS_SCHEMA']
catalog_name, schema_name = schema_fqn.split('.')
wh_id = '4dfab06c923fe3cc'
table_name = 'transactions9dd1da'
full_table = f'{catalog_name}.{schema_name}.{table_name}'


def run_sql(sql, desc=""):
    label = desc or sql[:80]
    print(f"Running: {label}")
    resp = w.statement_execution.execute_statement(
        statement=sql,
        warehouse_id=wh_id,
        catalog=catalog_name,
        schema=schema_name,
        wait_timeout='50s'
    )
    while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        time.sleep(5)
        resp = w.statement_execution.get_statement(resp.statement_id)
    if resp.status.state == StatementState.SUCCEEDED:
        print(f"  OK")
        if resp.result and resp.result.data_array:
            print(f"  Result: {resp.result.data_array[:3]}")
    else:
        print(f"  FAILED: {resp.status.error}")
        raise RuntimeError(f"SQL failed: {resp.status.error}")
    return resp


# Add primary key constraint
run_sql(
    f"ALTER TABLE {full_table} ADD CONSTRAINT pk_transactions9dd1da PRIMARY KEY (row_id)",
    "Add primary key constraint"
)

# Verify table definition
run_sql(f"DESCRIBE TABLE EXTENDED {full_table}", "Describe table")
