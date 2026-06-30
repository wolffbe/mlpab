"""Add primary key constraint to scaled7ecfaf and publish to online store."""
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode
import os, time

w = WorkspaceClient()
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]
FULL_TABLE = f"{SCHEMA}.scaled7ecfaf"
WAREHOUSE_ID = "4dfab06c923fe3cc"

def exec_sql(statement, timeout=300):
    resp = w.statement_execution.execute_statement(
        statement=statement,
        warehouse_id=WAREHOUSE_ID,
        wait_timeout="50s",
    )
    stmt_id = resp.statement_id
    start = time.time()
    while resp.status.state in (StatementState.PENDING, StatementState.RUNNING):
        if time.time() - start > timeout:
            raise TimeoutError(f"SQL timed out after {timeout}s")
        time.sleep(3)
        resp = w.statement_execution.get_statement(stmt_id)
    if resp.status.state not in (StatementState.SUCCEEDED,):
        raise RuntimeError(f"SQL failed ({resp.status.state}): {resp.status.error}")
    return resp

# Add primary key constraint
print("Adding primary key constraint...")
try:
    exec_sql(f"ALTER TABLE {FULL_TABLE} ADD CONSTRAINT pk_scaled7ecfaf PRIMARY KEY (row_id)")
    print("Primary key added successfully")
except Exception as e:
    print(f"PK error: {e}")

# Verify with DESCRIBE
resp = exec_sql(f"DESCRIBE TABLE EXTENDED {FULL_TABLE}")
print("Table details:")
for row in (resp.result.data_array or []):
    print(f"  {row}")

# Now publish to the online store
store_name = f"{PREFIX}-online-store"
online_table_name = f"{SCHEMA}.scaled7ecfaf_online"
print(f"\nPublishing {FULL_TABLE} to online store {store_name}...")

try:
    resp = w.feature_store.publish_table(
        source_table_name=FULL_TABLE,
        publish_spec=PublishSpec(
            online_store=store_name,
            online_table_name=online_table_name,
            publish_mode=PublishSpecPublishMode.SNAPSHOT,
        )
    )
    print(f"Published: {resp}")
except Exception as e:
    print(f"Publish error: {e}")
