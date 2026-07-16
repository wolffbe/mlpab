"""Add primary key constraint and create online table via feature store."""
import time
from databricks.sdk import WorkspaceClient
import databricks.sdk.service.ml as ml_svc

w = WorkspaceClient()
CATALOG = 'workspace'
SCHEMA = 'mlpab9db404'
TABLE_NAME = 'incremental3526e9'
FULL_TABLE = f'{CATALOG}.{SCHEMA}.{TABLE_NAME}'
WH_ID = '4dfab06c923fe3cc'
PREFIX = 'mlpab9db404'
ONLINE_STORE_NAME = f'{PREFIX}-online-store'
ONLINE_TABLE_NAME = f'{CATALOG}.{SCHEMA}.{TABLE_NAME}_serving'


def run_sql(statement, timeout='30s'):
    resp = w.statement_execution.execute_statement(
        statement=statement,
        warehouse_id=WH_ID,
        wait_timeout=timeout,
        catalog=CATALOG,
        schema=SCHEMA,
    )
    stmt_id = resp.statement_id
    while resp.status and resp.status.state and resp.status.state.value in ('PENDING', 'RUNNING'):
        time.sleep(2)
        resp = w.statement_execution.get_statement(stmt_id)
    state = resp.status.state.value if resp.status and resp.status.state else 'UNKNOWN'
    if state != 'SUCCEEDED':
        raise RuntimeError(f'SQL failed (state={state}): {resp.status.error}')
    return resp


# Step 1: Add primary key constraint to allow online publishing
print(f'Adding primary key constraint to {FULL_TABLE}...')
try:
    run_sql(f'ALTER TABLE {FULL_TABLE} ADD CONSTRAINT pk_row_id PRIMARY KEY (row_id)')
    print('Primary key constraint added.')
except Exception as e:
    if 'already exists' in str(e).lower() or 'ALREADY_EXISTS' in str(e):
        print('Primary key constraint already exists.')
    else:
        print(f'PK note: {e}')


# Step 2: Publish to online feature store
print(f'\nPublishing {FULL_TABLE} to online store {ONLINE_STORE_NAME}...')
print(f'Online table name: {ONLINE_TABLE_NAME}')
try:
    result = w.feature_store.publish_table(
        source_table_name=FULL_TABLE,
        publish_spec=ml_svc.PublishSpec(
            online_store=ONLINE_STORE_NAME,
            online_table_name=ONLINE_TABLE_NAME,
            publish_mode=ml_svc.PublishSpecPublishMode.TRIGGERED,
        ),
    )
    print(f'Published successfully: {result}')
except Exception as e:
    if 'already exists' in str(e).lower() or 'ALREADY_EXISTS' in str(e):
        print('Online table already published.')
    else:
        print(f'Publish error: {e}')


print('\nDone.')
