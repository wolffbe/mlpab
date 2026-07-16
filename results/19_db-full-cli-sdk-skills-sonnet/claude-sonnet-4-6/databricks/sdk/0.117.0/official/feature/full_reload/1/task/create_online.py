import os
import datetime

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import (
    Project, ProjectSpec, InitialEndpointSpec,
    SyncedTable, SyncedTableSyncedTableSpec,
    SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy,
    NewPipelineSpec,
)
from databricks.sdk.common.lro import LroOptions

SCHEMA = os.environ['MLPAB_DATABRICKS_SCHEMA']   # workspace.mlpab229d43
PREFIX = os.environ['MLPAB_DATABRICKS_PREFIX']    # mlpab229d43
catalog, db = SCHEMA.split('.', 1)

TABLE = 'customersc31b07'
FULL_TABLE = f'{SCHEMA}.{TABLE}'
PROJECT_ID = f'{PREFIX}-feat'
BRANCH_PATH = f'projects/{PROJECT_ID}/branches/production'

w = WorkspaceClient()

# ── Create Lakebase project ───────────────────────────────────────────────────
print(f'Creating Lakebase project: {PROJECT_ID}')
proj_op = w.postgres.create_project(
    project=Project(
        initial_endpoint_spec=InitialEndpointSpec(),
        spec=ProjectSpec(
            display_name=f'{PREFIX}_featurestore',
        ),
    ),
    project_id=PROJECT_ID,
)

print('Waiting for project to be ready...')
project = proj_op.wait(opts=LroOptions(timeout=datetime.timedelta(minutes=15)))
print(f'Project ready: {project.name}')

# ── Create synced table for online / low-latency access ───────────────────────
print(f'Creating synced table for {FULL_TABLE}...')
synced_op = w.postgres.create_synced_table(
    synced_table=SyncedTable(
        spec=SyncedTableSyncedTableSpec(
            branch=BRANCH_PATH,
            source_table_full_name=FULL_TABLE,
            primary_key_columns=['row_id'],
            timeseries_key='updated_at',
            postgres_database='databricks-postgres',
            create_database_objects_if_missing=True,
            new_pipeline_spec=NewPipelineSpec(
                storage_catalog=catalog,
                storage_schema=db,
            ),
            scheduling_policy=SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy.TRIGGERED,
        ),
    ),
    synced_table_id=FULL_TABLE,
)

print('Waiting for synced table...')
synced_table = synced_op.wait(opts=LroOptions(timeout=datetime.timedelta(minutes=20)))
print(f'Synced table ready: {synced_table.name}')
print(f'  Status: {synced_table.status}')
print('All done.')
