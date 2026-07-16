import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import (
    Project, ProjectSpec,
    SyncedTable, SyncedTableSyncedTableSpec,
    SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy,
    NewPipelineSpec,
)
w = WorkspaceClient()

SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]   # workspace.mlpab7c79f3
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]    # mlpab7c79f3
catalog, schema_name = SCHEMA.split(".")
TABLE_NAME = "accountse81ff1"
FULL_TABLE_NAME = f"{catalog}.{schema_name}.{TABLE_NAME}"

# Project ID must be RFC 1123: lowercase letters, numbers, hyphens
PROJECT_ID = PREFIX  # mlpab7c79f3

print(f"Project ID: {PROJECT_ID}")
print(f"Source table: {FULL_TABLE_NAME}")

# ── 1. Create (or verify) Lakebase Project ────────────────────────────────────
print("\n[1] Creating Lakebase project...")
try:
    op = w.postgres.create_project(
        project=Project(spec=ProjectSpec(display_name=f"{PREFIX} Feature Store")),
        project_id=PROJECT_ID,
    )
    print(f"  Project creation started, waiting...")
    project = op.wait()
    print(f"  Project ready: {project.name}, state: {project.status}")
except Exception as e:
    print(f"  Project creation error (may already exist): {e}")
    try:
        project = w.postgres.get_project(f"projects/{PROJECT_ID}")
        print(f"  Existing project: {project.name}")
    except Exception as e2:
        print(f"  Could not get project: {e2}")
        raise

# ── 2. Get the production branch ─────────────────────────────────────────────
print("\n[2] Getting production branch...")
branches = list(w.postgres.list_branches(f"projects/{PROJECT_ID}"))
for b in branches:
    print(f"  Branch: {b.name} state={b.status}")
branch_name = f"projects/{PROJECT_ID}/branches/production"
print(f"  Using branch: {branch_name}")

# ── 3. Create synced table for low-latency lookup ─────────────────────────────
synced_table_id = f"{catalog}.{schema_name}.{TABLE_NAME}_online"
print(f"\n[3] Creating synced table: {synced_table_id}...")
try:
    op = w.postgres.create_synced_table(
        synced_table=SyncedTable(
            spec=SyncedTableSyncedTableSpec(
                branch=branch_name,
                postgres_database="databricks-postgres",
                source_table_full_name=FULL_TABLE_NAME,
                primary_key_columns=["row_id"],
                timeseries_key="updated_at",
                scheduling_policy=SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy.TRIGGERED,
                create_database_objects_if_missing=True,
                new_pipeline_spec=NewPipelineSpec(
                    storage_catalog=catalog,
                    storage_schema=schema_name,
                ),
            )
        ),
        synced_table_id=synced_table_id,
    )
    print(f"  Synced table creation started, waiting (up to 20 min)...")
    synced_table = op.wait()
    print(f"  Synced table ready: {synced_table.name}")
    print(f"  Status: {synced_table.status}")
except Exception as e:
    print(f"  Synced table creation error: {e}")
    try:
        existing = w.postgres.get_synced_table(f"synced_tables/{synced_table_id}")
        print(f"  Existing synced table: {existing.name}")
        print(f"  Status: {existing.status}")
    except Exception as e2:
        print(f"  Could not retrieve synced table: {e2}")

print("\n=== ONLINE TABLE SETUP DONE ===")
print(f"Delta table (offline):  {FULL_TABLE_NAME}")
print(f"Synced table (online):  {synced_table_id} (Lakebase project: {PROJECT_ID})")
