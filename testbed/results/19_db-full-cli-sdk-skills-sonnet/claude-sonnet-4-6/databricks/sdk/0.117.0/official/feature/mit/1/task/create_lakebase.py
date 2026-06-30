"""Create Lakebase project and synced table for online feature serving."""
import os
import time

from databricks.sdk import WorkspaceClient
from databricks.sdk.service.postgres import (
    Project, ProjectSpec, InitialEndpointSpec, EndpointGroupSpec,
    Database, DatabaseDatabaseSpec,
    SyncedTable, SyncedTableSyncedTableSpec,
    SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy,
)

w = WorkspaceClient()
schema = os.environ['MLPAB_DATABRICKS_SCHEMA']
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
catalog, schema_name = schema.split('.')
table_name = f"{catalog}.{schema_name}.featuresb1ea93"
project_id = f"{prefix}-feat"

print(f"Schema: {schema}")
print(f"Prefix: {prefix}")
print(f"Project ID: {project_id}")


def wait_for_branch(branch_name, timeout=600):
    """Wait for a branch to become active."""
    start = time.time()
    last_state = None
    while time.time() - start < timeout:
        try:
            branch = w.postgres.get_branch(name=branch_name)
            status = branch.status
            st = str(status.state) if status and hasattr(status, 'state') else str(status)
        except Exception as e:
            st = f"error: {e}"
        if st != last_state:
            elapsed = int(time.time() - start)
            print(f"   [{elapsed}s] Branch status: {st}")
            last_state = st
        st_upper = st.upper()
        if any(x in st_upper for x in ['RUNNING', 'ACTIVE', 'HEALTHY', 'INIT_COMPLETED']):
            print("   Branch is ready!")
            return True
        if any(x in st_upper for x in ['FAIL', 'ERROR', 'DELETED']):
            print(f"   Branch issue: {st}")
            return False
        time.sleep(10)
    return False


# ── 1. Create Lakebase project ─────────────────────────────────────────────────
print("\n1. Creating Lakebase project...")
try:
    op = w.postgres.create_project(
        project=Project(
            spec=ProjectSpec(
                display_name=f"{prefix} Feature Store",
            ),
            initial_endpoint_spec=InitialEndpointSpec(
                group=EndpointGroupSpec(min=1, max=1),
            ),
        ),
        project_id=project_id,
    )
    print(f"   Project creation initiated: name=projects/{project_id}")
except Exception as e:
    if "already exists" in str(e).lower():
        print(f"   Project already exists: {project_id}")
    else:
        raise

# ── 2. Get the default branch and wait ────────────────────────────────────────
print("\n2. Getting default branch...")
branches = []
for i in range(30):
    try:
        branches = list(w.postgres.list_branches(parent=f"projects/{project_id}"))
        if branches:
            break
    except Exception as e:
        print(f"   Waiting for branches... ({e})")
    time.sleep(5)

if not branches:
    raise RuntimeError("No branches found after waiting")

default_branch = branches[0].name
print(f"   Default branch: {default_branch}")

print("   Waiting for branch to be ready...")
wait_for_branch(default_branch, timeout=600)

# ── 3. Create a database in the branch ────────────────────────────────────────
print("\n3. Creating database in branch...")
db_id = "featuredb"  # simple DNS-safe name
db_parent = default_branch

# Get the existing role
branch_roles = list(w.postgres.list_roles(parent=default_branch))
role_name = branch_roles[0].name if branch_roles else None
print(f"   Using role: {role_name}")

try:
    db_op = w.postgres.create_database(
        parent=db_parent,
        database=Database(
            spec=DatabaseDatabaseSpec(
                postgres_database=db_id,
                role=role_name,
            ),
        ),
        database_id=db_id,
    )
    print(f"   Database creation initiated: {db_id}")
    db_name = f"{db_parent}/databases/{db_id}"
except Exception as e:
    if "already exists" in str(e).lower():
        db_name = f"{db_parent}/databases/{db_id}"
        print(f"   Database already exists: {db_name}")
    else:
        print(f"   Database creation error: {e}")
        db_name = f"{db_parent}/databases/{db_id}"

print(f"   Database name: {db_name}")

# Wait for database to be ready
print("   Waiting for database...")
start = time.time()
last_state = None
while time.time() - start < 120:
    try:
        db = w.postgres.get_database(name=db_name)
        status = db.status
        st = str(status.state) if status and hasattr(status, 'state') else str(status)
    except Exception as e:
        st = f"error: {e}"
    if st != last_state:
        elapsed = int(time.time() - start)
        print(f"   [{elapsed}s] Database status: {st}")
        last_state = st
    st_upper = st.upper()
    if any(x in st_upper for x in ['RUNNING', 'ACTIVE', 'HEALTHY', 'READY']):
        print("   Database is ready!")
        break
    if any(x in st_upper for x in ['FAIL', 'ERROR', 'DELETED']):
        print(f"   Database issue: {st}")
        break
    time.sleep(10)

# ── 4. Create synced table ─────────────────────────────────────────────────────
print("\n4. Creating synced table for online access...")
synced_table_id = f"{catalog}.{schema_name}.featuresb1ea93_online"

try:
    op = w.postgres.create_synced_table(
        synced_table=SyncedTable(
            spec=SyncedTableSyncedTableSpec(
                source_table_full_name=table_name,
                primary_key_columns=["row_id"],
                timeseries_key="event_time",
                branch=default_branch,
                postgres_database=db_id,
                create_database_objects_if_missing=True,
                scheduling_policy=SyncedTableSyncedTableSpecSyncedTableSchedulingPolicy.TRIGGERED,
            ),
        ),
        synced_table_id=synced_table_id,
    )
    print(f"   Synced table creation initiated: {synced_table_id}")

    # Wait for synced table
    print("   Waiting for synced table to sync...")
    start = time.time()
    last_state = None
    while time.time() - start < 600:
        try:
            st_info = w.postgres.get_synced_table(name=f"syncedTables/{synced_table_id}")
            status = st_info.status
            st = str(status) if status else "UNKNOWN"
        except Exception as e_get:
            st = f"error: {e_get}"
        if st != last_state:
            elapsed = int(time.time() - start)
            print(f"   [{elapsed}s] Synced table status: {st}")
            last_state = st
        st_upper = st.upper()
        if any(x in st_upper for x in ['ONLINE', 'ACTIVE', 'SYNCED', 'READY', 'RUNNING']):
            print("   Synced table is ready!")
            break
        if any(x in st_upper for x in ['FAIL', 'OFFLINE_FAILED', 'PIPELINE_FAILED']):
            print(f"   Synced table issue: {st}")
            break
        time.sleep(20)

except Exception as e:
    print(f"   Synced table error: {e}")
    if "already exists" in str(e).lower():
        print("   (already exists - OK)")

print(f"\nAll done!")
print(f"Feature table (offline): {catalog}.{schema_name}.featuresb1ea93")
print(f"Synced table (online):   {synced_table_id}")
print(f"Lakebase project:        projects/{project_id}")
