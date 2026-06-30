"""
Create a point-in-time correct training dataset on Databricks.
"""
import os
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.sql import StatementState
from databricks.sdk.service.catalog import VolumeType

# Config
SCHEMA = os.environ['MLPAB_DATABRICKS_SCHEMA']  # workspace.mlpab35e20e
PREFIX = os.environ['MLPAB_DATABRICKS_PREFIX']   # mlpab35e20e
CATALOG, SCHEMA_NAME = SCHEMA.split('.')
WAREHOUSE_ID = '4dfab06c923fe3cc'
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
DATASET_NAME = 'churntraining807643'


def run_sql(w, sql, warehouse_id=WAREHOUSE_ID, timeout=300):
    """Execute SQL and wait for completion, returning result."""
    print(f"Running SQL: {sql[:100]}...")
    resp = w.statement_execution.execute_statement(
        warehouse_id=warehouse_id,
        statement=sql,
    )
    stmt_id = resp.statement_id

    # Poll until done
    start = time.time()
    while True:
        status = w.statement_execution.get_statement(stmt_id)
        state = status.status.state
        if state == StatementState.SUCCEEDED:
            print(f"  -> SUCCEEDED")
            return status
        elif state in (StatementState.FAILED, StatementState.CANCELED, StatementState.CLOSED):
            error = status.status.error
            raise RuntimeError(f"SQL failed ({state}): {error}")
        elif time.time() - start > timeout:
            raise TimeoutError(f"SQL timed out after {timeout}s")
        time.sleep(3)


def upload_csv_to_volume(w, local_path, volume_path):
    """Upload a CSV file to a Databricks volume."""
    print(f"Uploading {local_path} -> {volume_path}")
    with open(local_path, 'rb') as f:
        w.files.upload(volume_path, f, overwrite=True)
    print(f"  -> Uploaded")


def main():
    w = WorkspaceClient()
    print(f"Connected to: {w.config.host}")
    print(f"Schema: {SCHEMA}")

    # Step 1: Create a volume for CSV staging
    volume_name = f"{PREFIX}_staging"
    print(f"\nCreating volume {CATALOG}.{SCHEMA_NAME}.{volume_name}...")
    try:
        w.volumes.create(
            catalog_name=CATALOG,
            schema_name=SCHEMA_NAME,
            name=volume_name,
            volume_type=VolumeType.MANAGED,
        )
        print("  -> Volume created")
    except Exception as e:
        if 'already exists' in str(e).lower():
            print("  -> Volume already exists")
        else:
            raise

    volume_path_base = f"/Volumes/{CATALOG}/{SCHEMA_NAME}/{volume_name}"

    # Step 2: Upload CSVs to the volume
    csv_files = {
        'transactions': 'transactions.csv',
        'transactions_late': 'transactions_late.csv',
        'profiles': 'profiles.csv',
        'activity': 'activity.csv',
        'account_health': 'account_health.csv',
        'labels': 'labels.csv',
    }

    for table_key, filename in csv_files.items():
        local_path = os.path.join(DATA_DIR, filename)
        volume_path = f"{volume_path_base}/{filename}"
        upload_csv_to_volume(w, local_path, volume_path)

    # Step 3: Create staging tables from CSVs using CTAS with read_files
    print("\nCreating staging tables...")

    # Transactions (combined from both files using glob)
    run_sql(w, f"""
        CREATE OR REPLACE TABLE {SCHEMA}.transactions_raw AS
        SELECT account_id,
               CAST(event_time AS BIGINT) AS event_time,
               CAST(amount AS DOUBLE) AS amount,
               CAST(balance AS DOUBLE) AS balance
        FROM read_files('{volume_path_base}/transactions*.csv',
                        format => 'csv', header => 'true')
    """)

    run_sql(w, f"""
        CREATE OR REPLACE TABLE {SCHEMA}.profiles_raw AS
        SELECT account_id,
               CAST(event_time AS BIGINT) AS event_time,
               CAST(credit_score AS INT) AS credit_score,
               tier
        FROM read_files('{volume_path_base}/profiles.csv',
                        format => 'csv', header => 'true')
    """)

    run_sql(w, f"""
        CREATE OR REPLACE TABLE {SCHEMA}.activity_raw AS
        SELECT account_id,
               CAST(event_time AS BIGINT) AS event_time,
               CAST(sessions_7d AS INT) AS sessions_7d
        FROM read_files('{volume_path_base}/activity.csv',
                        format => 'csv', header => 'true')
    """)

    run_sql(w, f"""
        CREATE OR REPLACE TABLE {SCHEMA}.account_health_raw AS
        SELECT account_id,
               CAST(event_time AS BIGINT) AS event_time,
               CAST(health_score AS DOUBLE) AS health_score
        FROM read_files('{volume_path_base}/account_health.csv',
                        format => 'csv', header => 'true')
    """)

    run_sql(w, f"""
        CREATE OR REPLACE TABLE {SCHEMA}.labels_raw AS
        SELECT account_id,
               CAST(label_time AS BIGINT) AS label_time,
               CAST(churned AS INT) AS churned
        FROM read_files('{volume_path_base}/labels.csv',
                        format => 'csv', header => 'true')
    """)

    # Step 4: Create the point-in-time correct training dataset
    print(f"\nCreating training dataset {SCHEMA}.{DATASET_NAME}...")

    # Point-in-time join: for each (account_id, label_time), get most recent
    # feature value with event_time <= label_time
    run_sql(w, f"""
        CREATE OR REPLACE TABLE {SCHEMA}.{DATASET_NAME} AS
        WITH labels AS (
            SELECT account_id, label_time, churned
            FROM {SCHEMA}.labels_raw
        ),
        -- Get most recent transaction at or before label_time
        txn_ranked AS (
            SELECT t.account_id, t.event_time, t.amount, t.balance, l.label_time,
                   ROW_NUMBER() OVER (
                       PARTITION BY t.account_id, l.label_time
                       ORDER BY t.event_time DESC
                   ) AS rn
            FROM {SCHEMA}.transactions_raw t
            JOIN labels l ON t.account_id = l.account_id
            WHERE t.event_time <= l.label_time
        ),
        txn_pit AS (
            SELECT account_id, label_time, amount, balance
            FROM txn_ranked WHERE rn = 1
        ),
        -- Get most recent profile at or before label_time
        prof_ranked AS (
            SELECT p.account_id, p.event_time, p.credit_score, p.tier, l.label_time,
                   ROW_NUMBER() OVER (
                       PARTITION BY p.account_id, l.label_time
                       ORDER BY p.event_time DESC
                   ) AS rn
            FROM {SCHEMA}.profiles_raw p
            JOIN labels l ON p.account_id = l.account_id
            WHERE p.event_time <= l.label_time
        ),
        prof_pit AS (
            SELECT account_id, label_time, credit_score, tier
            FROM prof_ranked WHERE rn = 1
        ),
        -- Get most recent activity at or before label_time
        act_ranked AS (
            SELECT a.account_id, a.event_time, a.sessions_7d, l.label_time,
                   ROW_NUMBER() OVER (
                       PARTITION BY a.account_id, l.label_time
                       ORDER BY a.event_time DESC
                   ) AS rn
            FROM {SCHEMA}.activity_raw a
            JOIN labels l ON a.account_id = l.account_id
            WHERE a.event_time <= l.label_time
        ),
        act_pit AS (
            SELECT account_id, label_time, sessions_7d
            FROM act_ranked WHERE rn = 1
        ),
        -- Get most recent health at or before label_time
        health_ranked AS (
            SELECT h.account_id, h.event_time, h.health_score, l.label_time,
                   ROW_NUMBER() OVER (
                       PARTITION BY h.account_id, l.label_time
                       ORDER BY h.event_time DESC
                   ) AS rn
            FROM {SCHEMA}.account_health_raw h
            JOIN labels l ON h.account_id = l.account_id
            WHERE h.event_time <= l.label_time
        ),
        health_pit AS (
            SELECT account_id, label_time, health_score
            FROM health_ranked WHERE rn = 1
        )
        SELECT
            l.account_id,
            l.label_time,
            t.amount,
            t.balance,
            p.credit_score,
            p.tier,
            a.sessions_7d,
            h.health_score,
            l.churned
        FROM labels l
        LEFT JOIN txn_pit t ON l.account_id = t.account_id AND l.label_time = t.label_time
        LEFT JOIN prof_pit p ON l.account_id = p.account_id AND l.label_time = p.label_time
        LEFT JOIN act_pit a ON l.account_id = a.account_id AND l.label_time = a.label_time
        LEFT JOIN health_pit h ON l.account_id = h.account_id AND l.label_time = h.label_time
        ORDER BY l.account_id, l.label_time
    """)

    # Step 5: Verify the result
    print("\nVerifying result...")
    result = run_sql(w, f"""
        SELECT COUNT(*) as row_count FROM {SCHEMA}.{DATASET_NAME}
    """)

    # Get row count from result
    if result.result and result.result.data_array:
        row_count = result.result.data_array[0][0]
        print(f"  -> Row count: {row_count}")

    result = run_sql(w, f"""
        SELECT * FROM {SCHEMA}.{DATASET_NAME} LIMIT 5
    """)
    if result.result and result.result.data_array:
        print("\nSample rows:")
        for row in result.result.data_array:
            print(f"  {row}")

    # Step 6: Get table history to confirm version
    result = run_sql(w, f"""
        DESCRIBE HISTORY {SCHEMA}.{DATASET_NAME}
    """)
    if result.result and result.result.data_array:
        print("\nTable history:")
        for row in result.result.data_array[:3]:
            print(f"  version={row[0]}, operation={row[2]}")

    print(f"\nDone! Training dataset '{DATASET_NAME}' created in {SCHEMA}")


if __name__ == '__main__':
    main()
