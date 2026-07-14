"""
Recsys pipeline: compute top-5 recommendations per user and create feature table
with online access on Databricks.
"""
import os
import base64
import time
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import jobs, compute

# Config
SCHEMA = os.environ["MLPAB_DATABRICKS_SCHEMA"]  # workspace.mlpab2812e7
PREFIX = os.environ["MLPAB_DATABRICKS_PREFIX"]   # mlpab2812e7
CATALOG, SCHEMA_NAME = SCHEMA.split(".")
TABLE_NAME = "recs708df6"
FULL_TABLE = f"{CATALOG}.{SCHEMA_NAME}.{TABLE_NAME}"
USER = "benedict@logicalclocks.com"
NOTEBOOK_PATH = f"/Users/{USER}/{PREFIX}/recsys_build"
DATA_DIR = os.path.join(os.path.dirname(__file__), "data")

w = WorkspaceClient()

# ── 1. Read local CSVs and embed them directly in the notebook ──────────────
def read_csv_as_string(filename):
    with open(os.path.join(DATA_DIR, filename)) as f:
        return f.read()

interactions_csv = read_csv_as_string("interactions.csv")
user_emb_csv = read_csv_as_string("user_embeddings.csv")
item_emb_csv = read_csv_as_string("item_embeddings.csv")

# ── 2. Build notebook source ────────────────────────────────────────────────
notebook_source = f'''
import io
import pandas as pd
import numpy as np

# ── Inline data ──────────────────────────────────────────────────────────────
INTERACTIONS_CSV = """{interactions_csv}"""

USER_EMB_CSV = """{user_emb_csv}"""

ITEM_EMB_CSV = """{item_emb_csv}"""

interactions = pd.read_csv(io.StringIO(INTERACTIONS_CSV))
user_emb = pd.read_csv(io.StringIO(USER_EMB_CSV))
item_emb = pd.read_csv(io.StringIO(ITEM_EMB_CSV))

# ── Compute recommendations ──────────────────────────────────────────────────
emb_cols = ["e1","e2","e3","e4","e5","e6","e7","e8"]

U = user_emb[emb_cols].values          # (n_users, 8)
V = item_emb[emb_cols].values          # (n_items, 8)
scores_mat = U @ V.T                   # (n_users, n_items)

user_ids = user_emb["user_id"].tolist()
item_ids = item_emb["item_id"].tolist()
item_id_to_idx = {{iid: i for i, iid in enumerate(item_ids)}}

# Build set of already-interacted items per user
from collections import defaultdict
interacted = defaultdict(set)
for _, row in interactions.iterrows():
    interacted[row["user_id"]].add(item_id_to_idx.get(row["item_id"], -1))

rows = []
for u_idx, uid in enumerate(user_ids):
    excl = interacted[uid]
    scored = []
    for i_idx, iid in enumerate(item_ids):
        if i_idx not in excl:
            scored.append((scores_mat[u_idx, i_idx], iid))
    # Sort: descending score, then ascending item_id for ties
    scored.sort(key=lambda x: (-x[0], x[1]))
    for rank, (score, iid) in enumerate(scored[:5], start=1):
        rec_id = f"{{uid}}#{{rank}}"
        rows.append((rec_id, uid, rank, iid))

recs_df = pd.DataFrame(rows, columns=["rec_id", "user_id", "rank", "item_id"])
recs_df["rank"] = recs_df["rank"].astype(int)

print(f"Total recommendation rows: {{len(recs_df)}}")
print(recs_df.head(10))

# ── Create Spark DataFrame and write feature table ───────────────────────────
spark_df = spark.createDataFrame(recs_df)

CATALOG = "{CATALOG}"
SCHEMA_NAME = "{SCHEMA_NAME}"
TABLE_NAME = "{TABLE_NAME}"
FULL_TABLE = f"{{CATALOG}}.{{SCHEMA_NAME}}.{{TABLE_NAME}}"

# Enable change data feed (required for online table)
spark.sql(f"""
    CREATE OR REPLACE TABLE {{FULL_TABLE}} (
        rec_id   STRING  NOT NULL,
        user_id  STRING  NOT NULL,
        rank     INT     NOT NULL,
        item_id  STRING  NOT NULL,
        CONSTRAINT pk PRIMARY KEY (rec_id)
    )
    TBLPROPERTIES (
        'delta.enableChangeDataFeed' = 'true'
    )
""")

spark_df.write.mode("overwrite").option("overwriteSchema","true").saveAsTable(FULL_TABLE)

# Verify
count = spark.sql(f"SELECT COUNT(*) as n FROM {{FULL_TABLE}}").collect()[0]["n"]
print(f"Rows written: {{count}}")
spark.sql(f"SELECT * FROM {{FULL_TABLE}} ORDER BY user_id, rank LIMIT 15").show()
'''

# ── 3. Upload notebook ──────────────────────────────────────────────────────
print(f"Uploading notebook to {NOTEBOOK_PATH}")
# Ensure parent folder exists
parent_dir = f"/Users/{USER}/{PREFIX}"
try:
    w.workspace.mkdirs(path=parent_dir)
except Exception as e:
    print(f"  mkdir: {e}")

from databricks.sdk.service.workspace import ImportFormat, Language
w.workspace.import_(
    path=NOTEBOOK_PATH,
    content=base64.b64encode(notebook_source.encode()).decode(),
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    overwrite=True,
)
print("Notebook uploaded.")

# ── 4. Find smallest available cluster or create a one-time job ─────────────
# Use a job with a new cluster (serverless or smallest)
print("Creating and running job...")

JOB_NAME = f"{PREFIX}_recsys_build"

job = w.jobs.create(
    name=JOB_NAME,
    tasks=[
        jobs.Task(
            task_key="build_recs",
            notebook_task=jobs.NotebookTask(
                notebook_path=NOTEBOOK_PATH,
            ),
        )
    ],
)
job_id = job.job_id
print(f"Job created: {job_id}")

run = w.jobs.run_now(job_id=job_id)
run_id = run.run_id
print(f"Run started: {run_id}")

# ── 5. Wait for job to finish ────────────────────────────────────────────────
print("Waiting for job to complete...")
for _ in range(120):  # up to 20 minutes
    run_state = w.jobs.get_run(run_id=run_id)
    life_cycle = run_state.state.life_cycle_state
    print(f"  State: {life_cycle}")
    if life_cycle.value in ("TERMINATED", "SKIPPED", "INTERNAL_ERROR"):
        result = run_state.state.result_state
        print(f"  Result: {result}")
        if result and result.value != "SUCCESS":
            # Print output
            for task_run in run_state.tasks or []:
                try:
                    output = w.jobs.get_run_output(run_id=task_run.run_id)
                    if output.notebook_output:
                        print("Notebook output:", output.notebook_output.result)
                    if output.error:
                        print("Error:", output.error)
                    if output.error_trace:
                        print("Trace:", output.error_trace[:3000])
                except Exception as ex:
                    print(f"Could not get output: {ex}")
            raise RuntimeError(f"Job failed: {result}")
        break
    time.sleep(10)
else:
    raise TimeoutError("Job timed out after 20 minutes")

print("Job completed successfully!")

# ── 6. Create online table for low-latency access ────────────────────────────
print("Creating online table...")
from databricks.sdk.service import catalog as catalog_sdk

online_table_name = f"{FULL_TABLE}_online"
try:
    ot = w.online_tables.create(
        name=online_table_name,
        spec=catalog_sdk.OnlineTableSpec(
            source_table_full_name=FULL_TABLE,
            primary_key_columns=["rec_id"],
            run_triggered=catalog_sdk.OnlineTableSpecTriggeredSchedulingPolicy(),
        ),
    )
    print(f"Online table creation initiated: {ot.name}")
except Exception as e:
    print(f"Online table creation: {e}")
    # Try alternate API if available
    try:
        ot = w.online_tables.create(
            name=online_table_name,
            spec=catalog_sdk.OnlineTableSpec(
                source_table_full_name=FULL_TABLE,
                primary_key_columns=["rec_id"],
            ),
        )
        print(f"Online table (no schedule): {ot.name}")
    except Exception as e2:
        print(f"Alternate attempt: {e2}")

# Wait for online table to be provisioned
print("Waiting for online table to be active...")
for _ in range(60):  # up to 10 minutes
    try:
        ot_status = w.online_tables.get(name=online_table_name)
        state = ot_status.status.detailed_state if ot_status.status else "UNKNOWN"
        print(f"  Online table state: {state}")
        if hasattr(state, 'value'):
            state_str = state.value
        else:
            state_str = str(state)
        if "ACTIVE" in state_str or "ONLINE" in state_str:
            print("Online table is active!")
            break
        if "FAILED" in state_str or "ERROR" in state_str:
            print(f"Online table failed: {state_str}")
            break
    except Exception as e:
        print(f"  Status check error: {e}")
    time.sleep(10)

print("\nDone! Feature table and online table created.")
print(f"  Offline: {FULL_TABLE}")
print(f"  Online:  {online_table_name}")
