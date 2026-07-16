from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat, Language
import base64, time

w = WorkspaceClient()
user = 'benedict@logicalclocks.com'
prefix = 'mlpab88e583'

nb_content = r"""# Databricks notebook source
# Check feature engineering table status
SCHEMA = "workspace.mlpab88e583"

# Check if FeatureEngineeringClient is available
try:
    from databricks.feature_engineering import FeatureEngineeringClient
    fe = FeatureEngineeringClient()
    print("FeatureEngineeringClient available")

    # Check feature tables
    try:
        ft = fe.get_table(f"{SCHEMA}.ccpred739ee9")
        print(f"ccpred739ee9 is a feature table: {ft}")
    except Exception as e:
        print(f"ccpred739ee9 not a feature table: {e}")

    try:
        ft2 = fe.get_table(f"{SCHEMA}.ccpred739ee9_v2")
        print(f"ccpred739ee9_v2 is a feature table: {ft2}")
    except Exception as e:
        print(f"ccpred739ee9_v2 not a feature table: {e}")

    # Try to create feature table from v2 table (which has NOT NULL pk)
    print("Creating feature table from ccpred739ee9_v2...")
    pred_v2 = spark.table(f"{SCHEMA}.ccpred739ee9_v2")

    try:
        ft = fe.create_table(
            name=f"{SCHEMA}.ccpred739ee9",
            primary_keys=["transaction_id"],
            schema=pred_v2.schema,
            description="Fraud prediction probabilities"
        )
        print(f"Feature table created: {ft}")
        # Now write the data
        fe.write_table(
            name=f"{SCHEMA}.ccpred739ee9",
            df=pred_v2,
            mode="overwrite"
        )
        print("Data written to feature table")
    except Exception as e:
        print(f"Create/write error: {e}")

except ImportError as e:
    print(f"FeatureEngineeringClient not available: {e}")

# Show final state
spark.sql(f"SHOW TABLES IN {SCHEMA}").show(20)
spark.sql(f"SELECT COUNT(*) FROM {SCHEMA}.ccpred739ee9").show()

dbutils.notebook.exit("CHECK DONE")
"""

nb_path = f'/Users/{user}/{prefix}/ccfraud_check_fe'
encoded = base64.b64encode(nb_content.encode()).decode()
w.workspace.import_(
    path=nb_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=encoded,
    overwrite=True
)
print(f'Notebook uploaded: {nb_path}')

from databricks.sdk.service.jobs import NotebookTask, Task, Source
task = Task(
    task_key='check_fe',
    notebook_task=NotebookTask(notebook_path=nb_path, source=Source.WORKSPACE),
    timeout_seconds=300
)
job = w.jobs.create(name=f'{prefix}_check_fe', tasks=[task])
run = w.jobs.run_now(job_id=job.job_id)
print(f'Run: {run.run_id}')

start = time.time()
while True:
    r = w.jobs.get_run(run_id=run.run_id)
    lc = r.state.life_cycle_state.value if r.state.life_cycle_state else 'UNKNOWN'
    rs = r.state.result_state.value if r.state.result_state else ''
    elapsed = int(time.time() - start)
    print(f'[{elapsed}s] {lc} {rs}')
    if lc in ('TERMINATED', 'SKIPPED', 'INTERNAL_ERROR'):
        for t in (r.tasks or []):
            if t.run_id:
                out = w.jobs.get_run_output(run_id=t.run_id)
                if out.notebook_output:
                    print('Output:', out.notebook_output.result)
                if out.error:
                    print('Error:', out.error[:3000])
        print(f'FINAL: {lc} {rs}')
        break
    time.sleep(15)
