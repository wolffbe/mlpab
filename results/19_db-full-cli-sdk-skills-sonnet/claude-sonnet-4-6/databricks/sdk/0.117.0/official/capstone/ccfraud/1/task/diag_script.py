from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat, Language
import base64, time

w = WorkspaceClient()
user = 'benedict@logicalclocks.com'
prefix = 'mlpab88e583'

diag_nb = """# Databricks notebook source
import os
print("Current catalog:", spark.sql("SELECT current_catalog()").collect()[0][0])
print("Current schema:", spark.sql("SELECT current_schema()").collect()[0][0])
spark.sql("SHOW TABLES IN workspace.mlpab88e583").show(50)
try:
    spark.sql("CREATE TABLE IF NOT EXISTS workspace.mlpab88e583.test_diag (id INT, val STRING)")
    print("Created test_diag table")
    spark.sql("INSERT INTO workspace.mlpab88e583.test_diag VALUES (1, 'hello')")
    spark.sql("SELECT * FROM workspace.mlpab88e583.test_diag").show()
except Exception as e:
    print(f"Error: {e}")
dbutils.notebook.exit("DONE")
"""

nb_path = f'/Users/{user}/{prefix}/ccfraud_diag'
encoded = base64.b64encode(diag_nb.encode()).decode()
w.workspace.import_(
    path=nb_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=encoded,
    overwrite=True
)
print(f'Diagnostic notebook: {nb_path}')

from databricks.sdk.service.jobs import NotebookTask, Task, Source
task = Task(
    task_key='diag',
    notebook_task=NotebookTask(notebook_path=nb_path, source=Source.WORKSPACE),
    timeout_seconds=300
)
job = w.jobs.create(name=f'{prefix}_diag', tasks=[task])
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
                    print('Error:', out.error[:500])
                if out.error_trace:
                    print('Trace:', out.error_trace[:1000])
        break
    time.sleep(15)
