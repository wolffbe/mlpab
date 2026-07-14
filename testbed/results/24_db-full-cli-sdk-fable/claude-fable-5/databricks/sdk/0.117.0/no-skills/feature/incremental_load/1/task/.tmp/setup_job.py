import io, os
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat, Language
from databricks.sdk.service import jobs

w = WorkspaceClient()
catalog, schema = os.environ["MLPAB_DATABRICKS_SCHEMA"].split(".")
prefix = os.environ["MLPAB_DATABRICKS_PREFIX"]
TABLE = f"{catalog}.{schema}.incrementalf3c1bf"
ONLINE = f"{TABLE}_online"
VOLUME_PATH = f"/Volumes/{catalog}/{schema}/raw_increments/events"
me = w.current_user.me().user_name
NB_DIR = f"/Users/{me}/{prefix}"
NB_PATH = f"{NB_DIR}/incremental_ingest"

notebook = f'''# Databricks notebook source
# Daily incremental ingest: load any new increment CSVs from the volume
# into the feature table. COPY INTO is idempotent per file, so already
# loaded increments are skipped and only new daily files are ingested.
spark.sql("""
COPY INTO {TABLE}
FROM (SELECT row_id, account_id, CAST(event_time AS BIGINT) AS event_time,
             CAST(amount AS DOUBLE) AS amount, category
      FROM '{VOLUME_PATH}/')
FILEFORMAT = CSV
FORMAT_OPTIONS ('header' = 'true')
""").show()
display(spark.sql("SELECT COUNT(*) AS total_rows FROM {TABLE}"))
'''

w.workspace.mkdirs(NB_DIR)
w.workspace.upload(NB_PATH, io.BytesIO(notebook.encode()),
                   format=ImportFormat.SOURCE, language=Language.PYTHON,
                   overwrite=True)
print("notebook:", NB_PATH)

st = w.database.get_synced_database_table(ONLINE)
pipeline_id = st.data_synchronization_status.pipeline_id
print("online sync pipeline:", pipeline_id)

job = w.jobs.create(
    name="incrementaljobf3c1bf",
    tasks=[
        jobs.Task(
            task_key="ingest_increments",
            notebook_task=jobs.NotebookTask(notebook_path=NB_PATH),
        ),
        jobs.Task(
            task_key="refresh_online_store",
            depends_on=[jobs.TaskDependency(task_key="ingest_increments")],
            pipeline_task=jobs.PipelineTask(pipeline_id=pipeline_id),
        ),
    ],
    schedule=jobs.CronSchedule(
        quartz_cron_expression="0 30 3 * * ?",  # daily 03:30 UTC
        timezone_id="UTC",
        pause_status=jobs.PauseStatus.UNPAUSED,
    ),
    max_concurrent_runs=1,
)
print("job_id:", job.job_id)
j = w.jobs.get(job.job_id)
print("job name:", j.settings.name)
print("schedule:", j.settings.schedule)
