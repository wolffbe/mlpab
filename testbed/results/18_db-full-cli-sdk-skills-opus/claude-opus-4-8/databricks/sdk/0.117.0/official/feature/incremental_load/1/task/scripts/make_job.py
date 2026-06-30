import sys, json
sys.path.insert(0, 'scripts')
from common import *
from databricks.sdk.service import sql as sqlsvc
from databricks.sdk.service import jobs

JOB_NAME = f'{PREFIX}_incrementaljobb48074'
TABLE = f'{CAT}.{SCH}.incrementalb48074'
VOL_PATH = f'/Volumes/{CAT}/{SCH}/incr_files'
FOLDER = f'/Users/{USER}/{PREFIX}'

# workspace folder for query/notebook assets
try:
    w.workspace.mkdirs(FOLDER)
except Exception as e:
    print('mkdirs:', str(e)[:120])

# Idempotent incremental ingestion query: COPY INTO only loads files not yet loaded,
# so each daily run picks up newly-arrived increment_*.csv files.
COPY_SQL = f"""COPY INTO {TABLE}
FROM (
  SELECT row_id::STRING, account_id::STRING, event_time::BIGINT, amount::DOUBLE, category::STRING
  FROM '{VOL_PATH}'
)
FILEFORMAT = CSV
PATTERN = 'increment_*.csv'
FORMAT_OPTIONS ('header'='true', 'inferSchema'='false')
COPY_OPTIONS ('mergeSchema'='false')"""

q = w.queries.create(query=sqlsvc.CreateQueryRequestQuery(
    display_name=f'{PREFIX}_incrementaljobb48074_copyinto',
    query_text=COPY_SQL,
    warehouse_id=WH,
    parent_path=FOLDER,
))
print('query id:', q.id)

# Recurring job: ONE sql task running the COPY INTO, with a DAILY cron schedule attached.
task = jobs.Task(
    task_key='ingest_increments',
    sql_task=jobs.SqlTask(
        query=jobs.SqlTaskQuery(query_id=q.id),
        warehouse_id=WH,
    ),
)
schedule = jobs.CronSchedule(
    quartz_cron_expression='0 0 6 * * ?',   # every day at 06:00
    timezone_id='UTC',
    pause_status=jobs.PauseStatus.UNPAUSED,
)
created = w.jobs.create(
    name=JOB_NAME,
    tasks=[task],
    schedule=schedule,
    max_concurrent_runs=1,
)
print('job id:', created.job_id)

# verify schedule is attached
j = w.jobs.get(created.job_id)
print('job name:', j.settings.name)
print('schedule:', j.settings.schedule)
