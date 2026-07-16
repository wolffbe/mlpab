import databricks.sdk as dbsdk
from databricks.sdk.service import jobs, compute

w = dbsdk.WorkspaceClient()
user = w.current_user.me().user_name
prefix = "mlpab34dc41"
base_dir = "/Users/" + user + "/" + prefix
remote_path = base_dir + "/flaky62cc43.py"

job_name = prefix + "_flaky62cc43"

# Serverless environment for the python script task
env = jobs.JobEnvironment(
    environment_key="default_env",
    spec=compute.Environment(client="1"),
)

# Failure email notification mentioning the job (named ...flaky62cc43)
email_notif = jobs.JobEmailNotifications(on_failure=[user])

task = jobs.Task(
    task_key="run_flaky62cc43",
    description="Runs flaky62cc43 failing script; alerts on failure",
    spark_python_task=jobs.SparkPythonTask(
        python_file=remote_path,
        source=jobs.Source.WORKSPACE,
    ),
    environment_key="default_env",
    email_notifications=jobs.TaskEmailNotifications(on_failure=[user]),
)

created = w.jobs.create(
    name=job_name,
    tasks=[task],
    environments=[env],
    email_notifications=email_notif,
)
print("JOB_ID", created.job_id)
print("JOB_NAME", job_name)

# Run it once
run = w.jobs.run_now(job_id=created.job_id)
print("RUN_ID", run.run_id)
