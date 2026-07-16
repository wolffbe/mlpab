import databricks.sdk
from databricks.sdk.service.workspace import ImportFormat, Language
import io

w = databricks.sdk.WorkspaceClient()

# Create a notebook that will load, deduplicate, and save the data
notebook_content = """# Load and deduplicate accounts data

# Read all three batch files from workspace
batch1 = spark.read.csv("/Workspace/Users/benedict@hopsworks.ai/mlpabc1ee89/batch_1.csv", header=True, inferSchema=True)
batch2 = spark.read.csv("/Workspace/Users/benedict@hopsworks.ai/mlpabc1ee89/batch_2.csv", header=True, inferSchema=True)
batch3 = spark.read.csv("/Workspace/Users/benedict@hopsworks.ai/mlpabc1ee89/batch_3.csv", header=True, inferSchema=True)

# Union all batches
all_data = batch1.union(batch2).union(batch3)

# Deduplicate: keep only the latest revision per row_id based on updated_at
from pyspark.sql.window import Window
from pyspark.sql.functions import row_number, col

window_spec = Window.partitionBy("row_id").orderBy(col("updated_at").desc())
deduped = all_data.withColumn("rank", row_number().over(window_spec)) \\
                    .filter(col("rank") == 1) \\
                    .drop("rank")

# Write to the feature table
deduped.write.mode("overwrite").saveAsTable("workspace.mlpabc1ee89.accounts9ad208")

# Verify count
print(f"Total rows after deduplication: {deduped.count()}")
print(f"Unique row_ids: {deduped.select('row_id').distinct().count()}")
"""

# Create the notebook
notebook_path = "/Users/benedict@hopsworks.ai/mlpabc1ee89/load_accounts"

# Upload notebook
w.workspace.upload(
    notebook_path,
    io.StringIO(notebook_content).encode('utf-8'),
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    overwrite=True
)

print(f"Notebook created at {notebook_path}")

# Now run the notebook
print("Running notebook...")
result = w.jobs.run_now(
    notebook_task={
        "notebook_path": notebook_path
    },
    existing_cluster_id=None,  # Use serverless
    timeout_seconds=300
)

print(f"Run ID: {result.run_id}")
print(f"Run page URL: {result.run_page_url}")
print(f"Run name: {result.run_name}")

# Wait for completion
run = w.jobs.get_run(result.run_id)
print(f"Run state: {run.state}")
print(f"Run result state: {run.result_state}")
