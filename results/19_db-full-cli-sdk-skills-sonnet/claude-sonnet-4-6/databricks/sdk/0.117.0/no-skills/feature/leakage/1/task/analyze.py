import os
import io
import base64
import json
from databricks.sdk import WorkspaceClient
from databricks.sdk.service import catalog as cat_service

w = WorkspaceClient()
schema_full = os.environ['MLPAB_DATABRICKS_SCHEMA']
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
catalog_name, schema_name = schema_full.split('.')
me = w.current_user.me()
user = me.user_name

print(f"Schema: {schema_full}, Prefix: {prefix}, User: {user}")

# Step 1: Create volume
vol_name = f"{prefix}_data"
try:
    v = w.volumes.create(
        catalog_name=catalog_name,
        schema_name=schema_name,
        name=vol_name,
        volume_type=cat_service.VolumeType.MANAGED
    )
    print(f"Volume created: {v.full_name}")
except Exception as e:
    print(f"Volume note: {e}")

vol_path = f"/Volumes/{catalog_name}/{schema_name}/{vol_name}"

# Step 2: Upload the CSV file to the volume
csv_path = "data/training_data.csv"
with open(csv_path, "rb") as f:
    csv_data = f.read()

upload_path = f"{vol_path}/training_data.csv"
try:
    w.files.upload(upload_path, io.BytesIO(csv_data), overwrite=True)
    print(f"Uploaded CSV to {upload_path}")
except Exception as e:
    print(f"Upload error: {e}")
    import traceback
    traceback.print_exc()

# Step 3: Create analysis notebook
notebook_path = f"/Users/{user}/{prefix}/leakage_analysis"

# Build the notebook content with proper escaping
notebook_lines = [
    "import pandas as pd",
    "import numpy as np",
    "import json",
    "from scipy.stats import pointbiserialr",
    "",
    f'vol_path = "{vol_path}"',
    "",
    "# Read the CSV",
    "df = spark.read.csv(vol_path + '/training_data.csv', header=True, inferSchema=True)",
    "df.show(5)",
    "",
    "# Convert to pandas",
    "pdf = df.toPandas()",
    "features = ['f1', 'f2', 'f3', 'f4', 'f5', 'f6']",
    "label = 'label'",
    "",
    "# Absolute correlations",
    "correlations = {}",
    "for f in features:",
    "    corr = abs(pdf[f].corr(pdf[label]))",
    "    correlations[f] = float(corr)",
    "    print(f'Correlation {f}: {corr:.4f}')",
    "",
    "leaking_feature = max(correlations, key=correlations.get)",
    "print(f'\\nLeaking feature: {leaking_feature} (corr={correlations[leaking_feature]:.4f})')",
    "",
    "# Point-biserial correlation",
    "pb_results = {}",
    "for f in features:",
    "    r, p = pointbiserialr(pdf[label], pdf[f])",
    "    pb_results[f] = {'r': float(r), 'p': float(p)}",
    "    print(f'Point-biserial {f}: r={r:.4f}, p={p:.10f}')",
    "",
    "result = {",
    "    'leaking_feature': leaking_feature,",
    "    'correlations': correlations,",
    "    'point_biserial': pb_results",
    "}",
    "result_json = json.dumps(result)",
    "print(f'\\nRESULT: {result_json}')",
    "",
    "# Write to volume",
    "dbutils.fs.put(vol_path + '/result.json', result_json, overwrite=True)",
    "print('Result saved.')",
]

notebook_content = "\n".join(notebook_lines)
notebook_b64 = base64.b64encode(notebook_content.encode()).decode()

# Create directory and notebook
try:
    try:
        w.workspace.mkdirs(f"/Users/{user}/{prefix}")
    except Exception:
        pass

    from databricks.sdk.service.workspace import ImportFormat, Language
    w.workspace.import_(
        path=notebook_path,
        format=ImportFormat.SOURCE,
        language=Language.PYTHON,
        content=notebook_b64,
        overwrite=True
    )
    print(f"Notebook created: {notebook_path}")
except Exception as e:
    print(f"Notebook creation error: {e}")
    import traceback
    traceback.print_exc()

# Step 4: Submit job
from databricks.sdk.service.jobs import Task, NotebookTask, Source

try:
    job_name = f"{prefix}_leakage_analysis"
    run = w.jobs.submit(
        run_name=job_name,
        tasks=[Task(
            task_key="analyze",
            notebook_task=NotebookTask(
                notebook_path=notebook_path,
                source=Source.WORKSPACE
            ),
            new_cluster={
                "spark_version": "15.4.x-scala2.12",
                "node_type_id": "m5d.large",
                "num_workers": 1
            }
        )]
    )
    run_id = run.run_id
    print(f"Job submitted, run_id: {run_id}")
    with open("run_id.txt", "w") as fout:
        fout.write(str(run_id))

except Exception as e:
    print(f"Job submission error: {e}")
    import traceback
    traceback.print_exc()
