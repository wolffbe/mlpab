import os
import base64
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.workspace import ImportFormat, Language

w = WorkspaceClient()
me = w.current_user.me().user_name
prefix = os.environ['MLPAB_DATABRICKS_PREFIX']
schema = os.environ['MLPAB_DATABRICKS_SCHEMA']
nb_dir = f'/Users/{me}/{prefix}'
store_name = f'{prefix}-online-store'

nb_lines = [
    "# Databricks notebook source",
    f"schema = '{schema}'",
    f"store_name = '{store_name}'",
    "",
    "results = []",
    "",
    "# Test FeatureEngineeringClient",
    "try:",
    "    from databricks.feature_engineering import FeatureEngineeringClient",
    "    fe = FeatureEngineeringClient()",
    "    fe_methods = [m for m in dir(fe) if not m.startswith('_')]",
    "    results.append(f'FEC methods: {fe_methods}')",
    "except Exception as e:",
    "    results.append(f'FEC error: {e}')",
    "",
    "# Test 2-part name",
    "from databricks.sdk import WorkspaceClient",
    "from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode",
    "w2 = WorkspaceClient()",
    "try:",
    "    r = w2.feature_store.publish_table(",
    "        source_table_name='mlpab30eb4e.predictions7b586d',",
    "        publish_spec=PublishSpec(",
    "            online_store=store_name,",
    "            online_table_name='predictions7b586d',",
    "            publish_mode=PublishSpecPublishMode.TRIGGERED,",
    "        )",
    "    )",
    "    results.append(f'2part success: {r}')",
    "except Exception as e:",
    "    results.append(f'2part error: {e}')",
    "",
    "dbutils.notebook.exit('|||'.join(results))",
]
nb_content = "\n".join(nb_lines)

nb_path = f'{nb_dir}/fe_test2_nb'
w.workspace.import_(
    path=nb_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=base64.b64encode(nb_content.encode()).decode(),
    overwrite=True
)
print('Uploaded:', nb_path)
