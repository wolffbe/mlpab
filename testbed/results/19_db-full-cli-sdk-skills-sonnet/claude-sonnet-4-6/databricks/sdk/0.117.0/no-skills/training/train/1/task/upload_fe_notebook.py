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

# Try different approaches for online feature publishing
nb_lines = [
    "# Databricks notebook source",
    f"schema = '{schema}'",
    f"store_name = '{store_name}'",
    "",
    "# Test different ways to enable online access",
    "",
    "# 1. Try FeatureEngineeringClient",
    "try:",
    "    from databricks.feature_engineering import FeatureEngineeringClient",
    "    fe = FeatureEngineeringClient()",
    "    print('FeatureEngineeringClient available:', dir(fe))",
    "except Exception as e:",
    "    print('FeatureEngineeringClient error:', e)",
    "",
    "# 2. Try feature_store.publish_table with 2-part name",
    "from databricks.sdk import WorkspaceClient",
    "from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode",
    "w2 = WorkspaceClient()",
    "",
    "table_2part = f'mlpab30eb4e.predictions7b586d'",
    "print('Trying 2-part name:', table_2part)",
    "try:",
    "    result = w2.feature_store.publish_table(",
    "        source_table_name=table_2part,",
    "        publish_spec=PublishSpec(",
    "            online_store=store_name,",
    "            online_table_name='predictions7b586d',",
    "            publish_mode=PublishSpecPublishMode.TRIGGERED,",
    "        )",
    "    )",
    "    print('2-part success:', result)",
    "except Exception as e:",
    "    print('2-part error:', e)",
]
nb_content = "\n".join(nb_lines)

nb_path = f'{nb_dir}/fe_test_nb'
w.workspace.import_(
    path=nb_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=base64.b64encode(nb_content.encode()).decode(),
    overwrite=True
)
print('Uploaded notebook:', nb_path)
