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
    "from databricks.sdk import WorkspaceClient",
    "from databricks.sdk.service.ml import PublishSpec, PublishSpecPublishMode",
    "w2 = WorkspaceClient()",
    "",
    "table_full = f'{schema}.predictions7b586d'",
    "print('Publishing table:', table_full)",
    "print('To store:', store_name)",
    "",
    "result = w2.feature_store.publish_table(",
    "    source_table_name=table_full,",
    "    publish_spec=PublishSpec(",
    "        online_store=store_name,",
    "        online_table_name='predictions7b586d',",
    "        publish_mode=PublishSpecPublishMode.TRIGGERED,",
    "    )",
    ")",
    "print('Publish result:', result)",
]
nb_content = "\n".join(nb_lines)

nb_path = f'{nb_dir}/online_enable_nb'
w.workspace.import_(
    path=nb_path,
    format=ImportFormat.SOURCE,
    language=Language.PYTHON,
    content=base64.b64encode(nb_content.encode()).decode(),
    overwrite=True
)
print('Uploaded notebook:', nb_path)
