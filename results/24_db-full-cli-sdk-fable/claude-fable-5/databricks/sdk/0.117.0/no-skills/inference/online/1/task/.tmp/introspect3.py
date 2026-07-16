import inspect

import databricks.sdk

w = databricks.sdk.WorkspaceClient()
fns = [
    w.feature_store.create_online_store,
    w.feature_store.publish_table,
    w.feature_store.get_online_store,
    w.online_tables.create,
    w.database.create_database_instance,
    w.database.create_synced_database_table,
    w.database.create_database_catalog,
    w.database.generate_database_credential,
    w.serving_endpoints.query,
    w.statement_execution.execute_statement,
]
for f in fns:
    print("=" * 80)
    print(f.__qualname__, inspect.signature(f))
    doc = f.__doc__ or ""
    print(doc[:1500])
