# Databricks notebook source
# Check available modules and publish feature table to online store

# COMMAND ----------
import subprocess
result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
# Find relevant packages
for line in result.stdout.split('\n'):
    if 'feature' in line.lower() or 'databricks' in line.lower():
        print(line)

# COMMAND ----------
# Try classic feature store SDK
try:
    from databricks import feature_store
    print(f"feature_store version: {feature_store.__version__}")
except Exception as e:
    print(f"classic feature_store error: {e}")

try:
    import databricks.feature_store as dfs
    print(f"dfs: {dir(dfs)}")
except Exception as e:
    print(f"import error: {e}")

# COMMAND ----------
# Try to use feature store client
try:
    from databricks.feature_store import FeatureStoreClient
    fs = FeatureStoreClient()
    print("FeatureStoreClient available")

    # Try to register the UC table
    ft = fs.register_table(
        delta_table="workspace.mlpab0442b8.accountse81ff1",
        primary_keys=["row_id"],
        timestamp_keys=["updated_at"],
        description="Accounts feature table"
    )
    print(f"Registered: {ft}")
except Exception as e:
    print(f"FeatureStoreClient error: {e}")

# COMMAND ----------
# Check if table exists in feature store
try:
    from databricks.feature_store import FeatureStoreClient
    fs = FeatureStoreClient()
    ft = fs.get_table("workspace.mlpab0442b8.accountse81ff1")
    print(f"Feature table: {ft}")
    print(f"Primary keys: {ft.primary_keys}")
    print(f"Timestamp keys: {ft.timestamp_keys}")
except Exception as e:
    print(f"get_table error: {e}")

# COMMAND ----------
# Try publish to online store
try:
    from databricks.feature_store import FeatureStoreClient
    from databricks.feature_store.online_store_spec import AmazonDynamoDBSpec
    fs = FeatureStoreClient()

    online_store = AmazonDynamoDBSpec(
        region="us-east-1",
        write_secret_prefix="",
        read_secret_prefix="",
        table_name="accountse81ff1"
    )
    fs.publish_table(
        name="workspace.mlpab0442b8.accountse81ff1",
        online_store=online_store,
        mode="overwrite"
    )
    print("Published successfully!")
except Exception as e:
    print(f"publish error: {e}")

# COMMAND ----------
# Final: check what's available
import pkgutil
import databricks
for importer, modname, ispkg in pkgutil.walk_packages(
        path=databricks.__path__,
        prefix=databricks.__name__+'.',
        onerror=lambda x: None):
    print(modname)
