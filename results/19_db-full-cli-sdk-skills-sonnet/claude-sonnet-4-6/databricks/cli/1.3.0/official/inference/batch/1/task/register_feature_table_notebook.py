# Databricks notebook source
# COMMAND ----------
# Try to register as a Feature Store / Feature Engineering table
lines = []

try:
    from databricks.feature_engineering import FeatureEngineeringClient
    fe = FeatureEngineeringClient()
    lines.append("FeatureEngineeringClient available")

    # Check if the table is already a feature table
    try:
        ft = fe.get_table("workspace.mlpab6ef9cb.scores4f5893")
        lines.append(f"Existing feature table: {ft}")
    except Exception as e:
        lines.append(f"Not a feature table yet: {e}")

    # Try to create/register as feature table
    try:
        df = spark.sql("SELECT * FROM workspace.mlpab6ef9cb.scores4f5893")
        ft = fe.create_table(
            name="workspace.mlpab6ef9cb.scores4f5893",
            primary_keys=["account_id"],
            df=df,
            description="Batch scores for accounts"
        )
        lines.append(f"Feature table created: {ft}")
    except Exception as e:
        lines.append(f"create_table error: {type(e).__name__}: {str(e)[:300]}")

except ImportError as e:
    lines.append(f"FeatureEngineeringClient not available: {e}")

    try:
        from databricks import feature_store
        lines.append("feature_store module available")
        fs = feature_store.FeatureStoreClient()
        lines.append(f"FeatureStoreClient available")
        try:
            ft = fs.get_table("workspace.mlpab6ef9cb.scores4f5893")
            lines.append(f"Feature table info: {ft}")
        except Exception as e2:
            lines.append(f"get_table error: {e2}")
    except ImportError as e2:
        lines.append(f"feature_store not available: {e2}")

dbutils.notebook.exit("\n".join(lines))
