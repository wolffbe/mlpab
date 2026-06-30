import dlt
from pyspark.sql.functions import col

# Create a streaming/live table that mirrors the feature table for online access
@dlt.table(
    name="transactions9dd1da_online",
    comment="Online table for transactions9dd1da feature table - low-latency access",
    table_properties={
        "feature_store.source_table": "workspace.mlpabcbef07.transactions9dd1da",
        "feature_store.primary_key": "row_id",
        "feature_store.event_time_column": "event_time",
        "feature_store.type": "online_feature_table"
    }
)
def transactions_online():
    return (
        spark.read.table("workspace.mlpabcbef07.transactions9dd1da")
        .select(
            col("row_id"),
            col("account_id"),
            col("event_time"),
            col("amount"),
            col("category")
        )
    )
