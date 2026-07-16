# Create online/synced table for feature table using SDK
from databricks.sdk import WorkspaceClient
from databricks.sdk.service.catalog import OnlineTable, OnlineTableSpec, OnlineTableSpecTriggeredSchedulingPolicy
import json

w = WorkspaceClient()

# Try to create synced table via the SDK
table_name = "workspace.mlpabcbef07.transactions9dd1da_online"
source_table = "workspace.mlpabcbef07.transactions9dd1da"

try:
    online_table = w.online_tables.create(
        name=table_name,
        spec=OnlineTableSpec(
            source_table_full_name=source_table,
            primary_key_columns=["row_id"],
            timeseries_key="event_time",
            run_triggered=OnlineTableSpecTriggeredSchedulingPolicy()
        )
    )
    print(f"Online table created: {online_table}")
except Exception as e:
    print(f"Error with online_tables.create: {e}")

# Try using feature engineering client if available
try:
    from databricks.feature_engineering import FeatureEngineeringClient
    fe = FeatureEngineeringClient()
    print(f"FeatureEngineeringClient available: {dir(fe)}")
except Exception as e:
    print(f"FeatureEngineeringClient error: {e}")

# Check available SDK methods
try:
    print(f"SDK online_tables methods: {[m for m in dir(w.online_tables) if not m.startswith('_')]}")
except Exception as e:
    print(f"Error listing methods: {e}")
