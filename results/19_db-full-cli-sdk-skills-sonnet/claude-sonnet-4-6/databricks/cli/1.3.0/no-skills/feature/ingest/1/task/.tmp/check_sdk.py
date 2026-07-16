# Check what SDK capabilities are available
import sys
print(f"Python version: {sys.version}")

# Check databricks SDK version
try:
    import databricks.sdk
    print(f"Databricks SDK version: {databricks.sdk.__version__}")
except Exception as e:
    print(f"SDK import error: {e}")

# Check feature engineering
try:
    import databricks.feature_engineering
    print(f"Feature Engineering module: {dir(databricks.feature_engineering)}")
except Exception as e:
    print(f"Feature Engineering error: {e}")

# Check feature store
try:
    import databricks.feature_store
    print(f"Feature Store module: {dir(databricks.feature_store)}")
except Exception as e:
    print(f"Feature Store error: {e}")

# Check installed packages
import subprocess
result = subprocess.run(['pip', 'list'], capture_output=True, text=True)
for line in result.stdout.split('\n'):
    if any(x in line.lower() for x in ['feature', 'databricks', 'online']):
        print(line)
