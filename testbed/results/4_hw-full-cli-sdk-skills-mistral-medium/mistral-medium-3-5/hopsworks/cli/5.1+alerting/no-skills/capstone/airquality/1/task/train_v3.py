#!/usr/bin/env python3
import hopsworks

# Login
hopsworks.login()

# Try to get feature store - in 5.0.x this should work
try:
    fs = hopsworks.get_feature_store()
    print(f"Got feature store via get_feature_store: {type(fs)}")
except AttributeError as e:
    print(f"get_feature_store failed: {e}")
    # Try direct access
    try:
        fs = hopsworks.feature_store
        print(f"Got feature store via .feature_store: {type(fs)}")
    except AttributeError as e2:
        print(f".feature_store failed: {e2}")
        # Try importing directly
        try:
            from hopsworks import feature_store
            fs = feature_store
            print(f"Got feature store via import: {type(fs)}")
        except Exception as e3:
            print(f"Import failed: {e3}")
            # List all attributes
            print(f"Available in hopsworks: {[x for x in dir(hopsworks) if not x.startswith('_')]}")
            raise

print("Success!")
