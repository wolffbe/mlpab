import hopsworks

try:
    hopsworks.login()
    print("Logged in successfully")
    
    # Try different ways to get feature store
    try:
        fs = hopsworks.get_feature_store()
        print("get_feature_store() works")
    except AttributeError as e:
        print(f"get_feature_store() failed: {e}")
    
    try:
        fs = hopsworks.feature_store.get()
        print("feature_store.get() works")
    except AttributeError as e:
        print(f"feature_store.get() failed: {e}")
    
    try:
        from hopsworks import feature_store
        fs = feature_store.get()
        print("from hopsworks import feature_store; feature_store.get() works")
    except Exception as e:
        print(f"from hopsworks import feature_store failed: {e}")
        
    # List all attributes
    print("\nAll hopsworks attributes:")
    for attr in dir(hopsworks):
        if not attr.startswith('_'):
            print(f"  {attr}")
            
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
