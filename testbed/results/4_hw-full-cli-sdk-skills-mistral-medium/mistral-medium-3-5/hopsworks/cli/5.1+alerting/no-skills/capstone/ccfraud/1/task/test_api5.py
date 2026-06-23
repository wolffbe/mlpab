import hopsworks

try:
    hopsworks.login()
    print("Logged in successfully")
    
    # Try to get the current project
    from hopsworks_common.project import Project
    project = Project()
    print("Project created")
    
    # Try to get feature store from project
    try:
        fs = project.get_feature_store()
        print(f"project.get_feature_store() works!")
        print(f"Feature store type: {type(fs)}")
        
        # List feature store attributes
        print("\nFeature store attributes:")
        for attr in dir(fs):
            if not attr.startswith('_'):
                print(f"  {attr}")
    except Exception as e:
        print(f"project.get_feature_store() failed: {e}")
        import traceback
        traceback.print_exc()
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
