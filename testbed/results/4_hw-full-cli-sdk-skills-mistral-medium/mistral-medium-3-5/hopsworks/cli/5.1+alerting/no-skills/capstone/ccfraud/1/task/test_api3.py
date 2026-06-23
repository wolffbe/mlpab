import hopsworks

try:
    hopsworks.login()
    print("Logged in successfully")
    
    # Try project directly
    project = hopsworks.project
    print(f"Project: {project}")
    print(f"Project type: {type(project)}")
    
    # List project attributes
    print("\nProject attributes:")
    for attr in dir(project):
        if not attr.startswith('_'):
            print(f"  {attr}")
    
    # Try to get feature store from project
    try:
        fs = project.get_feature_store()
        print(f"\nproject.get_feature_store() works: {fs}")
    except Exception as e:
        print(f"\nproject.get_feature_store() failed: {e}")
        
    # Try client
    try:
        client = hopsworks.client
        print(f"\nclient: {client}")
        print(f"client type: {type(client)}")
    except Exception as e:
        print(f"\nclient failed: {e}")
        
except Exception as e:
    print(f"Error: {e}")
    import traceback
    traceback.print_exc()
