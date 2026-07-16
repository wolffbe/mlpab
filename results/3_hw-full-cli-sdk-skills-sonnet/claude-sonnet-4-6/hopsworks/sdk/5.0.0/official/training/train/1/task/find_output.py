"""Find the predictions.csv output from the job."""
import hopsworks

JOB_NAME = "trainjob7a8727"

def main():
    project = hopsworks.login()
    dataset_api = project.get_dataset_api()

    # Inspect what list returns
    try:
        items = dataset_api.list(f"Resources/{JOB_NAME}")
        print(f"Type of items: {type(items)}")
        if items:
            print(f"First item type: {type(items[0])}")
            print(f"First item: {items[0]}")
            print(f"All items: {items}")
        else:
            print("Empty list")
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()

    # Try get method
    try:
        result = dataset_api.get(f"Resources/{JOB_NAME}")
        print(f"\nGet result: {result}")
        print(f"Get result type: {type(result)}")
    except Exception as e:
        print(f"Get error: {e}")

    # Try exists
    for path in [
        f"Resources/{JOB_NAME}/predictions.csv",
        f"Resources/{JOB_NAME}/train.csv",
        f"Resources/{JOB_NAME}",
    ]:
        try:
            exists = dataset_api.exists(path)
            print(f"exists({path}): {exists}")
        except Exception as e:
            print(f"exists error for {path}: {e}")

if __name__ == "__main__":
    main()
