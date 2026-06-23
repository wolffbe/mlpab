import hopsworks
import json

print("Connecting to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

# Get existing feature groups
fg_a = fs.get_feature_group("rawa822586", version=1)
fg_b = fs.get_feature_group("rawb822586", version=1)
fg_derived = fs.get_feature_group("derived822586", version=1)

print(f"fg_derived id: {fg_derived.id}")

# Look at the feature group engine
import hsfs
from hsfs.engine import get_instance as get_engine
engine = get_engine()
print(f"Engine type: {type(engine)}")

# Look at the feature group API
from hsfs.core import feature_group_api
fgapi = feature_group_api.FeatureGroupApi()
print(f"FeatureGroupApi methods: {[m for m in dir(fgapi) if not m.startswith('_')]}")

# Look at the feature_group_base for update methods
from hsfs.feature_group import FeatureGroup
fg_methods = [m for m in dir(FeatureGroup) if not m.startswith('_')]
print(f"\nFeatureGroup public methods: {fg_methods}")

# Try update method
try:
    result = fgapi.update_metadata(fg_derived, fg_derived, "updateMetadata")
    print(f"updateMetadata result: {result}")
except Exception as e:
    print(f"updateMetadata error: {e}")

# Try to find what parameters the API accepts for setting parents
import inspect
try:
    src = inspect.getsource(fgapi.update_metadata)
    print(f"\nupdate_metadata source:\n{src}")
except Exception as e:
    print(f"Could not get source: {e}")

# Look at the FeatureGroup init to see how parents are passed to save
try:
    src = inspect.getsource(FeatureGroup.save)
    print(f"\nFeatureGroup.save source:\n{src[:2000]}")
except Exception as e:
    print(f"Could not get save source: {e}")

# Try to delete derived FG and recreate with parents
print("\n=== Alternative: check if we can pass parents during get_or_create ===")
# Check the FeatureGroup constructor
try:
    sig = inspect.signature(FeatureGroup.__init__)
    print(f"FeatureGroup.__init__ signature: {sig}")
except Exception as e:
    print(f"Signature error: {e}")

# Check how parents is used in the JSON serialization
try:
    fg_json = fg_derived.json()
    fg_dict = json.loads(fg_json)
    print(f"\nFeatureGroup JSON keys: {list(fg_dict.keys())}")
    if 'parents' in fg_dict:
        print(f"parents in JSON: {fg_dict['parents']}")
except Exception as e:
    print(f"JSON error: {e}")

# Check ExplicitProvenance for how parents are structured
from hsfs.core.explicit_provenance import Links
try:
    from hsfs import feature_group as fg_module
    # Find parent-related class
    import inspect
    src = inspect.getsource(fg_module)
    # Find 'parent' occurrences
    lines = src.split('\n')
    for i, line in enumerate(lines):
        if 'parent' in line.lower():
            print(f"Line {i}: {line}")
except Exception as e:
    print(f"Source inspection error: {e}")
