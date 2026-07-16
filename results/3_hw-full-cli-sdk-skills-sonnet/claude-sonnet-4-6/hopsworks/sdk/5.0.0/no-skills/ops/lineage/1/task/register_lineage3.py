import hopsworks
import json
import inspect

print("Connecting to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

# Get existing feature groups
fg_a = fs.get_feature_group("rawa822586", version=1)
fg_b = fs.get_feature_group("rawb822586", version=1)
fg_derived = fs.get_feature_group("derived822586", version=1)

# Inspect how the feature group JSON looks
fg_a_dict = json.loads(fg_a.json())
print(f"fg_a keys: {list(fg_a_dict.keys())}")
print(f"fg_a id: {fg_a_dict.get('id')}")
print(f"fg_a name: {fg_a_dict.get('name')}")

# Look at the to_dict method
try:
    src = inspect.getsource(fg_derived.to_dict)
    # Find parents related lines
    lines = [l for l in src.split('\n') if 'parent' in l.lower() or 'parents' in l]
    print(f"\nto_dict parents lines: {lines}")
except Exception as e:
    print(f"to_dict source error: {e}")

# Try using update_metadata to set parents
# First, set parents using feature group basic info
print("\nTrying update_metadata approach...")

# Check what fg_a looks like when serialized as parent
# The parent is likely just {id, name, version}
parent_a = {"id": fg_a.id, "name": fg_a.name, "version": fg_a.version}
parent_b = {"id": fg_b.id, "name": fg_b.name, "version": fg_b.version}
print(f"parent_a: {parent_a}")
print(f"parent_b: {parent_b}")

# Try to set parents as dicts and update via API
from hsfs.core import feature_group_api
from hopsworks_common import client

fgapi = feature_group_api.FeatureGroupApi()

# Approach 1: Set _parents as simple dicts
fg_derived._parents = [parent_a, parent_b]
fg_derived_json = json.loads(fg_derived.json())
print(f"\nDerived JSON with parents: {fg_derived_json.get('parents')}")

# Try update_metadata
try:
    result = fgapi.update_metadata(fg_derived, fg_derived, "updateMetadata")
    print(f"update_metadata result: {result.name} v{result.version}")
    print(f"Result parents: {result._parents}")
except Exception as e:
    print(f"update_metadata error: {e}")

# Check if parents were set
fg_derived_fresh = fs.get_feature_group("derived822586", version=1)
print(f"\nFresh derived parents: {fg_derived_fresh._parents}")
try:
    provenance = fg_derived_fresh.get_parent_feature_groups()
    print(f"Parent feature groups: {provenance}")
    if provenance and not provenance.is_empty():
        for p in provenance.accessible:
            print(f"  - Accessible: {p.name} v{p.version}")
        for p in provenance.deleted:
            print(f"  - Deleted: {p}")
        for p in provenance.inaccessible:
            print(f"  - Inaccessible: {p}")
except Exception as e:
    print(f"get_parent_feature_groups error: {e}")
