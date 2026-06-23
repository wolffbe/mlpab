import hopsworks
import json

print("Connecting to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

# Get existing feature groups
fg_a = fs.get_feature_group("rawa822586", version=1)
fg_b = fs.get_feature_group("rawb822586", version=1)
fg_derived = fs.get_feature_group("derived822586", version=1)

print(f"fg_a: {fg_a.name} v{fg_a.version}, id={fg_a.id}")
print(f"fg_b: {fg_b.name} v{fg_b.version}, id={fg_b.id}")
print(f"fg_derived: {fg_derived.name} v{fg_derived.version}, id={fg_derived.id}")

# Check current parents
print(f"\nCurrent parents of derived822586: {fg_derived.parents}")
print(f"Type of parents: {type(fg_derived.parents)}")

# Inspect the parents attribute
import hsfs
print(f"\nAvailable in hsfs: {[x for x in dir(hsfs) if 'parent' in x.lower() or 'lineage' in x.lower()]}")

# Try to find ExplicitProvenance or ParentFeatureGroup
try:
    from hsfs.core import explicit_provenance as ep_module
    print(f"explicit_provenance module: {dir(ep_module)}")
except Exception as e:
    print(f"explicit_provenance import failed: {e}")

try:
    from hsfs import feature_group as fg_module
    print(f"feature_group module attrs related to parent: {[x for x in dir(fg_module) if 'parent' in x.lower()]}")
except Exception as e:
    print(f"feature_group module import: {e}")

# Try setting parents directly
print("\nTrying to set parents on derived822586...")
try:
    # Check what type _parents expects
    print(f"_parents type: {type(fg_derived._parents)}")
    print(f"_parents value: {fg_derived._parents}")
except Exception as e:
    print(f"_parents error: {e}")

# Try adding parents via the parents setter
try:
    fg_derived.parents = [fg_a, fg_b]
    fg_derived.save()
    print("Parents set and saved successfully!")
except Exception as e:
    print(f"Setting parents failed: {e}")

# Try using explicit_provenance
try:
    from hsfs.core.explicit_provenance import Links
    print(f"Links class: {dir(Links)}")
except Exception as e:
    print(f"Links import failed: {e}")

try:
    # Try registering via API directly
    from hsfs.client import get_instance
    client = get_instance()
    print(f"Client type: {type(client)}")
except Exception as e:
    print(f"Client get failed: {e}")

# Check if there's a parent feature group type
try:
    from hsfs.feature_group import FeatureGroupBase
    print(f"FeatureGroupBase attrs: {[x for x in dir(FeatureGroupBase) if 'parent' in x.lower()]}")
except Exception as e:
    print(f"FeatureGroupBase import: {e}")

# Check lineage after attempting to set parents
print(f"\nFinal parents of derived822586: {fg_derived.parents}")

# Get provenance info
try:
    provenance = fg_derived.get_parent_feature_groups()
    print(f"\nParent feature groups: {provenance}")
    if provenance:
        for p in provenance.accessible:
            print(f"  - {p.name} v{p.version}")
except Exception as e:
    print(f"get_parent_feature_groups error: {e}")
