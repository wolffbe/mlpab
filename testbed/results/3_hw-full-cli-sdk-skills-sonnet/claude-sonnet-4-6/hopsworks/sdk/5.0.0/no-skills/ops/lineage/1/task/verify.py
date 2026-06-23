import hopsworks
import json
import os

print("Connecting to Hopsworks...")
project = hopsworks.login()
fs = project.get_feature_store()

# Verify all feature groups exist
print("\n=== Verifying feature groups ===")
for name in ["rawa822586", "rawb822586", "derived822586"]:
    fg = fs.get_feature_group(name, version=1)
    print(f"{name} v1: id={fg.id}, online_enabled={fg.online_enabled}, rows~{fg.statistics}")

# Check derived FG lineage
print("\n=== Checking lineage for derived822586 ===")
fg_derived = fs.get_feature_group("derived822586", version=1)
print(f"_parents: {fg_derived._parents}")

provenance = fg_derived.get_parent_feature_groups()
print(f"provenance: {provenance}")
if provenance:
    print(f"is_empty: {provenance.is_empty()}")
    if hasattr(provenance, 'accessible') and provenance.accessible:
        for p in provenance.accessible:
            print(f"  Accessible parent: {p.name} v{p.version}")
    if hasattr(provenance, 'deleted'):
        for p in provenance.deleted:
            print(f"  Deleted parent: {p}")
    if hasattr(provenance, 'inaccessible'):
        for p in provenance.inaccessible:
            print(f"  Inaccessible parent: {p}")

# Check online store
print("\n=== Checking online store for derived822586 ===")
fg_a = fs.get_feature_group("rawa822586", version=1)
fg_b = fs.get_feature_group("rawb822586", version=1)
print(f"rawa822586 online_enabled: {fg_a.online_enabled}")
print(f"rawb822586 online_enabled: {fg_b.online_enabled}")
print(f"derived822586 online_enabled: {fg_derived.online_enabled}")

# Read back some data from derived FG to verify correctness
print("\n=== Reading back data from derived822586 ===")
try:
    data = fg_derived.read()
    print(f"Read {len(data)} rows")
    print(data.head())
    # Save to submission
    os.makedirs("submission", exist_ok=True)
    data.to_csv("submission/derived822586.csv", index=False)
    print("Saved to submission/derived822586.csv")
except Exception as e:
    print(f"Read error: {e}")

# Verify answers.json
print("\n=== answers.json ===")
with open("submission/answers.json", "r") as f:
    answers = json.load(f)
print(json.dumps(answers, indent=2))

print("\nVerification complete!")
