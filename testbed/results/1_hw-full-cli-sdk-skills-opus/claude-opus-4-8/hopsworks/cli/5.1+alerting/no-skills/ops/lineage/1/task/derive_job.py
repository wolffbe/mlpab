"""Feature pipeline: derive derived2962af from rawa2962af + rawb2962af.

Runs on the Hopsworks platform as a PYTHON job. Reads the two raw feature
groups, inner-joins on row_id (keeping only row_ids present in BOTH),
computes col_sum = a_val + b_val rounded to 6 decimals, and writes a new
feature group whose parents are the two raw feature groups (so lineage is
recorded) with the online store enabled.
"""
import hopsworks

project = hopsworks.login()
fs = project.get_feature_store()

rawa = fs.get_feature_group("rawa2962af", version=1)
rawb = fs.get_feature_group("rawb2962af", version=1)

# Inner join on row_id -> only row_ids present in both sources.
query = rawa.select(["row_id", "a_val"]).join(
    rawb.select(["b_val"]), on=["row_id"], join_type="inner"
)
df = query.read()

df["col_sum"] = (df["a_val"] + df["b_val"]).round(6)
out = df[["row_id", "col_sum"]].sort_values("row_id").reset_index(drop=True)
print("Derived rows:", len(out))
print(out.head())

derived = fs.create_feature_group(
    name="derived2962af",
    version=1,
    description="col_sum = a_val + b_val for row_ids present in both rawa2962af and rawb2962af",
    primary_key=["row_id"],
    online_enabled=True,
    parents=[rawa, rawb],
)
derived.insert(out)
print("Inserted into derived2962af v1:", len(out), "rows")
