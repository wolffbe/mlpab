import json
import pandas as pd
import hopsworks
from hsfs.feature import Feature

project = hopsworks.login()
fs = project.get_feature_store()
print("Connected to project:", project.name)

raw_a = pd.read_csv("data/raw_a.csv")
raw_b = pd.read_csv("data/raw_b.csv")
raw_a["row_id"] = raw_a["row_id"].astype(str)
raw_b["row_id"] = raw_b["row_id"].astype(str)
raw_a["a_val"] = raw_a["a_val"].astype(float)
raw_b["b_val"] = raw_b["b_val"].astype(float)
print("raw_a", raw_a.shape, "raw_b", raw_b.shape)

# ---- Source FG rawa2962af ----
fga = fs.get_or_create_feature_group(
    name="rawa2962af",
    version=1,
    description="Raw source table A (row_id, a_val) loaded from raw_a.csv",
    primary_key=["row_id"],
    features=[
        Feature("row_id", "string", description="Record key"),
        Feature("a_val", "double", description="Raw value from source A"),
    ],
    online_enabled=False,
    statistics_config=False,
)
fga.insert(raw_a, wait=True)
print("Inserted rawa2962af, id=", fga.id)

# ---- Source FG rawb2962af ----
fgb = fs.get_or_create_feature_group(
    name="rawb2962af",
    version=1,
    description="Raw source table B (row_id, b_val) loaded from raw_b.csv",
    primary_key=["row_id"],
    features=[
        Feature("row_id", "string", description="Record key"),
        Feature("b_val", "double", description="Raw value from source B"),
    ],
    online_enabled=False,
    statistics_config=False,
)
fgb.insert(raw_b, wait=True)
print("Inserted rawb2962af, id=", fgb.id)

# ---- Platform-side inner join via the query engine ----
query = fga.select(["row_id", "a_val"]).join(
    fgb.select(["b_val"]), on=["row_id"], join_type="inner"
)
joined = query.read(dataframe_type="pandas")
print("Joined shape:", joined.shape)
print(joined.head())

# col_sum = a_val + b_val rounded to 6 decimals, only rows present in BOTH
derived = pd.DataFrame({
    "row_id": joined["row_id"].astype(str),
    "col_sum": (joined["a_val"].astype(float) + joined["b_val"].astype(float)).round(6),
})
derived = derived.dropna(subset=["row_id"]).drop_duplicates(subset=["row_id"]).reset_index(drop=True)
print("Derived shape:", derived.shape)
print(derived.head())

# Save local backup
import os
os.makedirs("submission", exist_ok=True)
derived.to_csv("submission/derived2962af.csv", index=False)

# ---- Derived FG with provenance + online enabled ----
derived_fg = fs.get_or_create_feature_group(
    name="derived2962af",
    version=1,
    description="Derived: col_sum = a_val + b_val for row_ids present in BOTH rawa2962af and rawb2962af (inner join), rounded to 6 dp.",
    primary_key=["row_id"],
    features=[
        Feature("row_id", "string", description="Record key, present in both sources"),
        Feature("col_sum", "double", description="a_val + b_val rounded to 6 decimals"),
    ],
    online_enabled=True,
    stream=True,
    parents=[fga, fgb],
    statistics_config=False,
)
derived_fg.insert(derived, wait=True)
print("Inserted derived2962af, id=", derived_fg.id)

# materialize offline so query reads work
try:
    derived_fg.materialization_job.run(await_termination=True)
    print("Materialization done")
except Exception as e:
    print("Materialization note:", e)

# ---- Answer ----
answers = {"derived_from": sorted(["rawa2962af", "rawb2962af"])}
with open("submission/answers.json", "w") as f:
    json.dump(answers, f, indent=2)
print("answers:", answers)
print("DONE")
