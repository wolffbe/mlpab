import warnings, urllib3, json, os
warnings.filterwarnings("ignore")
urllib3.disable_warnings()
import pandas as pd

# --- read raw, everything as string to detect empty/null amounts ---
raw = pd.read_csv("data/events.csv", dtype=str, keep_default_na=False)
print("total rows:", len(raw))

VALID_CATS = {"grocery", "travel", "salary", "rent", "other"}

def amount_val(s):
    s = (s or "").strip()
    if s == "":
        return None
    try:
        return float(s)
    except ValueError:
        return None

rejected = []
valid_idx = []
for i, r in raw.iterrows():
    amt = amount_val(r["amount"])
    cat = r["category"].strip()
    ok = True
    if amt is None:                       # rule 1: present + numeric
        ok = False
    elif not (0.0 <= amt <= 10000.0):     # rule 2: range
        ok = False
    elif cat not in VALID_CATS:           # rule 3: category
        ok = False
    if ok:
        valid_idx.append(i)
    else:
        rejected.append(r["row_id"])

print("valid:", len(valid_idx), "rejected:", len(rejected))

# build typed dataframe for the valid rows
df = raw.loc[valid_idx].copy()
df["event_time"] = df["event_time"].astype("int64")
df["amount"] = df["amount"].astype("float64")
df["row_id"] = df["row_id"].astype(str)
df["account_id"] = df["account_id"].astype(str)
df["category"] = df["category"].str.strip().astype(str)

# write submission
os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump({"rejected": rejected}, f)
print("wrote submission/answers.json")

# --- register on platform ---
import hopsworks
proj = hopsworks.login()
fs = proj.get_feature_store()
fg = fs.get_or_create_feature_group(
    name="events5b591e",
    version=1,
    description="events filtered per data contract",
    primary_key=["row_id"],
    event_time="event_time",
    online_enabled=True,
)
fg.insert(df)
print("inserted", len(df), "rows into feature group", fg.name, "v", fg.version)
