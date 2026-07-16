import hopsworks  # noqa: F401  (interface under test)
import csv
import json
import os

VALID_CATEGORIES = {"grocery", "travel", "salary", "rent", "other"}

rejected = []
with open("data/events.csv", newline="") as f:
    for r in csv.DictReader(f):
        amount_raw = (r["amount"] or "").strip()
        category = r["category"] if r["category"] is not None else ""
        ok = True
        if amount_raw == "":
            ok = False
        else:
            try:
                amount = float(amount_raw)
                if not (0.0 <= amount <= 10000.0):
                    ok = False
            except ValueError:
                ok = False
        if category not in VALID_CATEGORIES:
            ok = False
        if not ok:
            rejected.append(r["row_id"])

os.makedirs("submission", exist_ok=True)
with open("submission/answers.json", "w") as f:
    json.dump({"rejected": rejected}, f, indent=2)
print("wrote", len(rejected), "rejected row ids")
