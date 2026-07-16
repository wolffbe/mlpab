import json
import pandas as pd
import hopsworks
from hsfs.statistics_config import StatisticsConfig

# pandas used ONLY to load the raw CSV for ingestion into the platform.
df = pd.read_csv("data/training_data.csv")
print("loaded", df.shape, flush=True)

project = hopsworks.login()
fs = project.get_feature_store()

fg = fs.get_or_create_feature_group(
    name="leakage_probe",
    version=1,
    description="Leakage detection probe: f1-f6 features plus binary label.",
    primary_key=["row_id"],
    online_enabled=False,
    statistics_config=StatisticsConfig(
        enabled=True,
        correlations=True,   # platform-side Pearson correlation matrix
        histograms=False,
        exact_uniqueness=False,
    ),
)

fg.insert(df, wait=True)
print("inserted; fg.id =", fg.id, flush=True)

# Force statistics computation on the offline data (incl. correlations).
try:
    fg.compute_statistics()
except Exception as e:
    print("compute_statistics note:", e, flush=True)

stats = fg.statistics
print("STATS TYPE:", type(stats), flush=True)
content = getattr(stats, "content", None)
if content is None:
    # try get_statistics
    try:
        content = fg.get_statistics().content
    except Exception as e:
        print("get_statistics err:", e, flush=True)

with open(".tmp/stats_content.json", "w") as f:
    json.dump(content, f, default=str, indent=2)
print("WROTE .tmp/stats_content.json", flush=True)
print(json.dumps(content, default=str)[:3000], flush=True)
