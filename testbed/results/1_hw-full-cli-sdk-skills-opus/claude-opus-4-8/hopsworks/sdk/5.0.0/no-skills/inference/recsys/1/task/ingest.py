import pandas as pd
import hopsworks

proj = hopsworks.login()
fs = proj.get_feature_store()

# --- Read raw inputs purely for ingestion onto the platform (no local compute) ---
users = pd.read_csv("data/user_embeddings.csv")
items = pd.read_csv("data/item_embeddings.csv")
inter = pd.read_csv("data/interactions.csv")
print("shapes:", users.shape, items.shape, inter.shape)

user_fg = fs.get_or_create_feature_group(
    name="recsfd473b_user_emb", version=1,
    primary_key=["user_id"], online_enabled=False,
    description="Two-tower user embeddings (raw ingest)",
)
user_fg.insert(users)
print("inserted users")

item_fg = fs.get_or_create_feature_group(
    name="recsfd473b_item_emb", version=1,
    primary_key=["item_id"], online_enabled=False,
    description="Two-tower item embeddings (raw ingest)",
)
item_fg.insert(items)
print("inserted items")

inter_fg = fs.get_or_create_feature_group(
    name="recsfd473b_interactions", version=1,
    primary_key=["user_id", "item_id"], online_enabled=False,
    description="User-item interactions (raw ingest)",
)
inter_fg.insert(inter)
print("inserted interactions")

print("USER_TBL:", user_fg.name + "_" + str(user_fg.version))
print("ITEM_TBL:", item_fg.name + "_" + str(item_fg.version))
print("INTER_TBL:", inter_fg.name + "_" + str(inter_fg.version))
