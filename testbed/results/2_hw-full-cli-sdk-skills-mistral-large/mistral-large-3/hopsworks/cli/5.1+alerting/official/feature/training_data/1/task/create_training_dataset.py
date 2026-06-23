#!/usr/bin/env python3

import hopsworks

# Connect to Hopsworks
project = hopsworks.login()
fs = project.get_feature_store()

# Get feature groups
labels_fg = fs.get_feature_group("labels", version=1)
transactions_fg = fs.get_feature_group("transactions", version=1)
profiles_fg = fs.get_feature_group("profiles", version=1)
activity_fg = fs.get_feature_group("activity", version=1)
account_health_fg = fs.get_feature_group("account_health", version=1)

# Create Feature View with point-in-time joins
query = labels_fg.select_all() \
    .join(transactions_fg.select(["amount", "balance"]), \
          on=["account_id"], \
          join_type="LEFT", \
          prefix=None) \
    .join(profiles_fg.select(["credit_score", "tier"]), \
          on=["account_id"], \
          join_type="LEFT", \
          prefix=None) \
    .join(activity_fg.select(["sessions_7d"]), \
          on=["account_id"], \
          join_type="LEFT", \
          prefix=None) \
    .join(account_health_fg.select(["health_score"]), \
          on=["account_id"], \
          join_type="LEFT", \
          prefix=None)

# Apply point-in-time logic
query = query.filter(transactions_fg.event_time <= labels_fg.label_time) \
    .filter(profiles_fg.event_time <= labels_fg.label_time) \
    .filter(activity_fg.event_time <= labels_fg.label_time) \
    .filter(account_health_fg.event_time <= labels_fg.label_time)

# Get the most recent feature values
query = query.group_by(["account_id", "label_time", "churned"]) \
    .agg({
        "amount": "latest",
        "balance": "latest",
        "credit_score": "latest",
        "tier": "latest",
        "sessions_7d": "latest",
        "health_score": "latest"
    })

# Create Feature View
feature_view = fs.create_feature_view(
    name="churntraining1e2e16",
    version=1,
    query=query,
    labels=["churned"]
)

# Create Training Dataset
training_dataset = feature_view.create_training_dataset(
    name="churntraining1e2e16",
    version=1,
    description="Training data for churn prediction",
    data_format="csv",
    statistics_config=False
)

training_dataset.compute()

print("Training dataset 'churntraining1e2e16', version 1 created successfully.")