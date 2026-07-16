import google.cloud.aiplatform_v1 as v1
print("FeatureGroup.BigQuery fields:")
for f in v1.FeatureGroup.BigQuery.pb().DESCRIPTOR.fields:
    print("  ", f.name)
print("FeatureGroup.BigQuery.TimeSeries fields:")
for f in v1.FeatureGroup.BigQuery.TimeSeries.pb().DESCRIPTOR.fields:
    print("  ", f.name)
print("FeatureView fields:")
for f in v1.FeatureView.pb().DESCRIPTOR.fields:
    print("  ", f.name)
print("FeatureView.BigQuerySource fields:")
for f in v1.FeatureView.BigQuerySource.pb().DESCRIPTOR.fields:
    print("  ", f.name)
print("FeatureView.SyncConfig fields:")
for f in v1.FeatureView.SyncConfig.pb().DESCRIPTOR.fields:
    print("  ", f.name)
