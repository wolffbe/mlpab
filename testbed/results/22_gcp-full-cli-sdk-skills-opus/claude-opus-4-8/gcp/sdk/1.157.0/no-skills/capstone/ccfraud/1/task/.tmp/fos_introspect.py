from google.cloud import aiplatform_v1 as v1
print("Admin client:", hasattr(v1,"FeatureOnlineStoreAdminServiceClient"))
print("FOS fields:", [f.name for f in v1.FeatureOnlineStore.pb(v1.FeatureOnlineStore()).DESCRIPTOR.fields])
print("Bigtable fields:", [f.name for f in v1.FeatureOnlineStore.Bigtable.pb(v1.FeatureOnlineStore.Bigtable()).DESCRIPTOR.fields])
print("Optimized fields:", [f.name for f in v1.FeatureOnlineStore.Optimized.pb(v1.FeatureOnlineStore.Optimized()).DESCRIPTOR.fields])
print("FeatureView fields:", [f.name for f in v1.FeatureView.pb(v1.FeatureView()).DESCRIPTOR.fields])
print("BQSource fields:", [f.name for f in v1.FeatureView.BigQuerySource.pb(v1.FeatureView.BigQuerySource()).DESCRIPTOR.fields])
print("SyncConfig fields:", [f.name for f in v1.FeatureView.SyncConfig.pb(v1.FeatureView.SyncConfig()).DESCRIPTOR.fields])
adm=v1.FeatureOnlineStoreAdminServiceClient
print("admin methods:", [m for m in dir(adm) if 'feature' in m.lower() and not m.startswith('_')])
