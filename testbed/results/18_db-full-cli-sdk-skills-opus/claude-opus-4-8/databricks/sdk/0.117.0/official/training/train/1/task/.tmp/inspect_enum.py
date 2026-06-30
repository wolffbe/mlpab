from databricks.sdk.service import database as d
print('SchedulingPolicy:', [x for x in dir(d.SyncedTableSchedulingPolicy) if not x.startswith('_')])
print('PgSpecificType:', [x for x in dir(d.SyncedTableSpecPgSpecificType) if not x.startswith('_')])
print('DatabaseInstanceState:', [x for x in dir(d.DatabaseInstanceState) if not x.startswith('_')])
