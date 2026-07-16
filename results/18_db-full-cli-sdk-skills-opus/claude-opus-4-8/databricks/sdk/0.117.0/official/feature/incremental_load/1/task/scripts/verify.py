import sys
sys.path.insert(0, 'scripts')
from common import *

TABLE = f'{CAT}.{SCH}.incrementalb48074'
ONLINE = f'{CAT}.{SCH}.incrementalb48074_online'

print('=== feature table ===')
r = run(f'SELECT COUNT(*) n, COUNT(DISTINCT row_id) d, MIN(event_time) mn, MAX(event_time) mx FROM {TABLE}')
print('count/distinct/min/max event_time:', r.data_array)

print('=== primary key & timeseries constraint ===')
r = run(f"""SELECT constraint_name, constraint_type FROM {CAT}.information_schema.table_constraints
            WHERE table_schema='{SCH}' AND table_name='incrementalb48074'""")
print(r.data_array)
r = run(f"""SELECT column_name FROM {CAT}.information_schema.constraint_column_usage
            WHERE table_name='incrementalb48074'""")
print('pk columns:', r.data_array)

print('=== synced online table ===')
t = w.database.get_synced_database_table(name=ONLINE)
ds = t.as_dict().get('data_synchronization_status', {})
print('detailed_state:', ds.get('detailed_state'))

print('=== job ===')
j = w.jobs.get(458151473938343)
print('name:', j.settings.name)
print('schedule:', j.settings.schedule.quartz_cron_expression, j.settings.schedule.pause_status)
