import sys
sys.path.insert(0, '.tmp')
from common import *

tbl = f'{CAT}.{SCH}.__ts_test'
run(f'DROP TABLE IF EXISTS {tbl}')
try:
    run(f'CREATE TABLE {tbl} (row_id STRING NOT NULL, event_time BIGINT NOT NULL, CONSTRAINT pk PRIMARY KEY(row_id, event_time TIMESERIES))')
    print('BIGINT TIMESERIES: OK')
except Exception as e:
    print('BIGINT TIMESERIES FAILED:', str(e)[:400])
run(f'DROP TABLE IF EXISTS {tbl}')
