import sys, io, glob, os
sys.path.insert(0, '.tmp')
from common import *
from databricks.sdk.service.catalog import VolumeType

TABLE = f'{CAT}.{SCH}.incrementalb48074'
VOL = 'incr_files'
VOL_PATH = f'/Volumes/{CAT}/{SCH}/{VOL}'

# 1. Volume for raw increment files
try:
    w.volumes.create(catalog_name=CAT, schema_name=SCH, name=VOL, volume_type=VolumeType.MANAGED)
    print('volume created')
except Exception as e:
    print('volume exists/err:', str(e)[:120])

# 2. Upload all increment CSVs into the volume
for f in sorted(glob.glob('data/increment_*.csv')):
    name = os.path.basename(f)
    with open(f, 'rb') as fh:
        w.files.upload(f'{VOL_PATH}/{name}', io.BytesIO(fh.read()), overwrite=True)
    print('uploaded', name)

# 3. Create the feature table: PK row_id, TIMESERIES event_time (epoch ms, bigint), CDF on
run(f'DROP TABLE IF EXISTS {TABLE}')
run(f"""
CREATE TABLE {TABLE} (
  row_id STRING NOT NULL,
  account_id STRING,
  event_time BIGINT NOT NULL,
  amount DOUBLE,
  category STRING,
  CONSTRAINT incrementalb48074_pk PRIMARY KEY (row_id, event_time TIMESERIES)
)
TBLPROPERTIES (delta.enableChangeDataFeed = true)
""")
print('table created:', TABLE)

# 4. Load ALL provided increments via COPY INTO (idempotent, tracks loaded files)
run(f"""
COPY INTO {TABLE}
FROM (
  SELECT row_id::STRING, account_id::STRING, event_time::BIGINT, amount::DOUBLE, category::STRING
  FROM '{VOL_PATH}'
)
FILEFORMAT = CSV
PATTERN = 'increment_*.csv'
FORMAT_OPTIONS ('header'='true', 'inferSchema'='false')
COPY_OPTIONS ('mergeSchema'='false')
""")

res = run(f'SELECT COUNT(*) AS n, COUNT(DISTINCT row_id) AS d FROM {TABLE}')
print('row count / distinct row_id:', res.data_array)
