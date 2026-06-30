import dbx

run, TBL, CAT, SCH = dbx.run, dbx.TBL, dbx.CAT, dbx.SCH

run(f'DROP TABLE IF EXISTS {TBL}')
ddl = f'''CREATE TABLE {TBL} (
  row_id STRING NOT NULL,
  account_id STRING,
  event_time BIGINT NOT NULL,
  amount DOUBLE,
  category STRING,
  CONSTRAINT incrementalb48074_pk PRIMARY KEY (row_id, event_time TIMESERIES)
) TBLPROPERTIES (delta.enableChangeDataFeed = true)'''
run(ddl)
print('table created with PK row_id + event_time TIMESERIES')

# Load all increments from the volume
vol = f'/Volumes/{CAT}/{SCH}/inc_raw'
copy = f'''COPY INTO {TBL}
FROM (SELECT row_id::STRING, account_id::STRING, event_time::BIGINT, amount::DOUBLE, category::STRING
      FROM '{vol}')
FILEFORMAT = CSV
FILES = ('increment_01.csv','increment_02.csv','increment_03.csv','increment_04.csv','increment_05.csv','increment_06.csv')
FORMAT_OPTIONS ('header'='true', 'inferSchema'='false')
COPY_OPTIONS ('mergeSchema'='false')'''
run(copy)
print('COPY INTO done')

r = run(f'SELECT count(*) AS c, count(distinct row_id) AS d FROM {TBL}')
print('rows:', r.result.data_array)
