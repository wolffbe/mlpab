import os
import google.cloud.bigquery as bigquery

proj = os.environ['GCP_PROJECT']
ds = os.environ['GCP_BQ_DATASET']
c = bigquery.Client(project=proj)
base = "{}.{}".format(proj, ds)

dot = " + ".join(["u.e{n}*i.e{n}".format(n=n) for n in range(1, 9)])

sql = """
CREATE OR REPLACE TABLE `{base}.recs75dfad` AS
WITH inter AS (
  SELECT string_field_0 AS user_id, string_field_1 AS item_id
  FROM `{base}.stg_interactions`
),
scores AS (
  SELECT u.user_id, i.item_id, ({dot}) AS score
  FROM `{base}.stg_user_emb` u
  CROSS JOIN `{base}.stg_item_emb` i
  WHERE NOT EXISTS (
    SELECT 1 FROM inter x
    WHERE x.user_id = u.user_id AND x.item_id = i.item_id
  )
),
ranked AS (
  SELECT user_id, item_id, score,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY score DESC, item_id ASC) AS rank
  FROM scores
)
SELECT
  CONCAT(user_id, '#', CAST(rank AS STRING)) AS rec_id,
  user_id,
  CAST(rank AS INT64) AS rank,
  item_id
FROM ranked
WHERE rank <= 5
ORDER BY user_id, rank
""".format(base=base, dot=dot)

job = c.query(sql)
job.result()
t = c.get_table("{}.recs75dfad".format(base))
print("recs75dfad rows:", t.num_rows)
print("schema:", [(s.name, s.field_type) for s in t.schema])

# sanity sample
for row in c.query("SELECT * FROM `{}.recs75dfad` ORDER BY user_id, rank LIMIT 12".format(base)).result():
    print(dict(row))
# distinct users and rows per user check
for row in c.query("SELECT COUNT(DISTINCT user_id) users, COUNT(*) total, MIN(rank) minr, MAX(rank) maxr FROM `{}.recs75dfad`".format(base)).result():
    print("summary:", dict(row))
