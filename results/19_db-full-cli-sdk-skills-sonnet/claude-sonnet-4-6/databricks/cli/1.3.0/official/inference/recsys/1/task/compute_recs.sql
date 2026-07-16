CREATE OR REPLACE TABLE workspace.mlpabff48bb.recs708df6 AS
WITH interactions_raw AS (
  SELECT user_id, item_id
  FROM read_files(
    '/Volumes/workspace/mlpabff48bb/recsys_data/interactions.csv',
    format => 'csv',
    header => 'true'
  )
),
user_emb AS (
  SELECT
    user_id,
    CAST(e1 AS DOUBLE) AS e1, CAST(e2 AS DOUBLE) AS e2,
    CAST(e3 AS DOUBLE) AS e3, CAST(e4 AS DOUBLE) AS e4,
    CAST(e5 AS DOUBLE) AS e5, CAST(e6 AS DOUBLE) AS e6,
    CAST(e7 AS DOUBLE) AS e7, CAST(e8 AS DOUBLE) AS e8
  FROM read_files(
    '/Volumes/workspace/mlpabff48bb/recsys_data/user_embeddings.csv',
    format => 'csv',
    header => 'true'
  )
),
item_emb AS (
  SELECT
    item_id,
    CAST(e1 AS DOUBLE) AS e1, CAST(e2 AS DOUBLE) AS e2,
    CAST(e3 AS DOUBLE) AS e3, CAST(e4 AS DOUBLE) AS e4,
    CAST(e5 AS DOUBLE) AS e5, CAST(e6 AS DOUBLE) AS e6,
    CAST(e7 AS DOUBLE) AS e7, CAST(e8 AS DOUBLE) AS e8
  FROM read_files(
    '/Volumes/workspace/mlpabff48bb/recsys_data/item_embeddings.csv',
    format => 'csv',
    header => 'true'
  )
),
dot_products AS (
  SELECT
    u.user_id,
    i.item_id,
    u.e1*i.e1 + u.e2*i.e2 + u.e3*i.e3 + u.e4*i.e4 +
    u.e5*i.e5 + u.e6*i.e6 + u.e7*i.e7 + u.e8*i.e8 AS score
  FROM user_emb u
  CROSS JOIN item_emb i
  WHERE NOT EXISTS (
    SELECT 1 FROM interactions_raw ir
    WHERE ir.user_id = u.user_id AND ir.item_id = i.item_id
  )
),
ranked AS (
  SELECT
    user_id,
    item_id,
    score,
    ROW_NUMBER() OVER (
      PARTITION BY user_id
      ORDER BY score DESC, item_id ASC
    ) AS rank
  FROM dot_products
)
SELECT
  CONCAT(user_id, '#', CAST(rank AS STRING)) AS rec_id,
  user_id,
  CAST(rank AS INT) AS rank,
  item_id
FROM ranked
WHERE rank <= 5
ORDER BY user_id, rank
