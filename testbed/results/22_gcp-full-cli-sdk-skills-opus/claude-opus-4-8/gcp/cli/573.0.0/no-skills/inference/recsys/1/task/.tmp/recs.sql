CREATE OR REPLACE TABLE `mlpab_mlpabb4a889.recs75dfad` AS
WITH scores AS (
  SELECT
    u.user_id,
    i.item_id,
    (u.e1*i.e1 + u.e2*i.e2 + u.e3*i.e3 + u.e4*i.e4 + u.e5*i.e5 + u.e6*i.e6 + u.e7*i.e7 + u.e8*i.e8) AS score
  FROM `mlpab_mlpabb4a889.user_embeddings` u
  CROSS JOIN `mlpab_mlpabb4a889.item_embeddings` i
  WHERE NOT EXISTS (
    SELECT 1 FROM `mlpab_mlpabb4a889.interactions` x
    WHERE x.user_id = u.user_id AND x.item_id = i.item_id
  )
),
ranked AS (
  SELECT
    user_id, item_id, score,
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
