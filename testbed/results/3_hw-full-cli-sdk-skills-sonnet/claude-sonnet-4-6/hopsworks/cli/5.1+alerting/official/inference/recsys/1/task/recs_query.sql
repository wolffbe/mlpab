WITH scores AS (
  SELECT
    u.user_id,
    i.item_id,
    (u.e1 * i.e1 + u.e2 * i.e2 + u.e3 * i.e3 + u.e4 * i.e4 +
     u.e5 * i.e5 + u.e6 * i.e6 + u.e7 * i.e7 + u.e8 * i.e8) AS score
  FROM delta.mlpab1a96c3_featurestore.user_embeddings_tmp_1 u
  CROSS JOIN delta.mlpab1a96c3_featurestore.item_embeddings_tmp_1 i
),
filtered AS (
  SELECT s.user_id, s.item_id, s.score
  FROM scores s
  LEFT JOIN delta.mlpab1a96c3_featurestore.interactions_tmp_1 ex
    ON s.user_id = ex.user_id AND s.item_id = ex.item_id
  WHERE ex.user_id IS NULL
),
ranked AS (
  SELECT
    user_id,
    item_id,
    score,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY score DESC, item_id ASC) AS rank
  FROM filtered
)
SELECT
  user_id || '#' || CAST(rank AS VARCHAR) AS rec_id,
  user_id,
  CAST(rank AS INTEGER) AS rank,
  item_id
FROM ranked
WHERE rank <= 5
ORDER BY user_id, rank
