WITH scores AS (
  SELECT u.user_id AS uid, i.item_id AS iid,
    u.e1*i.e1+u.e2*i.e2+u.e3*i.e3+u.e4*i.e4+u.e5*i.e5+u.e6*i.e6+u.e7*i.e7+u.e8*i.e8 AS score
  FROM user_embeddings_1 u CROSS JOIN item_embeddings_1 i
  WHERE NOT EXISTS (SELECT 1 FROM interactions_1 x WHERE x.user_id = u.user_id AND x.item_id = i.item_id)
),
ranked AS (
  SELECT uid, iid, score,
    ROW_NUMBER() OVER (PARTITION BY uid ORDER BY score DESC, iid ASC) AS rnk
  FROM scores
)
SELECT uid || '#' || CAST(rnk AS varchar) AS rec_id, uid AS user_id, rnk AS rank, iid AS item_id
FROM ranked
WHERE rnk <= 5
ORDER BY uid, rnk
