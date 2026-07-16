WITH scores AS (
  SELECT u.user_id AS user_id, i.item_id AS item_id,
         u.e1*i.e1 + u.e2*i.e2 + u.e3*i.e3 + u.e4*i.e4
       + u.e5*i.e5 + u.e6*i.e6 + u.e7*i.e7 + u.e8*i.e8 AS score
  FROM mlpabac18fb_featurestore.user_emb_1 u
  CROSS JOIN mlpabac18fb_featurestore.item_emb_1 i
  WHERE NOT EXISTS (
    SELECT 1 FROM mlpabac18fb_featurestore.interactions_1 x
    WHERE x.user_id = u.user_id AND x.item_id = i.item_id
  )
),
ranked AS (
  SELECT user_id, item_id, score,
         CAST(ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY score DESC, item_id ASC) AS integer) AS rank
  FROM scores
)
SELECT user_id || '#' || CAST(rank AS varchar) AS rec_id, user_id, rank, item_id
FROM ranked
WHERE rank <= 5
ORDER BY user_id, rank
