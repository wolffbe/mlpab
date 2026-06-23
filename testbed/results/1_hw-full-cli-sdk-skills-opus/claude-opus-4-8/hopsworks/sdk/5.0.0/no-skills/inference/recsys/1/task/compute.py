import hopsworks

proj = hopsworks.login()
fs = proj.get_feature_store()

dot = " + ".join([f"u.e{k}*i.e{k}" for k in range(1, 9)])

sql = f"""
WITH scores AS (
  SELECT u.user_id AS user_id, i.item_id AS item_id, ({dot}) AS score
  FROM recsfd473b_user_emb_1 u
  CROSS JOIN recsfd473b_item_emb_1 i
  WHERE NOT EXISTS (
    SELECT 1 FROM recsfd473b_interactions_1 x
    WHERE x.user_id = u.user_id AND x.item_id = i.item_id
  )
),
ranked AS (
  SELECT user_id, item_id, score,
         ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY score DESC, item_id ASC) AS rank
  FROM scores
)
SELECT user_id || '#' || CAST(rank AS VARCHAR) AS rec_id,
       user_id, CAST(rank AS INT) AS rank, item_id
FROM ranked
WHERE rank <= 5
ORDER BY user_id, rank
"""

df = fs.sql(sql, dataframe_type="pandas")
print("ROWS:", len(df))
print("USERS:", df["user_id"].nunique())
print("COLS:", list(df.columns))
print(df.head(12).to_string())
df.to_csv("recs_result.csv", index=False)
print("saved recs_result.csv")
