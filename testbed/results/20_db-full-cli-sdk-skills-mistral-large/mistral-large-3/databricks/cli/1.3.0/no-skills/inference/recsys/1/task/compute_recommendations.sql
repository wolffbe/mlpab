-- Compute top-5 recommendations for every user
CREATE OR REPLACE TABLE workspace.mlpab5a0819.recse3a36e AS
WITH user_item_scores AS (
  SELECT
    u.user_id,
    i.item_id,
    (u.e1 * i.e1 + u.e2 * i.e2 + u.e3 * i.e3 + u.e4 * i.e4 + u.e5 * i.e5 + u.e6 * i.e6 + u.e7 * i.e7 + u.e8 * i.e8) AS score
  FROM workspace.mlpab5a0819.user_embeddings u
  CROSS JOIN workspace.mlpab5a0819.item_embeddings i
),
filtered_scores AS (
  SELECT
    uis.user_id,
    uis.item_id,
    uis.score
  FROM user_item_scores uis
  LEFT JOIN workspace.mlpab5a0819.interactions int
    ON uis.user_id = int.user_id AND uis.item_id = int.item_id
  WHERE int.user_id IS NULL
),
ranked_scores AS (
  SELECT
    user_id,
    item_id,
    score,
    ROW_NUMBER() OVER (PARTITION BY user_id ORDER BY score DESC, item_id ASC) AS rank
  FROM filtered_scores
)
SELECT
  CONCAT(user_id, '#', rank) AS rec_id,
  user_id,
  rank,
  item_id
FROM ranked_scores
WHERE rank <= 5;

-- Create a Feature Store table and enable online serving
CREATE FEATURE TABLE IF NOT EXISTS workspace.mlpab5a0819.recse3a36e_feature (
  rec_id STRING,
  user_id STRING,
  rank INT,
  item_id STRING
)
TBLPROPERTIES (
  "primary-key" = "rec_id",
  "online-store-enabled" = "true"
);

-- Populate the Feature Store table
INSERT INTO workspace.mlpab5a0819.recse3a36e_feature
SELECT rec_id, user_id, rank, item_id FROM workspace.mlpab5a0819.recse3a36e;