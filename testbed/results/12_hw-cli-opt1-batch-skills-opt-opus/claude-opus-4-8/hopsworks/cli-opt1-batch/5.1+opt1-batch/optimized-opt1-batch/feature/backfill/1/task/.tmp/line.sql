SELECT row_id || ',' || status || ',' || CAST(balance AS VARCHAR) || ',' || CAST(updated_at AS VARCHAR) AS line
FROM (
  SELECT row_id, status, balance, updated_at,
         ROW_NUMBER() OVER (PARTITION BY row_id ORDER BY updated_at DESC) AS rn
  FROM delta.mlpabc8e17e_featurestore.accountsc94febstg_1
) t
WHERE rn = 1
