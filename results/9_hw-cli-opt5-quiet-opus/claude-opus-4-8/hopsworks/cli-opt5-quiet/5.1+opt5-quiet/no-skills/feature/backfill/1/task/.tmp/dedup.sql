SELECT row_id || ',' || status || ',' || CAST(CAST(balance AS DECIMAL(18,2)) AS VARCHAR) || ',' || CAST(updated_at AS VARCHAR) AS csv
FROM (
  SELECT row_id, status, balance, updated_at,
         ROW_NUMBER() OVER (PARTITION BY row_id ORDER BY updated_at DESC) AS rn
  FROM delta.mlpab1fcd3e_featurestore.accounts196576_1
)
WHERE rn = 1
