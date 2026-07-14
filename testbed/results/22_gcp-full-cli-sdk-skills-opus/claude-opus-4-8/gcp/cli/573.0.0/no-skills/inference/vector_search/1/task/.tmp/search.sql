WITH vs AS (
  SELECT query.query_id AS query_id, base.item_id AS item_id, distance
  FROM VECTOR_SEARCH(
    TABLE `***REDACTED***.mlpab_mlpab9be817.itemsb472cd`, 'embedding',
    TABLE `***REDACTED***.mlpab_mlpab9be817.itemsb472cd_queries`, 'embedding',
    top_k => 5,
    distance_type => 'EUCLIDEAN')
)
SELECT query_id,
       ARRAY_TO_STRING(ARRAY_AGG(item_id ORDER BY distance LIMIT 5), '|') AS ids
FROM vs
GROUP BY query_id
ORDER BY query_id;
