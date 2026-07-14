CREATE OR REPLACE TABLE `***REDACTED***.mlpab_mlpab9be817.itemsb472cd_queries` AS
SELECT query_id,
       ARRAY(SELECT CAST(x AS FLOAT64) FROM UNNEST(JSON_EXTRACT_ARRAY(embedding)) AS x) AS embedding
FROM `***REDACTED***.mlpab_mlpab9be817.itemsb472cd_queries_staging`;
