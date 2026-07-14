CREATE OR REPLACE TABLE `***REDACTED***.mlpab_mlpab9be817.itemsb472cd` AS
SELECT item_id,
       label,
       ARRAY(SELECT CAST(x AS FLOAT64) FROM UNNEST(JSON_EXTRACT_ARRAY(embedding)) AS x) AS embedding
FROM `***REDACTED***.mlpab_mlpab9be817.itemsb472cd_staging`;
