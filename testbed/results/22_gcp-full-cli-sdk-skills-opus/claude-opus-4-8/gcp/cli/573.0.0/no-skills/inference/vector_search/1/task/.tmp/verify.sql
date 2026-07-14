SELECT COUNT(*) AS n,
       MIN(ARRAY_LENGTH(embedding)) AS mind,
       MAX(ARRAY_LENGTH(embedding)) AS maxd
FROM `***REDACTED***.mlpab_mlpab9be817.itemsb472cd`;
