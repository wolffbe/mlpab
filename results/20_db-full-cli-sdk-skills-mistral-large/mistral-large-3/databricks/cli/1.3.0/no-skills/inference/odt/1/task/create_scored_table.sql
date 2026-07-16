CREATE OR REPLACE TABLE workspace.mlpabc9a00f.scoreda4f6e2 AS
SELECT 
    r.request_id,
    r.account_id,
    ROUND(SQRT(POWER(r.request_lat - p.home_lat, 2) + POWER(r.request_lon - p.home_lon, 2)), 6) AS distance_deg,
    ROUND(p.base_score - 0.1 * ROUND(SQRT(POWER(r.request_lat - p.home_lat, 2) + POWER(r.request_lon - p.home_lon, 2)), 6), 6) AS score
FROM 
    csv."`dbfs:/Volumes/workspace/mlpabc9a00f/data_volume/requests.csv`" r
JOIN 
    csv."`dbfs:/Volumes/workspace/mlpabc9a00f/data_volume/profiles.csv`" p
ON 
    r.account_id = p.account_id;