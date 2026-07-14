CREATE OR REPLACE TABLE mlpab_mlpab0b1d22.scored3ace95 AS
WITH joined AS (
  SELECT
    r.request_id,
    r.account_id,
    ROUND(SQRT(POW(r.request_lat - p.home_lat, 2) + POW(r.request_lon - p.home_lon, 2)), 6) AS distance_deg,
    p.base_score
  FROM mlpab_mlpab0b1d22.stg_requests r
  JOIN mlpab_mlpab0b1d22.stg_profiles p USING (account_id)
)
SELECT
  request_id,
  account_id,
  distance_deg,
  ROUND(base_score - 0.1 * distance_deg, 6) AS score
FROM joined
