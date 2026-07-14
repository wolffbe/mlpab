SELECT
  r.request_id AS request_id,
  r.account_id AS account_id,
  ROUND(SQRT(POW(r.request_lat - p.home_lat, 2) + POW(r.request_lon - p.home_lon, 2)), 6) AS distance_deg,
  ROUND(p.base_score - 0.1 * ROUND(SQRT(POW(r.request_lat - p.home_lat, 2) + POW(r.request_lon - p.home_lon, 2)), 6), 6) AS score
FROM mlpab_mlpabc32fea.raw_requests r
JOIN mlpab_mlpabc32fea.raw_profiles p USING (account_id)
