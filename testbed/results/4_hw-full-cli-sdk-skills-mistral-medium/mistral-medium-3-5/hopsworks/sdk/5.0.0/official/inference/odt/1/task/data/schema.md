# Schema

- **requests.csv**: request_id (unique key), account_id, request_lat, request_lon (the request's coordinates — only known at request time), requested_at (ISO UTC)
- **profiles.csv**: account_id (unique key), home_lat, home_lon, base_score (precomputed profile score)
