-- Compute daily statistics for each feature
WITH daily_stats AS (
    SELECT
        CAST(event_time AS DATE) AS event_date,
        AVG(f1) AS f1_mean,
        STDDEV(f1) AS f1_stddev,
        AVG(f2) AS f2_mean,
        STDDEV(f2) AS f2_stddev,
        AVG(f3) AS f3_mean,
        STDDEV(f3) AS f3_stddev,
        AVG(f4) AS f4_mean,
        STDDEV(f4) AS f4_stddev,
        AVG(f5) AS f5_mean,
        STDDEV(f5) AS f5_stddev,
        AVG(f6) AS f6_mean,
        STDDEV(f6) AS f6_stddev,
        COUNT(*) AS count
    FROM workspace.mlpabf81e30.mlpabf81e30_features
    GROUP BY CAST(event_time AS DATE)
),

-- Compute rolling Z-scores for each feature
rolling_stats AS (
    SELECT
        event_date,
        f1_mean,
        f1_stddev,
        LAG(f1_mean, 1) OVER (ORDER BY event_date) AS f1_prev_mean,
        LAG(f1_stddev, 1) OVER (ORDER BY event_date) AS f1_prev_stddev,
        f2_mean,
        f2_stddev,
        LAG(f2_mean, 1) OVER (ORDER BY event_date) AS f2_prev_mean,
        LAG(f2_stddev, 1) OVER (ORDER BY event_date) AS f2_prev_stddev,
        f3_mean,
        f3_stddev,
        LAG(f3_mean, 1) OVER (ORDER BY event_date) AS f3_prev_mean,
        LAG(f3_stddev, 1) OVER (ORDER BY event_date) AS f3_prev_stddev,
        f4_mean,
        f4_stddev,
        LAG(f4_mean, 1) OVER (ORDER BY event_date) AS f4_prev_mean,
        LAG(f4_stddev, 1) OVER (ORDER BY event_date) AS f4_prev_stddev,
        f5_mean,
        f5_stddev,
        LAG(f5_mean, 1) OVER (ORDER BY event_date) AS f5_prev_mean,
        LAG(f5_stddev, 1) OVER (ORDER BY event_date) AS f5_prev_stddev,
        f6_mean,
        f6_stddev,
        LAG(f6_mean, 1) OVER (ORDER BY event_date) AS f6_prev_mean,
        LAG(f6_stddev, 1) OVER (ORDER BY event_date) AS f6_prev_stddev
    FROM daily_stats
)

-- Compute Z-scores for each feature
SELECT
    event_date,
    CASE WHEN f1_prev_stddev > 0 THEN ABS(f1_mean - f1_prev_mean) / f1_prev_stddev ELSE NULL END AS f1_z_score,
    CASE WHEN f2_prev_stddev > 0 THEN ABS(f2_mean - f2_prev_mean) / f2_prev_stddev ELSE NULL END AS f2_z_score,
    CASE WHEN f3_prev_stddev > 0 THEN ABS(f3_mean - f3_prev_mean) / f3_prev_stddev ELSE NULL END AS f3_z_score,
    CASE WHEN f4_prev_stddev > 0 THEN ABS(f4_mean - f4_prev_mean) / f4_prev_stddev ELSE NULL END AS f4_z_score,
    CASE WHEN f5_prev_stddev > 0 THEN ABS(f5_mean - f5_prev_mean) / f5_prev_stddev ELSE NULL END AS f5_z_score,
    CASE WHEN f6_prev_stddev > 0 THEN ABS(f6_mean - f6_prev_mean) / f6_prev_stddev ELSE NULL END AS f6_z_score
FROM rolling_stats
ORDER BY event_date;