SELECT
    t.row_id,
    t.account_id,
    t.event_time,
    t.amount * f.fx_rate AS amount_usd,
    CASE
        WHEN DAYOFWEEK(FROM_UNIXTIME(t.event_time / 1000)) IN (1, 7) THEN 1
        ELSE 0
    END AS is_weekend,
    SUM(t.amount) OVER (
        PARTITION BY t.account_id
        ORDER BY t.event_time
        RANGE BETWEEN INTERVAL '7' DAY PRECEDING AND CURRENT ROW
    ) AS amount_7d
FROM
    ${transactions_raw} t
LEFT JOIN
    ${fx_rates_raw} f
ON
    t.currency = f.currency