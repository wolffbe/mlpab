# Schema

- **transactions.csv**: row_id (unique key), account_id, event_time (bigint, epoch MILLISECONDS), amount, currency
- **fx_rates.csv**: currency, fx_rate (multiply amount by this to get USD)
