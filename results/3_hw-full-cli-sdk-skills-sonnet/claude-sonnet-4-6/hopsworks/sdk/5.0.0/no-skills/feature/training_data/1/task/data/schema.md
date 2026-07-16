# Schema

All tables join on `account_id`. `event_time` (bigint, epoch MILLISECONDS) is when the row became valid.

- **transactions.csv**: account_id, event_time (epoch ms), amount, balance
- **profiles.csv**: account_id, event_time (epoch ms), credit_score, tier
- **activity.csv**: account_id, event_time (epoch ms), sessions_7d
- **account_health.csv**: account_id, event_time (epoch ms), health_score
- **labels.csv**: account_id, label_time (epoch ms), churned (1 = churned)
