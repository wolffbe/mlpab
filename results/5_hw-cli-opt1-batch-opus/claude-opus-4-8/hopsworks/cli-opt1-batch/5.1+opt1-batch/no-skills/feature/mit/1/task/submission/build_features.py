"""Feature pipeline: build derived feature group `featuresf39b89` v1.

Runs ON the Hopsworks platform as a PYTHON job. Reads the raw `transactions`
and `fx_rates` feature groups, computes the derived columns, and writes the
result to an online-enabled feature group.
"""
from datetime import datetime, timezone

import hopsworks

WEEK_MS = 7 * 24 * 60 * 60 * 1000  # 7 days in epoch milliseconds


def main():
    project = hopsworks.login()
    fs = project.get_feature_store()

    tx = fs.get_feature_group("transactions", version=1).read()
    fx = fs.get_feature_group("fx_rates", version=1).read()

    # bring everything to pandas (python engine already returns pandas)
    tx = tx.copy()
    fx = fx.copy()

    # --- amount_usd: amount * fx_rate of the row's currency -----------------
    rate = dict(zip(fx["currency"], fx["fx_rate"]))
    tx["fx_rate"] = tx["currency"].map(rate)
    tx["amount_usd"] = tx["amount"].astype(float) * tx["fx_rate"].astype(float)

    # --- is_weekend: 1 if event_time is Sat/Sun in UTC ----------------------
    def weekend(ms):
        d = datetime.fromtimestamp(int(ms) / 1000.0, tz=timezone.utc)
        return 1 if d.weekday() >= 5 else 0  # Mon=0 .. Sat=5, Sun=6

    tx["is_weekend"] = tx["event_time"].apply(weekend).astype(int)

    # --- amount_7d: sum of THIS account's amount over [t-7d, t] inclusive ---
    et = tx["event_time"].astype("int64").to_numpy()
    amt = tx["amount"].astype(float).to_numpy()
    acct = tx["account_id"].to_numpy()
    out = []
    for i in range(len(tx)):
        t = et[i]
        a = acct[i]
        lo = t - WEEK_MS
        s = 0.0
        for j in range(len(tx)):
            if acct[j] == a and lo <= et[j] <= t:
                s += amt[j]
        out.append(s)
    tx["amount_7d"] = out

    result = tx[
        ["row_id", "account_id", "event_time", "amount_usd", "is_weekend", "amount_7d"]
    ].copy()
    result["event_time"] = result["event_time"].astype("int64")
    result["is_weekend"] = result["is_weekend"].astype("int64")
    result["amount_usd"] = result["amount_usd"].astype(float)
    result["amount_7d"] = result["amount_7d"].astype(float)

    print("derived rows:", len(result))
    print(result.head().to_string())

    feat_fg = fs.get_or_create_feature_group(
        name="featuresf39b89",
        version=1,
        description="Derived transaction features (amount_usd, is_weekend, amount_7d)",
        primary_key=["row_id"],
        event_time="event_time",
        online_enabled=True,
    )
    feat_fg.insert(result, write_options={"wait_for_job": True})
    print("inserted into featuresf39b89 v1")


if __name__ == "__main__":
    main()
