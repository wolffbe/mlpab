import pandas as pd

@udf(int)
def is_weekend(event_time: int) -> int:
    # event_time is epoch milliseconds
    dt = pd.to_datetime(event_time, unit='ms', utc=True)
    return 1 if dt.weekday() >= 5 else 0