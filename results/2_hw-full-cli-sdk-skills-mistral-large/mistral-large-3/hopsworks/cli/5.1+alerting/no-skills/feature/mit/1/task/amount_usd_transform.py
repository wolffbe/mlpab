@udf(float)
def amount_usd(amount: float, fx_rate: float) -> float:
    return amount * fx_rate