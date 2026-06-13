"""Build the airquality raw fixture — ONE-TIME, OFFLINE.

The mlfs-book air-quality service predicts PM2.5 from weather. The WAQI live
feed (with your token) only returns the CURRENT reading + a short forecast, not
the multi-year history a regression task needs — but it DOES resolve a real
station's coordinates. So the default build:

  * --waqi-token + --city : resolve the station's lat/lon from WAQI, then pull
    the historical daily PM2.5 (Open-Meteo air-quality archive, KEYLESS) and
    weather (Open-Meteo weather archive, KEYLESS) at those coordinates.
  * --aqicn-csv + --lat/--lon : alternative — a PM2.5 history CSV you downloaded
    from the AQICN data platform (columns: date, pm25), weather from Open-Meteo.
  * --synthetic : offline schema-correct placeholder (no network, no key).

All sources are merged on `date` and committed as `airquality_raw.csv`, so the
testbed never touches the network or a key at run time — it just reads the CSV.
The schema is identical across modes, so swapping fixtures needs no code changes.

    python -m evals.capstone._data.build_airquality --waqi-token <TOKEN> --city stockholm
    python -m evals.capstone._data.build_airquality --aqicn-csv hist.csv --lat 59.33 --lon 18.07
    python -m evals.capstone._data.build_airquality --synthetic
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).parent / "airquality_raw.csv"
COLUMNS = ["date", "pm25", "temperature", "humidity", "wind_speed", "pressure", "precipitation"]
SYNTH_SEED = 4242


def _synthetic(days: int) -> pd.DataFrame:
    """Offline placeholder: PM2.5 = weather signal + AR(1) + seasonality + noise,
    so a weather/lag model genuinely beats the mean (gates depend on this)."""
    rng = np.random.default_rng(SYNTH_SEED)
    dates = pd.date_range("2023-01-01", periods=days, freq="D")
    doy = dates.dayofyear.to_numpy()
    season = 10 * np.cos(2 * np.pi * (doy - 15) / 365)            # winter peak
    temp = 12 + 11 * np.cos(2 * np.pi * (doy - 200) / 365) + rng.normal(0, 2.5, days)
    humidity = np.clip(70 - 0.4 * (temp - 12) + rng.normal(0, 8, days), 20, 100)
    wind = np.clip(rng.gamma(2.0, 1.6, days), 0.2, None)
    pressure = 1013 + rng.normal(0, 7, days)
    precip = rng.gamma(0.4, 4.0, days) * (rng.random(days) < 0.35)
    pm = np.empty(days)
    pm[0] = 25.0
    for i in range(1, days):                                     # autocorrelation
        drive = (18 + season[i] - 2.4 * wind[i] + 0.12 * humidity[i]
                 - 3.0 * np.log1p(precip[i]) + 0.05 * (pressure[i] - 1013))
        pm[i] = max(1.0, 0.55 * pm[i - 1] + 0.45 * drive + rng.normal(0, 3.0))
    df = pd.DataFrame({"date": dates.strftime("%Y-%m-%d"), "pm25": np.round(pm, 1),
                       "temperature": np.round(temp, 1), "humidity": np.round(humidity, 1),
                       "wind_speed": np.round(wind, 2), "pressure": np.round(pressure, 1),
                       "precipitation": np.round(precip, 2)})
    return df[COLUMNS]


def _waqi_geo(token: str, city: str) -> tuple[float, float, str]:
    """Resolve a real WAQI station's coordinates (validates the token)."""
    import requests
    r = requests.get(f"https://api.waqi.info/feed/{city}/", params={"token": token}, timeout=30)
    r.raise_for_status()
    js = r.json()
    if js.get("status") != "ok":
        raise RuntimeError(f"WAQI error for {city!r}: {js.get('data')}")
    c = js["data"]["city"]
    lat, lon = c["geo"]
    return float(lat), float(lon), c["name"]


def _open_meteo_pm25(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    """Historical daily-mean PM2.5 from Open-Meteo's air-quality archive (keyless)."""
    import requests
    r = requests.get("https://air-quality-api.open-meteo.com/v1/air-quality", params={
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "hourly": "pm2_5", "timezone": "UTC"}, timeout=120)
    r.raise_for_status()
    h = r.json()["hourly"]
    s = pd.DataFrame({"dt": pd.to_datetime(h["time"]), "pm25": h["pm2_5"]}).dropna()
    daily = s.groupby(s["dt"].dt.strftime("%Y-%m-%d"))["pm25"].mean().round(1)
    return daily.rename_axis("date").reset_index()


def _open_meteo_weather(lat: float, lon: float, start: str, end: str) -> pd.DataFrame:
    import requests
    r = requests.get("https://archive-api.open-meteo.com/v1/archive", params={
        "latitude": lat, "longitude": lon, "start_date": start, "end_date": end,
        "daily": ("temperature_2m_mean,relative_humidity_2m_mean,wind_speed_10m_max,"
                  "surface_pressure_mean,precipitation_sum"),
        "timezone": "UTC"}, timeout=120)
    r.raise_for_status()
    d = r.json()["daily"]
    return pd.DataFrame({"date": d["time"], "temperature": d["temperature_2m_mean"],
                         "humidity": d["relative_humidity_2m_mean"],
                         "wind_speed": d["wind_speed_10m_max"],
                         "pressure": d["surface_pressure_mean"],
                         "precipitation": d["precipitation_sum"]})


def _real(lat: float, lon: float, start: str, end: str,
          aqicn_csv: Path | None) -> pd.DataFrame:
    if aqicn_csv is not None:
        pm = pd.read_csv(aqicn_csv)
        pm.columns = [c.strip().lower() for c in pm.columns]
        if "pm25" not in pm.columns:
            for c in ("pm2.5", "pm2_5", "value", "median"):
                if c in pm.columns:
                    pm = pm.rename(columns={c: "pm25"}); break
        pm["date"] = pd.to_datetime(pm["date"]).dt.strftime("%Y-%m-%d")
        pm = pm[["date", "pm25"]].dropna().drop_duplicates("date")
        start, end = pm["date"].min(), pm["date"].max()
    else:
        pm = _open_meteo_pm25(lat, lon, start, end)
    wx = _open_meteo_weather(lat, lon, start, end)
    return pm.merge(wx, on="date", how="inner").dropna().sort_values("date")[COLUMNS]


def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--synthetic", action="store_true",
                    help="emit an offline schema-correct placeholder (no key)")
    ap.add_argument("--days", type=int, default=900, help="synthetic horizon")
    ap.add_argument("--waqi-token", help="WAQI token (resolves the station's lat/lon)")
    ap.add_argument("--city", default="stockholm", help="WAQI city/station slug")
    ap.add_argument("--aqicn-csv", type=Path, help="downloaded PM2.5 history CSV")
    ap.add_argument("--lat", type=float)
    ap.add_argument("--lon", type=float)
    ap.add_argument("--start", default="2022-08-01")
    ap.add_argument("--end", default="2026-06-01")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args(argv)
    if args.synthetic:
        df = _synthetic(args.days)
        src = "synthetic placeholder"
    elif args.waqi_token:
        lat, lon, name = _waqi_geo(args.waqi_token, args.city)
        print(f"[build_airquality] WAQI station: {name} ({lat:.4f}, {lon:.4f})")
        df = _real(lat, lon, args.start, args.end, None)
        src = f"WAQI:{name} + open-meteo {args.start}..{args.end}"
    elif args.aqicn_csv and args.lat is not None and args.lon is not None:
        df = _real(args.lat, args.lon, args.start, args.end, args.aqicn_csv)
        src = f"{args.aqicn_csv} + open-meteo"
    else:
        ap.error("need --waqi-token (+--city), or --aqicn-csv +--lat/--lon, or --synthetic")
    df.dropna().to_csv(args.out, index=False)
    print(f"[build_airquality] wrote {len(df)} daily rows ({src}) -> {args.out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
