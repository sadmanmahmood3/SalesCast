"""Generate SYNTHETIC demo sales data in the style of a small Bangladeshi
clothing brand (hoodies + punjabi).

This is NOT real sales data. It exists so the app has something realistic
to show before a business uploads its own file. Patterns built in:
  * hoodies sell in winter (Nov-Feb), almost nothing in summer
  * punjabi sales jump in the 3 weeks before Eid ul-Fitr, and before
    Eid ul-Adha and Pohela Boishakh
  * Friday and Saturday are the busiest days (Bangladesh weekend)
  * the business grows over time

Run from the project folder:
    python scripts/generate_demo_data.py
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
from core.holidays import load_holidays  # noqa: E402

START, END = "2024-01-01", "2026-09-30"
SEED = 7

# name: (price in BDT, base units per day, kind)
PRODUCTS = {
    "Oversized Hoodie - Black": (1450, 2.2, "hoodie"),
    "Oversized Hoodie - Ash Grey": (1450, 1.5, "hoodie"),
    "Zip Hoodie - Navy": (1650, 0.9, "hoodie"),
    "Classic Punjabi - White": (1850, 1.0, "punjabi"),
    "Embroidered Punjabi - Maroon": (2450, 0.5, "punjabi"),
}
CHANNELS = ["Facebook", "Website", "Instagram"]


def winter_curve(dates: pd.DatetimeIndex) -> np.ndarray:
    """Peaks around 5 January, lowest in early July."""
    day = (dates.dayofyear - 5) / 365.25
    return 0.15 + 1.85 * (0.5 + 0.5 * np.cos(2 * np.pi * day)) ** 2


def festival_boost(dates: pd.DatetimeIndex, holidays: pd.DataFrame, kind: str) -> np.ndarray:
    boost = np.ones(len(dates))
    strength = {
        "punjabi": {"Eid ul-Fitr": 6.0, "Eid ul-Adha": 2.5, "Pohela Boishakh": 2.2, "Pohela Falgun": 1.4},
        "hoodie": {"Eid ul-Fitr": 1.4, "Victory Day": 1.3, "Pohela Falgun": 1.2},
    }[kind]
    for _, h in holidays.iterrows():
        peak = strength.get(h["holiday"])
        if not peak:
            continue
        for offset in range(h["lower_window"], h["upper_window"] + 1):
            # build up towards the festival, peaking 2-4 days before it
            ramp = 1 - abs(offset + 3) / (abs(h["lower_window"]) + 1)
            idx = dates.get_indexer([h["ds"] + pd.Timedelta(days=offset)])[0]
            if idx >= 0:
                boost[idx] = max(boost[idx], 1 + (peak - 1) * max(ramp, 0.15))
    return boost


def main() -> None:
    rng = np.random.default_rng(SEED)
    holidays = load_holidays()
    dates = pd.date_range(START, END, freq="D")
    years = (dates - dates[0]).days / 365.25

    growth = 1.0 + 0.45 * years  # business grows ~45% a year
    weekday = np.select([dates.dayofweek == 4, dates.dayofweek == 5], [1.45, 1.2], 0.9)
    salary_week = np.where(dates.day <= 7, 1.15, 1.0)  # payday at month start

    rows = []
    for name, (price, base, kind) in PRODUCTS.items():
        season = winter_curve(dates) if kind == "hoodie" else np.ones(len(dates))
        rate = base * growth * weekday * salary_week * season * festival_boost(dates, holidays, kind)
        units = rng.poisson(rate)
        for day, qty in zip(dates, units):
            if qty > 0:
                rows.append(
                    {
                        "date": day.strftime("%Y-%m-%d"),
                        "product": name,
                        "quantity": int(qty),
                        "unit_price": price,
                        "channel": rng.choice(CHANNELS, p=[0.6, 0.25, 0.15]),
                    }
                )

    out = pd.DataFrame(rows)
    path = ROOT / "data" / "demo_sales.csv"
    out.to_csv(path, index=False)
    print(f"Wrote {len(out):,} rows ({out['quantity'].sum():,} units) to {path}")


if __name__ == "__main__":
    main()
