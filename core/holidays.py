"""Bangladesh holiday calendar.

Shopping in Bangladesh is driven by festivals: people buy clothes in the
weeks BEFORE Eid, before Pohela Boishakh, etc. Generic forecasting tools
don't know this, which is what makes SalesCast useful for local businesses.

The dates live in data/bd_holidays.csv so anyone can add or fix a row
(for example, Durga Puja, or a shop's own sale days) without touching code.

Columns (Prophet's holiday format):
    holiday       name of the event
    ds            date of the event
    lower_window  how many days BEFORE the event its effect starts (negative number)
    upper_window  how many days AFTER the event its effect lasts
"""

from pathlib import Path

import pandas as pd

HOLIDAY_FILE = Path(__file__).resolve().parent.parent / "data" / "bd_holidays.csv"


def load_holidays(path: Path = HOLIDAY_FILE) -> pd.DataFrame:
    """Read the holiday calendar CSV into a DataFrame."""
    df = pd.read_csv(path)
    df["ds"] = pd.to_datetime(df["ds"])
    df["lower_window"] = df["lower_window"].astype(int)
    df["upper_window"] = df["upper_window"].astype(int)
    return df


def holiday_window_days(holidays: pd.DataFrame) -> pd.DataFrame:
    """Expand each holiday into one row per day of its shopping window.

    Returns columns: date, holiday, offset (days relative to the holiday;
    -7 means "one week before").
    """
    rows = []
    for _, h in holidays.iterrows():
        for offset in range(h["lower_window"], h["upper_window"] + 1):
            rows.append(
                {
                    "date": h["ds"] + pd.Timedelta(days=offset),
                    "holiday": h["holiday"],
                    "offset": offset,
                }
            )
    return pd.DataFrame(rows, columns=["date", "holiday", "offset"])
