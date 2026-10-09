"""Seasonal patterns hidden in the sales history.

These are simple averages - no model - so they are easy to trust and explain:
which weekday sells most, which months are busy, and how much each festival
lifts sales compared with a normal day.
"""

import pandas as pd

from core.holidays import holiday_window_days

WEEKDAYS = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]
MONTHS = ["Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec"]


def weekday_pattern(daily: pd.DataFrame) -> pd.DataFrame:
    """Average units sold on each day of the week."""
    out = daily.groupby(daily["ds"].dt.dayofweek)["y"].mean().reindex(range(7), fill_value=0)
    return pd.DataFrame({"weekday": WEEKDAYS, "avg_units_per_day": out.to_numpy().round(2)})


def monthly_pattern(daily: pd.DataFrame) -> pd.DataFrame:
    """Average units per day in each calendar month (fair even for short months)."""
    out = daily.groupby(daily["ds"].dt.month)["y"].mean().reindex(range(1, 13))
    return pd.DataFrame({"month": MONTHS, "avg_units_per_day": out.to_numpy().round(2)})


def holiday_lift(daily: pd.DataFrame, holidays: pd.DataFrame, compare_days: int = 30) -> pd.DataFrame:
    """How much busier each festival's shopping window is than normal days
    around the same time of year.

    For every festival in the data, the window is compared with the
    `compare_days` days just before and after it (skipping other festivals).
    Comparing with nearby days - not the whole year - keeps the season out
    of it: Eid ul-Adha in summer shouldn't look "slow" just because
    hoodies don't sell in June.

    lift = 2.0 means twice the normal daily sales during the window.
    """
    first, last = daily["ds"].min(), daily["ds"].max()
    by_date = daily.set_index("ds")["y"]
    all_window_days = set(holiday_window_days(holidays)["date"])

    per_event = []
    for _, h in holidays.iterrows():
        start = h["ds"] + pd.Timedelta(days=int(h["lower_window"]))
        end = h["ds"] + pd.Timedelta(days=int(h["upper_window"]))
        if start < first or end > last:
            continue  # only use festivals fully inside the data
        window = by_date[start:end]
        around = pd.concat(
            [
                by_date[start - pd.Timedelta(days=compare_days) : start - pd.Timedelta(days=1)],
                by_date[end + pd.Timedelta(days=1) : end + pd.Timedelta(days=compare_days)],
            ]
        )
        around = around[~around.index.isin(all_window_days)]
        if len(around) < 7:
            continue
        per_event.append(
            {"festival": h["holiday"], "during": window.mean(), "normal": around.mean()}
        )

    cols = ["festival", "avg_units_per_day", "normal_day_units", "lift_x", "times_seen"]
    if not per_event:
        return pd.DataFrame(columns=cols)
    ev = pd.DataFrame(per_event)
    out = ev.groupby("festival").agg(
        avg_units_per_day=("during", "mean"), normal_day_units=("normal", "mean"), times_seen=("during", "size")
    )
    out = out[out["normal_day_units"] > 0]
    out["lift_x"] = out["avg_units_per_day"] / out["normal_day_units"]
    out = out.round(2).reset_index()[cols]
    return out.sort_values("lift_x", ascending=False).reset_index(drop=True)


def recent_trend(daily: pd.DataFrame, days: int = 30) -> dict:
    """Compare the last `days` days with the same number of days before them."""
    last = daily["y"].iloc[-days:].sum()
    before = daily["y"].iloc[-2 * days : -days].sum()
    change = 100 * (last - before) / before if before > 0 else None
    return {"last_period_units": float(last), "previous_period_units": float(before), "change_%": change}
