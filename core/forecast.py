"""Forecasting future daily sales.

Two engines:

* "prophet"  - Meta's Prophet library. The main engine. Learns trend,
               weekly pattern, yearly pattern and Bangladesh festival effects.
* "fast"     - a small built-in model (linear regression on the same kinds of
               features). No extra install, runs in a blink. Used as a backup
               when Prophet isn't installed, and as a baseline to compare with.

Both engines model log(1 + units) instead of raw units. That keeps forecasts
from going negative and lets effects scale with the size of the business
(an Eid rush adds "3x sales", not "+20 hoodies").
"""

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from core.holidays import holiday_window_days

Z_80 = 1.2816  # 80% prediction interval -> +/- 1.28 standard deviations

try:  # Prophet is optional so the app still runs without it
    from prophet import Prophet

    PROPHET_AVAILABLE = True
except Exception:  # ImportError, or a broken install on Windows
    PROPHET_AVAILABLE = False


def available_engines() -> list[str]:
    return (["prophet"] if PROPHET_AVAILABLE else []) + ["fast"]


# ---------------------------------------------------------------- Prophet ---
def _prophet_forecast(history: pd.DataFrame, horizon: int, holidays: pd.DataFrame) -> pd.DataFrame:
    import logging

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)

    train = history.assign(y=np.log1p(history["y"]))
    model = Prophet(
        holidays=holidays,
        weekly_seasonality=True,
        # A yearly pattern can only be learned from at least a year of data.
        yearly_seasonality=len(history) >= 365,
        daily_seasonality=False,
        interval_width=0.80,
    )
    model.fit(train)

    future = model.make_future_dataframe(periods=horizon, freq="D")
    pred = model.predict(future)

    fitted = pred["yhat"].iloc[: len(train)].to_numpy()
    smear = _smearing_factor(train["y"].to_numpy(), fitted)

    fc = pred.iloc[len(train):][["ds", "yhat", "yhat_lower", "yhat_upper"]].copy()
    fc["yhat"] = np.expm1(fc["yhat"]) * smear
    fc["yhat_lower"] = np.expm1(fc["yhat_lower"])
    fc["yhat_upper"] = np.expm1(fc["yhat_upper"]) * smear
    return fc


# ------------------------------------------------------------- Fast model ---
def _features(dates: pd.Series, start: pd.Timestamp, holidays: pd.DataFrame, yearly: bool) -> pd.DataFrame:
    """Build the inputs the fast model learns from, one row per day."""
    dates = pd.to_datetime(pd.Series(dates)).reset_index(drop=True)
    X = pd.DataFrame(index=dates.index)

    # Trend: years since the first day of data.
    X["trend"] = (dates - start).dt.days / 365.25

    # Weekly pattern: one column per weekday (Monday is the baseline).
    for d in range(1, 7):
        X[f"weekday_{d}"] = (dates.dt.dayofweek == d).astype(float)

    # Yearly pattern: smooth sine/cosine waves over the year.
    if yearly:
        day_of_year = dates.dt.dayofyear / 365.25
        for k in range(1, 5):
            X[f"sin_{k}"] = np.sin(2 * np.pi * k * day_of_year)
            X[f"cos_{k}"] = np.cos(2 * np.pi * k * day_of_year)

    # Festivals: one column per (holiday, days-before/after) pair, so the
    # model can learn that "10 days before Eid" is busier than "20 days before".
    windows = holiday_window_days(holidays)
    windows["key"] = windows["holiday"] + "_" + windows["offset"].astype(str)
    lookup = windows.groupby("date")["key"].apply(list).to_dict()
    for key in sorted(windows["key"].unique()):
        X[key] = 0.0
    for i, day in enumerate(dates):
        for key in lookup.get(day, []):
            X.at[i, key] = 1.0
    return X


def _fast_forecast(history: pd.DataFrame, horizon: int, holidays: pd.DataFrame) -> pd.DataFrame:
    start = history["ds"].min()
    yearly = len(history) >= 365
    y = np.log1p(history["y"].to_numpy())

    X_train = _features(history["ds"], start, holidays, yearly)
    model = Ridge(alpha=1.0).fit(X_train, y)

    fitted = model.predict(X_train)
    resid_sd = float(np.std(y - fitted))
    smear = _smearing_factor(y, fitted)

    future_dates = pd.date_range(history["ds"].max() + pd.Timedelta(days=1), periods=horizon, freq="D")
    log_pred = model.predict(_features(future_dates, start, holidays, yearly))

    return pd.DataFrame(
        {
            "ds": future_dates,
            "yhat": np.expm1(log_pred) * smear,
            "yhat_lower": np.expm1(log_pred - Z_80 * resid_sd),
            "yhat_upper": np.expm1(log_pred + Z_80 * resid_sd) * smear,
        }
    )


# ------------------------------------------------------------- Shared ------
def _smearing_factor(y_log: np.ndarray, fitted_log: np.ndarray) -> float:
    """Correct the bias from modelling log(units).

    Converting a log forecast straight back gives the *typical* day, which
    is a bit lower than the *average* day. Averaging exp(residuals) - Duan's
    smearing estimate - scales it back up so totals come out right.
    """
    return float(np.mean(np.exp(y_log - fitted_log)))


def forecast(history: pd.DataFrame, horizon: int, holidays: pd.DataFrame, engine: str = "prophet") -> pd.DataFrame:
    """Forecast the next `horizon` days.

    history: columns ds (date) and y (units per day, zeros included).
    Returns: ds, yhat (expected units), yhat_lower / yhat_upper (80% range).
    """
    if len(history) < 30:
        raise ValueError("Need at least 30 days of sales history to forecast.")
    if engine == "prophet" and not PROPHET_AVAILABLE:
        engine = "fast"

    if engine == "prophet":
        fc = _prophet_forecast(history, horizon, holidays)
    else:
        fc = _fast_forecast(history, horizon, holidays)

    for col in ["yhat", "yhat_lower", "yhat_upper"]:
        fc[col] = fc[col].clip(lower=0)
    return fc.reset_index(drop=True)


def backtest(history: pd.DataFrame, holidays: pd.DataFrame, engine: str = "prophet", holdout_days: int = 28) -> dict:
    """How accurate would the model have been recently?

    Hide the last `holdout_days`, forecast them from the data before, and
    compare with what really happened.

    WAPE (weighted absolute percentage error) = total |error| / total sales.
    It works even on days with zero sales, unlike plain MAPE.
    """
    train, test = history.iloc[:-holdout_days], history.iloc[-holdout_days:]
    fc = forecast(train, holdout_days, holidays, engine)
    actual = test["y"].to_numpy()
    predicted = fc["yhat"].to_numpy()
    total = actual.sum()
    return {
        "holdout_days": holdout_days,
        "actual_units": float(total),
        "forecast_units": float(predicted.sum()),
        "wape_%": float(100 * np.abs(actual - predicted).sum() / total) if total > 0 else float("nan"),
        "total_error_%": float(100 * (predicted.sum() - total) / total) if total > 0 else float("nan"),
        "comparison": pd.DataFrame({"ds": test["ds"].to_numpy(), "actual": actual, "forecast": predicted}),
    }
