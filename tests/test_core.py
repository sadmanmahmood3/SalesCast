"""Basic tests for SalesCast's core logic.

Run from the project folder:
    python -m pytest
"""

import sys
from pathlib import Path

import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from core.data import daily_series, guess_columns, parse_dates, prepare_sales  # noqa: E402
from core.forecast import backtest, forecast  # noqa: E402
from core.holidays import load_holidays  # noqa: E402
from core.insights import holiday_lift, weekday_pattern  # noqa: E402
from core.inventory import recommend  # noqa: E402

HOLIDAYS = load_holidays()


@pytest.fixture(scope="module")
def demo_sales():
    raw = pd.read_csv(ROOT / "data" / "demo_sales.csv")
    sales, _ = prepare_sales(raw, "date", "product", "quantity", "unit_price")
    return sales


def test_day_first_dates():
    parsed = parse_dates(pd.Series(["05/12/2025", "25/12/2025"]))
    assert parsed.iloc[0] == pd.Timestamp("2025-12-05")  # 5 December, not 12 May


def test_iso_dates_are_not_flipped():
    parsed = parse_dates(pd.Series(["2025-01-05", "2025-01-12"]))
    assert parsed.iloc[1] == pd.Timestamp("2025-01-12")


def test_guess_columns():
    df = pd.DataFrame(columns=["Order Date", "Item Name", "Qty", "Unit Price"])
    assert guess_columns(df) == {"date": "Order Date", "product": "Item Name", "quantity": "Qty", "price": "Unit Price"}


def test_cleaning_drops_bad_rows():
    raw = pd.DataFrame(
        {"date": ["2025-01-01", "oops", "2025-01-02", "2025-01-03"], "qty": [2, 1, -1, "abc"]}
    )
    sales, report = prepare_sales(raw, "date", None, "qty")
    assert len(sales) == 1
    assert report["dropped_bad_date"] == 1
    assert report["dropped_zero_or_negative"] == 1
    assert report["dropped_bad_quantity"] == 1


def test_missing_days_become_zero():
    sales = pd.DataFrame(
        {"date": pd.to_datetime(["2025-01-01", "2025-01-04"]), "product": "A", "units": [3, 1], "revenue": [0, 0]}
    )
    daily = daily_series(sales)
    assert daily["y"].tolist() == [3, 0, 0, 1]


@pytest.mark.parametrize("engine", ["fast", "prophet"])
def test_forecast_shape_and_no_negatives(demo_sales, engine):
    fc = forecast(daily_series(demo_sales), 60, HOLIDAYS, engine)
    assert len(fc) == 60
    assert (fc["yhat"] >= 0).all()
    assert (fc["yhat_lower"] <= fc["yhat_upper"]).all()


def test_backtest_total_is_close(demo_sales):
    result = backtest(daily_series(demo_sales), HOLIDAYS, "fast")
    assert abs(result["total_error_%"]) < 25


def test_too_little_history_is_rejected():
    short = pd.DataFrame({"ds": pd.date_range("2025-01-01", periods=10), "y": 1.0})
    with pytest.raises(ValueError):
        forecast(short, 30, HOLIDAYS, "fast")


def test_friday_is_busiest_in_demo(demo_sales):
    pattern = weekday_pattern(daily_series(demo_sales))
    assert pattern.loc[pattern["avg_units_per_day"].idxmax(), "weekday"] == "Friday"


def test_punjabi_eid_rush_detected(demo_sales):
    lift = holiday_lift(daily_series(demo_sales, "Classic Punjabi - White"), HOLIDAYS)
    eid = lift.loc[lift["festival"] == "Eid ul-Fitr", "lift_x"].iloc[0]
    assert eid > 2


def test_inventory_math():
    fc = pd.DataFrame(
        {"ds": pd.date_range("2026-01-01", periods=60), "yhat": 10.0, "yhat_lower": 10.0, "yhat_upper": 10.0}
    )
    # No uncertainty -> no safety stock. 10/day, 14 day lead time, 30 days coverage.
    r = recommend("A", fc, current_stock=100, lead_time_days=14, coverage_days=30, service_level=0.95)
    assert r["reorder_point"] == 140
    assert r["status"] == "🔴 Order now"
    assert r["recommended_order_qty"] == 440 - 100
    assert r["days_of_stock_left"] == 10  # cumulative sales pass 100 on day 11 (index 10)

    r = recommend("A", fc, current_stock=1000, lead_time_days=14, coverage_days=30)
    assert r["status"] == "🟢 Stock OK"
    assert r["recommended_order_qty"] == 0
