"""Turning a forecast into an inventory decision.

The question a shop owner actually asks is not "what will sales be?" but
"how many should I order, and when?". This module answers that.

Words used:
    lead time      days between placing an order with the supplier/factory
                   and the stock arriving
    coverage       how many days of sales one order should last after it arrives
    safety stock   extra units kept in case sales come in higher than forecast
    reorder point  when stock falls to this level, order now
"""

import math
from statistics import NormalDist

import numpy as np
import pandas as pd

Z_80 = 1.2816


def recommend(
    product: str,
    fc: pd.DataFrame,
    current_stock: float,
    lead_time_days: int,
    coverage_days: int,
    service_level: float = 0.95,
) -> dict:
    """Inventory advice for one product from its daily forecast.

    fc: output of core.forecast.forecast (ds, yhat, yhat_upper).
    service_level: chance of NOT running out before the next delivery
                   (0.95 = stock out at most 1 time in 20).
    """
    yhat = fc["yhat"].to_numpy()
    # Daily uncertainty, recovered from the 80% interval width.
    daily_sd = np.maximum(fc["yhat_upper"].to_numpy() - yhat, 0) / Z_80

    lead = min(lead_time_days, len(yhat))
    need_days = min(lead_time_days + coverage_days, len(yhat))

    demand_lead = float(yhat[:lead].sum())
    demand_need = float(yhat[:need_days].sum())

    z = NormalDist().inv_cdf(service_level)
    safety = float(z * math.sqrt(float((daily_sd[:lead] ** 2).sum())))
    reorder_point = demand_lead + safety
    order_qty = max(0, math.ceil(demand_need + safety - current_stock))

    # On which day does the stock run out if we order nothing?
    cumulative = np.cumsum(yhat)
    sold_out = np.nonzero(cumulative > current_stock)[0]
    if len(sold_out):
        days_left = int(sold_out[0])
        stockout_date = fc["ds"].iloc[days_left].date()
    else:
        days_left, stockout_date = None, None

    if current_stock <= reorder_point:
        status = "🔴 Order now"
    elif days_left is not None and days_left <= lead_time_days + 7:
        status = "🟠 Order this week"
    else:
        status = "🟢 Stock OK"

    # Latest day to order so the delivery arrives before stock runs out.
    # If stock runs out sooner than the lead time, it's already too late to
    # avoid a gap - say so instead of showing a date in the past.
    if days_left is None:
        order_by = "-"
    elif days_left < lead_time_days:
        order_by = "Overdue - gap before delivery"
    else:
        order_by = str((fc["ds"].iloc[0] + pd.Timedelta(days=days_left - lead_time_days)).date())

    return {
        "product": product,
        "status": status,
        "current_stock": current_stock,
        "days_of_stock_left": days_left,
        "runs_out_on": stockout_date,
        "order_by": order_by,
        "recommended_order_qty": order_qty,
        "demand_during_lead_time": round(demand_lead, 1),
        "safety_stock": math.ceil(safety),
        "reorder_point": math.ceil(reorder_point),
    }
