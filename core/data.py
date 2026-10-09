"""Loading and cleaning sales data.

Any business can upload its own CSV. The columns don't need fixed names:
the app guesses which column is the date, product, quantity and price,
and the user can change the guess in the sidebar.
"""

import pandas as pd

# Words we look for when guessing which column is which.
_GUESSES = {
    "date": ["date", "order_date", "day", "time", "created", "timestamp"],
    "product": ["product", "item", "sku", "name", "category", "title"],
    "quantity": ["quantity", "qty", "units", "count", "pcs", "sold"],
    "price": ["unit_price", "price", "rate", "mrp"],
}


def guess_columns(df: pd.DataFrame) -> dict:
    """Guess which column holds date / product / quantity / price.

    Returns a dict like {"date": "Order Date", "product": "Item", ...}.
    A value is None if nothing looked right.
    """
    lower = {c: str(c).strip().lower().replace(" ", "_") for c in df.columns}
    result = {}
    used = set()
    for role, words in _GUESSES.items():
        found = None
        # exact match first, then "contains"
        for word in words:
            for col, low in lower.items():
                if col not in used and low == word:
                    found = col
                    break
            if found:
                break
        if not found:
            for word in words:
                for col, low in lower.items():
                    if col not in used and word in low:
                        found = col
                        break
                if found:
                    break
        result[role] = found
        if found:
            used.add(found)
    return result


def parse_dates(values: pd.Series, dayfirst: bool = True) -> pd.Series:
    """Parse a column of dates written in any common style.

    ISO dates (2025-12-25) are never ambiguous, so they are read as-is.
    Other styles like 05/12/2025 are ambiguous: Bangladeshi spreadsheets
    usually mean 5 December (day first), which is the default here.
    """
    iso = pd.to_datetime(values, errors="coerce", format="ISO8601")
    if iso.notna().mean() >= 0.95:
        return iso
    return pd.to_datetime(values, errors="coerce", dayfirst=dayfirst, format="mixed")


def prepare_sales(
    df: pd.DataFrame,
    date_col: str,
    product_col: str | None,
    qty_col: str,
    price_col: str | None = None,
    dayfirst: bool = True,
) -> tuple[pd.DataFrame, dict]:
    """Turn a raw sales table into a clean one.

    Output columns: date, product, units, revenue
    (revenue is units x price, or NaN when there is no price column).

    Also returns a small report of what was cleaned, so the user can see
    that nothing was silently thrown away.
    """
    report = {"rows_in": len(df)}
    out = pd.DataFrame()

    out["date"] = parse_dates(df[date_col], dayfirst).dt.normalize()

    out["product"] = (
        df[product_col].astype(str).str.strip() if product_col else "All products"
    )
    out["units"] = pd.to_numeric(df[qty_col], errors="coerce")

    if price_col:
        price = pd.to_numeric(df[price_col], errors="coerce")
        out["revenue"] = out["units"] * price
    else:
        out["revenue"] = float("nan")

    bad_date = out["date"].isna()
    bad_qty = out["units"].isna()
    not_sale = out["units"] <= 0  # returns / cancellations / zero rows
    report["dropped_bad_date"] = int(bad_date.sum())
    report["dropped_bad_quantity"] = int((bad_qty & ~bad_date).sum())
    report["dropped_zero_or_negative"] = int((not_sale & ~bad_qty & ~bad_date).sum())

    out = out[~bad_date & ~bad_qty & ~not_sale].copy()
    out = out.sort_values("date").reset_index(drop=True)
    report["rows_kept"] = len(out)
    report["has_price"] = bool(price_col) and out["revenue"].notna().any()
    return out, report


def daily_series(sales: pd.DataFrame, product: str | None = None) -> pd.DataFrame:
    """Total units sold per day, with ZERO filled in for days with no sales.

    Filling the gaps matters: a day with no row in the file is a day with
    0 sales, and forecasting models need to see those zeros.

    Returns columns ds (date) and y (units) - the names Prophet expects.
    The range always runs to the last date in the whole dataset, so every
    product's series ends on the same day.
    """
    start, end = sales["date"].min(), sales["date"].max()
    part = sales if product is None else sales[sales["product"] == product]
    daily = part.groupby("date")["units"].sum()
    all_days = pd.date_range(start, end, freq="D")
    daily = daily.reindex(all_days, fill_value=0)
    return pd.DataFrame({"ds": daily.index, "y": daily.values.astype(float)})


def product_summary(sales: pd.DataFrame) -> pd.DataFrame:
    """Units, revenue and share of sales for each product, best seller first."""
    summary = (
        sales.groupby("product")
        .agg(units=("units", "sum"), revenue=("revenue", "sum"), days_sold=("date", "nunique"))
        .sort_values("units", ascending=False)
    )
    summary["share_of_units_%"] = (100 * summary["units"] / summary["units"].sum()).round(1)
    return summary.reset_index()
