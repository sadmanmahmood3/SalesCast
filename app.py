"""SalesCast - sales forecasting & inventory planning for small businesses.

Run with:
    streamlit run app.py
"""

import io
from pathlib import Path

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

from core.data import daily_series, guess_columns, prepare_sales, product_summary
from core.forecast import PROPHET_AVAILABLE, available_engines, backtest, forecast
from core.holidays import load_holidays
from core.insights import holiday_lift, monthly_pattern, recent_trend, weekday_pattern
from core.inventory import recommend

ROOT = Path(__file__).resolve().parent
DEMO_FILE = ROOT / "data" / "demo_sales.csv"
TEMPLATE_FILE = ROOT / "data" / "sales_template.csv"

ACTUAL_COLOR = "#2a78d6"  # blue  - what really happened
FORECAST_COLOR = "#eb6834"  # orange - what the model expects
ALL = "All products"

st.set_page_config(page_title="SalesCast", page_icon="📈", layout="wide")


# ------------------------------------------------------------ cached work ---
@st.cache_data
def read_csv(content: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.BytesIO(content))
    except UnicodeDecodeError:  # files saved from older Excel versions
        return pd.read_csv(io.BytesIO(content), encoding="latin-1")


@st.cache_data
def get_holidays() -> pd.DataFrame:
    return load_holidays()


@st.cache_data(show_spinner=False)
def run_forecast(history: pd.DataFrame, horizon: int, engine: str) -> pd.DataFrame:
    return forecast(history, horizon, get_holidays(), engine)


@st.cache_data(show_spinner=False)
def run_backtest(history: pd.DataFrame, engine: str) -> dict:
    return backtest(history, get_holidays(), engine)


def style_chart(fig: go.Figure, height: int = 380) -> go.Figure:
    fig.update_layout(
        height=height,
        margin=dict(l=10, r=10, t=30, b=10),
        hovermode="x unified",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    return fig


def bar_chart(x, y, y_title: str, height: int = 320) -> go.Figure:
    fig = go.Figure(go.Bar(x=x, y=y, marker_color=ACTUAL_COLOR, hovertemplate="%{x}: %{y:.2f}<extra></extra>"))
    fig.update_yaxes(title=y_title)
    return style_chart(fig, height)


# ---------------------------------------------------------------- sidebar ---
st.sidebar.title("📈 SalesCast")
st.sidebar.caption("Forecast sales · spot seasons · plan stock")

source = st.sidebar.radio("Sales data", ["Demo store (synthetic data)", "Upload my own CSV"])

if source.startswith("Demo"):
    raw = read_csv(DEMO_FILE.read_bytes())
    cols = guess_columns(raw)
    dayfirst = True
else:
    uploaded = st.sidebar.file_uploader("Your sales file (.csv)", type="csv")
    st.sidebar.download_button(
        "Download a blank template", TEMPLATE_FILE.read_bytes(), "sales_template.csv", "text/csv"
    )
    if uploaded is None:
        st.title("Upload your sales to get started")
        st.markdown(
            """
            SalesCast needs a CSV with **one row per sale** (or per product per day).
            Column names can be anything - you'll match them up in the sidebar.

            | Column | Required? | Example |
            |---|---|---|
            | Date | ✅ | 2026-03-14 or 14/03/2026 |
            | Quantity sold | ✅ | 2 |
            | Product name | recommended | Oversized Hoodie - Black |
            | Unit price | optional (for revenue) | 1450 |

            You need at least **30 days** of sales; a full **year** lets SalesCast learn
            your seasons and festival rushes. Not sure? Try the demo store first.
            """
        )
        st.stop()
    raw = read_csv(uploaded.getvalue())
    guess = guess_columns(raw)
    options = list(raw.columns)
    none_options = ["(none)"] + options

    def pick(label, role, allow_none=False):
        choices = none_options if allow_none else options
        default = guess[role] if guess[role] in choices else choices[0]
        value = st.sidebar.selectbox(label, choices, index=choices.index(default))
        return None if value == "(none)" else value

    st.sidebar.subheader("Match your columns")
    cols = {
        "date": pick("Date column", "date"),
        "quantity": pick("Quantity column", "quantity"),
        "product": pick("Product column", "product", allow_none=True),
        "price": pick("Price column", "price", allow_none=True),
    }
    dayfirst = st.sidebar.checkbox("Dates are day-first (14/03/2026)", value=True)

st.sidebar.subheader("Forecast settings")
horizon = st.sidebar.slider("Days to forecast", 14, 180, 90, step=7)
engines = available_engines()
engine_labels = {"prophet": "Prophet (most accurate)", "fast": "Fast built-in model"}
engine = st.sidebar.selectbox("Forecasting engine", engines, format_func=engine_labels.get)
if not PROPHET_AVAILABLE:
    st.sidebar.info("Prophet isn't installed, so the fast built-in model is used. `pip install prophet` to enable it.")

# ------------------------------------------------------------ clean data ---
try:
    sales, report = prepare_sales(raw, cols["date"], cols["product"], cols["quantity"], cols["price"], dayfirst)
except Exception as err:  # wrong column picked, unreadable values, ...
    st.error(f"Couldn't read the sales data with these columns: {err}")
    st.stop()

if sales.empty:
    st.error("No valid sales rows found. Check that the date and quantity columns are matched correctly.")
    st.stop()

holidays = get_holidays()
products = sorted(sales["product"].unique())
product_options = [ALL] + [p for p in products if p != ALL]
total_daily = daily_series(sales)
has_revenue = report["has_price"]

if len(total_daily) < 30:
    st.error(f"Only {len(total_daily)} days of data. SalesCast needs at least 30 days of sales history.")
    st.stop()

st.title("SalesCast")
if source.startswith("Demo"):
    st.caption(
        "Showing a **demo clothing store with synthetic data** (hoodies + punjabi, Bangladesh). "
        "Switch to *Upload my own CSV* in the sidebar to analyse your business."
    )

tab_overview, tab_forecast, tab_seasons, tab_stock, tab_data = st.tabs(
    ["Overview", "Forecast", "Seasonal patterns", "Inventory planner", "Data"]
)

# --------------------------------------------------------------- overview ---
with tab_overview:
    trend = recent_trend(total_daily)
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Units sold", f"{int(sales['units'].sum()):,}")
    c2.metric("Revenue (৳)", f"{sales['revenue'].sum():,.0f}" if has_revenue else "-")
    c3.metric("Products", len(products))
    c4.metric(
        "Last 30 days",
        f"{trend['last_period_units']:,.0f} units",
        f"{trend['change_%']:+.1f}% vs previous 30" if trend["change_%"] is not None else None,
    )
    st.caption(f"Data from {sales['date'].min():%d %b %Y} to {sales['date'].max():%d %b %Y}")

    st.subheader("Daily units sold")
    rolling = total_daily["y"].rolling(7, min_periods=1).mean()
    fig = go.Figure()
    fig.add_trace(
        go.Scatter(x=total_daily["ds"], y=total_daily["y"], name="Daily units", mode="lines",
                   line=dict(color=ACTUAL_COLOR, width=1), opacity=0.35)
    )
    fig.add_trace(
        go.Scatter(x=total_daily["ds"], y=rolling, name="7-day average", mode="lines",
                   line=dict(color=ACTUAL_COLOR, width=2))
    )
    st.plotly_chart(style_chart(fig))

    st.subheader("Products")
    summary = product_summary(sales)
    if not has_revenue:
        summary = summary.drop(columns="revenue")
    st.dataframe(summary, hide_index=True)

# --------------------------------------------------------------- forecast ---
with tab_forecast:
    choice = st.selectbox("Product", product_options, key="fc_product")
    history = total_daily if choice == ALL else daily_series(sales, choice)

    with st.spinner("Forecasting..."):
        fc = run_forecast(history, horizon, engine)

    c1, c2, c3 = st.columns(3)
    c1.metric(f"Expected units, next {horizon} days", f"{fc['yhat'].sum():,.0f}")
    same_period_last_year = history[history["ds"] > history["ds"].max() - pd.Timedelta(days=365)].head(horizon)
    if len(history) >= 365 + horizon:
        c2.metric("Same period last year", f"{same_period_last_year['y'].sum():,.0f}")
    peak = fc.loc[fc["yhat"].idxmax()]
    c3.metric("Busiest forecast day", f"{peak['ds']:%d %b}", f"~{peak['yhat']:.0f} units", delta_color="off")

    show_from = history["ds"].max() - pd.Timedelta(days=365)
    recent = history[history["ds"] > show_from]
    fig = go.Figure()
    fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_upper"], mode="lines", line=dict(width=0),
                             showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat_lower"], mode="lines", line=dict(width=0),
                             fill="tonexty", fillcolor="rgba(235,104,52,0.18)", name="Likely range (80%)",
                             hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=recent["ds"], y=recent["y"].rolling(7, min_periods=1).mean(),
                             name="Actual (7-day avg)", mode="lines", line=dict(color=ACTUAL_COLOR, width=2)))
    fig.add_trace(go.Scatter(x=fc["ds"], y=fc["yhat"], name="Forecast", mode="lines",
                             line=dict(color=FORECAST_COLOR, width=2)))
    fig.update_yaxes(title="Units per day", rangemode="tozero")
    st.plotly_chart(style_chart(fig, 420))

    with st.expander("How accurate is this? (backtest on the last 28 days)"):
        with st.spinner("Testing..."):
            bt = run_backtest(history, engine)
        st.write(
            f"We hid the last {bt['holdout_days']} days, forecast them from the earlier data, "
            f"then compared with what actually sold."
        )
        b1, b2, b3 = st.columns(3)
        b1.metric("Actually sold", f"{bt['actual_units']:,.0f}")
        b2.metric("Forecast said", f"{bt['forecast_units']:,.0f}")
        b3.metric("Total error", f"{bt['total_error_%']:+.1f}%")
        st.caption(
            f"Day-by-day error (WAPE): {bt['wape_%']:.0f}%. Single days are noisy for small shops; "
            "totals over weeks are what matter for ordering stock."
        )

    table = fc.rename(columns={"ds": "date", "yhat": "expected_units", "yhat_lower": "low", "yhat_upper": "high"})
    table["date"] = table["date"].dt.date
    st.download_button(
        "Download forecast (CSV)", table.round(2).to_csv(index=False).encode(),
        f"forecast_{choice.replace(' ', '_')}.csv", "text/csv",
    )

# ------------------------------------------------------ seasonal patterns ---
with tab_seasons:
    choice_s = st.selectbox("Product", product_options, key="season_product")
    hist_s = total_daily if choice_s == ALL else daily_series(sales, choice_s)

    left, right = st.columns(2)
    with left:
        st.subheader("Best days of the week")
        wd = weekday_pattern(hist_s)
        st.plotly_chart(bar_chart(wd["weekday"], wd["avg_units_per_day"], "Avg units per day"))
    with right:
        st.subheader("Busy months")
        mp = monthly_pattern(hist_s)
        st.plotly_chart(bar_chart(mp["month"], mp["avg_units_per_day"], "Avg units per day"))
        if len(hist_s) < 365:
            st.caption("Less than a year of data - some months are missing or based on one visit.")

    st.subheader("Festival rush")
    lift = holiday_lift(hist_s, holidays)
    if lift.empty:
        st.info("No festivals fall fully inside your data yet.")
    else:
        top = lift.iloc[0]
        st.markdown(
            f"**{top['festival']}** is the biggest rush: about **{top['lift_x']:.1f}×** normal daily sales "
            "in its shopping window, compared with the weeks around it."
        )
        st.plotly_chart(bar_chart(lift["festival"], lift["lift_x"], "× normal sales", 300))
        st.dataframe(lift, hide_index=True)
    st.caption("Festival dates come from data/bd_holidays.csv - add your own events there.")

# ------------------------------------------------------ inventory planner ---
with tab_stock:
    st.subheader("How much should I order?")
    first_day = total_daily["ds"].max() + pd.Timedelta(days=1)
    st.write("Enter what you have in stock now. SalesCast uses the forecast to say when each product runs out "
             "and how much to order.")
    st.caption(f"Plans start on **{first_day:%d %b %Y}**, the day after the last sale in your data. "
               "Keep your sales file up to date so this matches today.")

    c1, c2, c3 = st.columns(3)
    lead_time = c1.number_input("Lead time (days until a new order arrives)", 1, 120, 14)
    coverage = c2.number_input("Order should last (days after arrival)", 7, 180, 30)
    service = c3.select_slider("Safety level", [0.80, 0.90, 0.95, 0.98, 0.99], value=0.95,
                               format_func=lambda v: f"{v:.0%} no stock-out")

    if lead_time + coverage > horizon:
        st.warning(f"Lead time + coverage ({lead_time + coverage} days) is longer than the forecast "
                   f"({horizon} days). Increase *Days to forecast* in the sidebar.")

    stock_input = st.data_editor(
        pd.DataFrame({"product": products, "current_stock": [0] * len(products)}),
        column_config={
            "product": st.column_config.TextColumn("Product", disabled=True),
            "current_stock": st.column_config.NumberColumn("Current stock (units)", min_value=0, step=1),
        },
        hide_index=True,
        key="stock_editor",
    )

    if st.button("Plan my orders", type="primary"):
        rows = []
        progress = st.progress(0.0, "Forecasting each product...")
        for i, product in enumerate(products):
            fc_p = run_forecast(daily_series(sales, product), horizon, engine)
            stock = float(stock_input.loc[stock_input["product"] == product, "current_stock"].fillna(0).iloc[0])
            rows.append(recommend(product, fc_p, stock, int(lead_time), int(coverage), service))
            progress.progress((i + 1) / len(products), f"Forecasting {product}...")
        progress.empty()

        plan = pd.DataFrame(rows)
        n_now = (plan["status"].str.contains("Order now")).sum()
        if n_now:
            st.error(f"{n_now} product(s) need ordering now.")
        else:
            st.success("No product needs ordering right now.")
        st.dataframe(plan, hide_index=True)
        st.download_button("Download order plan (CSV)", plan.to_csv(index=False).encode(),
                           "order_plan.csv", "text/csv")
        with st.expander("How is this calculated?"):
            st.markdown(
                """
                - **Demand during lead time** - forecast units sold while you wait for the order.
                - **Safety stock** - extra units to cover sales coming in higher than forecast,
                  sized by the forecast's uncertainty and your safety level.
                - **Reorder point** = demand during lead time + safety stock. At or below it → *Order now*.
                - **Recommended order** = forecast demand for lead time + coverage days, plus safety stock,
                  minus what you have.
                - **Order by** - latest day to order so stock arrives before you run out.
                """
            )

# ------------------------------------------------------------------ data ---
with tab_data:
    st.subheader("Data check")
    st.write(
        f"Read **{report['rows_in']:,}** rows, kept **{report['rows_kept']:,}**. "
        f"Dropped: {report['dropped_bad_date']} with unreadable dates, "
        f"{report['dropped_bad_quantity']} with unreadable quantities, "
        f"{report['dropped_zero_or_negative']} with zero/negative quantities (returns or cancellations)."
    )
    st.dataframe(sales, hide_index=True, height=400)
