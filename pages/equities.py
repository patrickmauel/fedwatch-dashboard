"""
Equities page.

Reads only the files pipelines/equities.py writes to data/equities/ --
never touches the network itself. Run the pipeline first
(`python pipelines/equities.py` from the repo root).
"""
import datetime as dt
import json
from pathlib import Path

import plotly.io as pio
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "equities"
FIGURES_DIR = DATA_DIR / "figures"


@st.cache_data(ttl=3600)
def load_meta():
    path = DATA_DIR / "meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data(ttl=3600)
def load_figure(key):
    path = FIGURES_DIR / f"{key}.json"
    if not path.exists():
        return None
    return pio.read_json(path)


def chart(key, container=st):
    fig = load_figure(key)
    if fig is None:
        container.warning(f"Missing chart: {key}. Run `python pipelines/equities.py` to generate it.")
        return
    container.plotly_chart(fig, use_container_width=True)


meta = load_meta()

st.title("Equities")

REQUIRED_META_KEYS = {
    "last_updated", "data_asof", "shiller_asof", "earnings_yield", "cape_yield", "dividend_yield",
    "treasury_10y", "baa_yield", "earnings_yield_vs_baa", "implied_earnings_growth",
    "implied_fwd_earnings_yield", "curve_bias_extension", "curve_bias_trend", "curve_bias_asof",
}
if meta is None or not REQUIRED_META_KEYS.issubset(meta):
    # meta.json is written in one atomic json.dump() at the very end of a
    # successful pipeline run, so a real file should never be partial --
    # missing keys here means something upstream is stale (e.g. a cache or
    # deploy-in-progress race) rather than a data problem worth rendering
    # around piecemeal. Clearing the cache resolves it if so.
    load_meta.clear()
    st.error(
        "Data file is missing or incomplete. This usually clears itself -- try refreshing the "
        "page. If it persists, run `python pipelines/equities.py` locally (or wait for the next "
        "scheduled GitHub Actions run) to (re)populate `data/equities/`, or reboot the app from "
        "Streamlit Cloud's 'Manage app' menu to clear a stale cache."
    )
    st.stop()

updated = dt.datetime.fromisoformat(meta["last_updated"])
age = dt.datetime.now(dt.timezone.utc) - updated
staleness_note = ""
if age > dt.timedelta(hours=36):
    staleness_note = f" :orange[(**{age.days}d {age.seconds // 3600}h** old -- check the daily refresh job)]"
st.caption(
    f"Price as of **{meta['data_asof']}**, Shiller fundamentals (dividends/EPS/CAPE) complete through "
    f"**{meta['shiller_asof']}** -- updated **{updated:%Y-%m-%d %H:%M UTC}**{staleness_note}"
)

c1, c2, c3, c4, c5, c6, c7 = st.columns(7)
c1.metric("Trailing Earnings Yield", f"{meta['earnings_yield']:.2f}%", help="Trailing 12-month S&P 500 EPS / price = 1 / trailing P/E")
c2.metric("CAPE Yield", f"{meta['cape_yield']:.2f}%", help="1 / Shiller CAPE (price over 10-year average real EPS)")
c3.metric("Dividend Yield", f"{meta['dividend_yield']:.2f}%")
c4.metric("10-Year Treasury", f"{meta['treasury_10y']:.2f}%", help="Risk-free")
c5.metric(
    "Earnings Yield vs. Baa Credit", f"{meta['earnings_yield_vs_baa']:+.2f} pp",
    help="Trailing earnings yield minus Moody's Baa corporate bond yield. Negative means equities currently "
         "compensate less than investment-grade credit -- one read of \"expensive.\"",
)
c6.metric(
    "Implied Earnings Growth", f"{meta['implied_earnings_growth']:.1f}%",
    help="The constant annual earnings growth rate, starting from latest TTM earnings, that makes the "
         "*average* trailing-earnings-yield path over the next 20 years equal today's Baa yield -- i.e. "
         "the growth priced in for stocks to merely break even against investment-grade credit. A model "
         "output, not a forecast; see the pipeline's docstring.",
)
c7.metric(
    "Implied Fwd Earnings Yield", f"{meta['implied_fwd_earnings_yield']:.2f}%",
    help="Trailing earnings yield compounded one year at the Implied Earnings Growth rate -- year 1 of "
         "that same modeled path, expressed as a yield rather than a growth rate.",
)

st.subheader("Fundamental equity yields vs. the capital structure")
st.caption(
    "Three ways to price the S&P 500 as a yield -- trailing earnings yield (E/P), CAPE yield "
    "(cyclically-adjusted, smooths out any one quarter's depressed or inflated earnings), and "
    "dividend yield -- next to the risk-free 10-year Treasury and Baa investment-grade credit. "
    "The equity lines update daily: EPS/dividends/CAPE's earnings base hold at their last-reported "
    "value between Shiller's monthly updates while price moves live, same as how a real-time P/E or "
    "dividend yield is computed anywhere else. A true *forward* (analyst-consensus) earnings yield "
    "isn't shown -- checked S&P's own estimate file (403s scripted requests), Yardeni Research "
    "(paid Refinitiv feed behind the free charts), and multpl.com (confirmed trailing-only); none "
    "of them expose a free, continuously-updated series. See the pipeline's docstring for the "
    "full rundown, including why a high-yield credit line was tried and then dropped. Also shown: "
    "**Implied Forward Earnings Yield** (dashed) -- today's trailing yield compounded one year at "
    "the growth rate that would make the *average* earnings yield over the next 20 years equal "
    "today's Baa yield. One more **modeled** line, the growth rate itself (Implied Earnings "
    "Growth), is in the legend but hidden by default (click to show) -- it swings far wider than "
    "every other line here (>20pp in both directions during real crises) and would flatten them "
    "if shown at the same time."
)
chart("yield_comparison")

st.divider()
st.subheader("Curve bias")
st.caption(
    "How stretched the S&P 500's own price action currently is, in two dimensions, both scaled by "
    "the trailing 21-trading-day (~1 month) volatility of daily point moves so they're directly "
    "comparable: **y** is how far price sits above/below its own 21-day average right now (a "
    "short-term extension read); **x** is the trailing-21-day return scaled by that same "
    f"volatility's square-root-of-time-scaled 21-day expected move (a medium-term trend-strength "
    "read). The muted cloud is the full available history "
    f"(FRED's S&P 500 daily series is capped to a trailing ~10 years by licensing -- not this "
    "page's choice); the highlighted path is the last 60 sessions, so the *direction* of travel "
    "through this space is visible, not just today's snapshot. Top-right / bottom-left = price "
    "extended in the same direction it has been trending (trend-following regime); top-left / "
    "bottom-right = price extended *against* its own trailing trend (a possible mean-reversion "
    "setup, or an early trend reversal -- this chart alone doesn't say which)."
)
chart("curve_bias")

col_ext, col_trend = st.columns(2)
col_ext.caption("The y-axis on its own, over time: short-term extension above/below the 21-day average.")
chart("curve_bias_extension", container=col_ext)
col_trend.caption("The x-axis on its own, over time: medium-term (21-day) trend strength.")
chart("curve_bias_trend", container=col_trend)

st.divider()
st.caption(
    "Data: Robert Shiller's public U.S. stock market dataset (price, dividends, trailing EPS, "
    "CAPE) extended with FRED's daily S&P 500 close, FRED (10-year Treasury yield, Moody's Baa "
    "corporate yield). Refreshed daily by GitHub Actions. Not investment advice."
)
