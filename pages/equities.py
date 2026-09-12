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
    "last_updated", "data_asof", "earnings_yield", "cape_yield", "dividend_yield",
    "treasury_10y", "baa_yield", "earnings_yield_vs_baa",
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
st.caption(f"Data as of **{meta['data_asof']}** (Shiller dataset), updated **{updated:%Y-%m-%d %H:%M UTC}**{staleness_note}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Trailing Earnings Yield", f"{meta['earnings_yield']:.2f}%", help="Trailing 12-month S&P 500 EPS / price = 1 / trailing P/E")
c2.metric("CAPE Yield", f"{meta['cape_yield']:.2f}%", help="1 / Shiller CAPE (price over 10-year average real EPS)")
c3.metric("Dividend Yield", f"{meta['dividend_yield']:.2f}%")
c4.metric("10-Year Treasury", f"{meta['treasury_10y']:.2f}%", help="Risk-free")
c5.metric(
    "Earnings Yield vs. Baa Credit", f"{meta['earnings_yield_vs_baa']:+.2f} pp",
    help="Trailing earnings yield minus Moody's Baa corporate bond yield. Negative means equities currently "
         "compensate less than investment-grade credit -- one read of \"expensive.\"",
)

st.subheader("Fundamental equity yields vs. the capital structure")
st.caption(
    "Three ways to price the S&P 500 as a yield -- trailing earnings yield (E/P), CAPE yield "
    "(cyclically-adjusted, smooths out any one quarter's depressed or inflated earnings), and "
    "dividend yield -- next to the risk-free rate and two tiers of corporate credit. Reading down "
    "the legend at any point in time is reading up the capital structure by risk: Treasuries, then "
    "investment-grade (Baa) credit, then high-yield credit, then equities. A true *forward* "
    "(analyst-consensus) earnings yield isn't shown -- there's no free, continuously-updated public "
    "series for it (S&P's own estimate file requires a login); see the pipeline's docstring."
)
chart("yield_comparison")
st.caption(
    "High-yield credit (ICE BofA effective yield) only appears over roughly the last 3 years -- "
    "FRED's public feed for that series is licensed to expose just a trailing window, not its full "
    "history back to 1996."
)

st.divider()
st.caption(
    "Data: Robert Shiller's public U.S. stock market dataset (price, dividends, trailing EPS, "
    "CAPE), FRED (Treasury yield, Moody's Baa corporate yield, ICE BofA high-yield effective "
    "yield). Refreshed daily by GitHub Actions. Not investment advice."
)
