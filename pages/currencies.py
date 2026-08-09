"""
Currencies page.

Reads only the files pipelines/currencies.py writes to data/currencies/ --
never touches the network itself. Run the pipeline first
(`python pipelines/currencies.py` from the repo root).
"""
import datetime as dt
import json
from pathlib import Path

import plotly.io as pio
import streamlit as st

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "currencies"
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
        container.warning(f"Missing chart: {key}. Run `python pipelines/currencies.py` to generate it.")
        return
    container.plotly_chart(fig, use_container_width=True)


meta = load_meta()

st.title("Currencies")

if meta is None:
    st.error(
        "No data found yet. Run `python pipelines/currencies.py` locally (or wait for the "
        "first scheduled GitHub Actions run) to populate the `data/currencies/` folder."
    )
    st.stop()

updated = dt.datetime.fromisoformat(meta["last_updated"])
age = dt.datetime.now(dt.timezone.utc) - updated
staleness_note = ""
if age > dt.timedelta(hours=36):
    staleness_note = f" :orange[(**{age.days}d {age.seconds // 3600}h** old -- check the daily refresh job)]"
st.caption(f"Data as of **{updated:%Y-%m-%d %H:%M UTC}**{staleness_note}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("US Policy Rate", f"{meta['us_policy_rate']:.2f}%")
c2.metric("Euro Area Policy Rate", f"{meta['ea_policy_rate']:.2f}%")
c3.metric("Canada Policy Rate", f"{meta['ca_policy_rate']:.2f}%")
c4.metric("EUR/USD", f"{meta['eurusd_spot']:.4f}")
c5.metric("USD/CAD", f"{1/meta['cadusd_spot']:.4f}", help=f"= {meta['cadusd_spot']:.4f} USD per CAD")

st.caption(
    f"Holston-Laubach-Williams natural-rate model (US / Euro Area / Canada), simulated from the "
    f"NY Fed's last published quarter (**{meta['hlw_anchor_quarter']}**) out to **{meta['fcst_end']}**. "
    f"FX forecasts derived from the simulated interest-rate differential vs. the US. Model's own "
    f"published quarter typically lags the current one by a quarter, same as the domestic rates "
    f"model -- see the note in pipelines/currencies.py."
)

tab_overview, tab_us, tab_ea, tab_ca, tab_trade = st.tabs(
    ["Overview & FX", "United States", "Euro Area", "Canada", "Trade Balance"]
)

with tab_overview:
    st.subheader("Policy rates: US vs. Euro Area vs. Canada")
    chart("rates_comparison")
    st.subheader("FX forecasts")
    col1, col2 = st.columns(2)
    chart("fx_eurusd", col1)
    chart("fx_cadusd", col2)

with tab_us:
    chart("us_gdp")
    col1, col2 = st.columns(2)
    chart("us_inflation", col1)
    chart("us_rate", col2)

with tab_ea:
    chart("euro_area_gdp")
    col1, col2 = st.columns(2)
    chart("euro_area_inflation", col1)
    chart("euro_area_rate", col2)

with tab_ca:
    chart("canada_gdp")
    col1, col2 = st.columns(2)
    chart("canada_inflation", col1)
    chart("canada_rate", col2)

with tab_trade:
    st.subheader("Net exports (% of GDP)")
    chart("trade_balance")

st.divider()
st.caption(
    "Data: NY Fed (Holston-Laubach-Williams estimates), FRED (FX rates, US trade/GDP), Eurostat "
    "(EU trade). Refreshed daily by GitHub Actions. Not investment advice."
)
