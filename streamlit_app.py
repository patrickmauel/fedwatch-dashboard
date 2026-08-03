"""
FedWatch Streamlit dashboard.

Reads only the files pipeline.py writes to data/ -- never touches the
network itself. Run the pipeline first (`python pipeline.py`), then:
    streamlit run streamlit_app.py
"""
import datetime as dt
import json
from pathlib import Path

import pandas as pd
import plotly.io as pio
import streamlit as st

DATA_DIR = Path(__file__).parent / "data"
FIGURES_DIR = DATA_DIR / "figures"

st.set_page_config(page_title="FedWatch Dashboard", page_icon="\U0001F4C8", layout="wide")


@st.cache_data(ttl=3600)
def load_meta():
    path = DATA_DIR / "meta.json"
    if not path.exists():
        return None
    return json.loads(path.read_text())


@st.cache_data(ttl=3600)
def load_grid():
    path = DATA_DIR / "fomc_grid.csv"
    if not path.exists():
        return None
    return pd.read_csv(path)


@st.cache_data(ttl=3600)
def load_figure(key):
    path = FIGURES_DIR / f"{key}.json"
    if not path.exists():
        return None
    return pio.read_json(path)


def chart(key, container=st):
    fig = load_figure(key)
    if fig is None:
        container.warning(f"Missing chart: {key}. Run `python pipeline.py` to generate it.")
        return
    container.plotly_chart(fig, use_container_width=True)


meta = load_meta()

st.title("FedWatch Dashboard")

if meta is None:
    st.error(
        "No data found yet. Run `python pipeline.py` locally (or wait for the "
        "first scheduled GitHub Actions run) to populate the `data/` folder."
    )
    st.stop()

updated = dt.datetime.fromisoformat(meta["last_updated"])
age = dt.datetime.now(dt.timezone.utc) - updated
staleness_note = ""
if age > dt.timedelta(hours=36):
    staleness_note = f" :orange[(**{age.days}d {age.seconds // 3600}h** old -- check the daily refresh job)]"
st.caption(f"Data as of **{updated:%Y-%m-%d %H:%M UTC}**{staleness_note}")

c1, c2, c3, c4, c5 = st.columns(5)
c1.metric("Fed Funds (effective)", f"{meta['fed_funds_effective']:.2f}%")
c2.metric("Unemployment Rate", f"{meta['unemployment_rate']:.2f}%")
c3.metric("Core PCE (YoY)", f"{meta['core_pce_yoy']:.2f}%")
c4.metric("Core CPI (YoY)", f"{meta['core_cpi_yoy']:.2f}%")
c5.metric("Outsized releases", meta["outsized_release_count"], help="FOMC data-grid rows with |z-score| >= 2 over the last release")

st.caption(
    f"Model bridges from LW's last complete quarter (**{meta['fcst_start_lw']}**) to the last quarter "
    f"with full actual data (**{meta['last_actual_quarter']}**), then Monte Carlo-simulates forward to "
    f"**{meta['fcst_end']}**. Treasury curve as of **{meta['treasury_asof_date']}**."
)

tab_overview, tab_grid, tab_credit, tab_labor = st.tabs(
    ["Overview", "FOMC Data Grid", "Credit Markets", "Labor Market"]
)

with tab_overview:
    st.subheader("GDP, inflation, fed funds & term structure")
    chart("main_dashboard")

with tab_grid:
    st.subheader("FOMC-relevant data releases")
    st.caption(
        "Z-score = how many standard deviations the latest period-over-period change is from that "
        "series' own trailing 10-year distribution of such changes. |z| >= 2 flagged **outsized**, "
        "1.5-2 **elevated**."
    )
    chart("fomc_grid")
    grid_df = load_grid()
    if grid_df is not None:
        with st.expander("Raw table"):
            st.dataframe(grid_df, use_container_width=True, hide_index=True)

with tab_credit:
    st.subheader("Household credit")
    chart("household_debt")
    col1, col2 = st.columns(2)
    chart("household_dsr", col1)
    chart("household_sr", col2)

    st.subheader("Corporate & economy-wide leverage")
    chart("total_leverage")
    chart("corporate_debt")
    chart("corporate_dsr")

    st.subheader("Government credit")
    chart("government_debt")
    chart("gov_interest_gdp")
    chart("gov_int_deficit")
    chart("gov_int_vs_growth")

with tab_labor:
    st.subheader("Labor market")
    col1, col2 = st.columns(2)
    chart("labor_unemployment", col1)
    chart("labor_participation", col2)
    chart("labor_payrolls")
    chart("labor_claims")
    chart("labor_demand")

st.divider()
st.caption(
    "Data: FRED, NY Fed (Laubach-Williams r*, Nowcast, MCT inflation), Cleveland Fed, U.S. Treasury. "
    "Refreshed daily by GitHub Actions. Not investment advice."
)
