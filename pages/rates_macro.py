"""
Rates & Macro page.

Reads only the files pipelines/rates_macro.py writes to data/rates_macro/
-- OUR code never touches the network itself. Run the pipeline first
(`python pipelines/rates_macro.py` from the repo root), then open the
dashboard via streamlit_app.py (this page is one section of it, not a
standalone entry point -- no st.set_page_config here, that lives in
streamlit_app.py since it can only be called once per app).

One exception to "never touches the network": the NY Fed Nowcast tab
embeds newyorkfed.org's own live Nowcast tool in an iframe -- that's the
VIEWER's browser loading it directly (nothing in this pipeline/page fetches
or caches it), so it's exempt from the "pipeline writes data/, page only
reads data/" pattern every other tab follows. See that tab's own comment
below for why an iframe was used instead of pulling the data in.
"""
import datetime as dt
import json
from pathlib import Path

import pandas as pd
import plotly.io as pio
import streamlit as st
import streamlit.components.v1 as components

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "rates_macro"
FIGURES_DIR = DATA_DIR / "figures"


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
        container.warning(f"Missing chart: {key}. Run `python pipelines/rates_macro.py` to generate it.")
        return
    container.plotly_chart(fig, use_container_width=True)


meta = load_meta()

st.title("Rates & Macro")

REQUIRED_META_KEYS = {
    "last_updated", "fcst_start_lw", "last_actual_quarter", "fcst_end", "treasury_asof_date",
    "fed_funds_effective", "unemployment_rate", "core_pce_yoy", "core_cpi_yoy", "outsized_release_count",
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
        "page. If it persists, run `python pipelines/rates_macro.py` locally (or wait for the "
        "next scheduled GitHub Actions run) to (re)populate `data/rates_macro/`, or reboot the "
        "app from Streamlit Cloud's 'Manage app' menu to clear a stale cache."
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

tab_overview, tab_grid, tab_credit, tab_labor, tab_nowcast = st.tabs(
    ["Overview", "FOMC Data Grid", "Credit Markets", "Labor Market", "NY Fed Nowcast"]
)

with tab_overview:
    st.subheader("GDP, inflation, fed funds & term structure")
    # MOBILE REWRITE (2026-09-21): this used to be one make_subplots(2, 2)
    # figure ("main_dashboard") -- see the matching MOBILE REWRITE comment
    # in pipelines/rates_macro.py's main() for why that broke on a phone.
    # Now 4 independent charts, 2 per row via st.columns(2) -- the same
    # pattern already used everywhere else on this page (labor, credit),
    # which Streamlit stacks to a single column automatically below phone
    # width, unlike one shared subplot grid.
    col1, col2 = st.columns(2)
    chart("rgdp_realizations", col1)
    chart("pce_realizations", col2)
    col3, col4 = st.columns(2)
    chart("gdp_modeled", col3)
    chart("fed_funds_rstar", col4)

with tab_grid:
    st.subheader("FOMC-relevant data releases")
    st.caption(
        "Z-score = how many standard deviations the latest period-over-period change is from that "
        "series' own trailing 10-year distribution of such changes. |z| >= 2 flagged **outsized**, "
        "1.5-2 **elevated**."
    )
    grid_df = load_grid()
    if grid_df is None:
        st.warning("Missing FOMC grid data. Run `python pipelines/rates_macro.py` to generate it.")
    else:
        # MOBILE REWRITE (2026-09-21): this used to be a go.Table PLOTLY
        # FIGURE (8 fixed-pixel columns, illegible once squeezed to phone
        # width -- see the matching comment in pipelines/rates_macro.py's
        # FOMC grid section). st.dataframe is a real HTML/React grid: it
        # horizontal-scrolls with an actual finger swipe on a phone instead
        # of shrinking text to fit, and the pandas Styler below reproduces
        # the same category-tint + z-score heat coloring the old Table had,
        # just as CSS background-color instead of baked-in pixel fill.
        def zscore_style(z):
            if pd.isna(z):
                return ""
            t = max(-1.0, min(1.0, z / 3.0))
            if t >= 0:
                r, g, b = 255, int(255 - t * 155), int(255 - t * 155)
            else:
                t = -t
                r, g, b = int(255 - t * 155), int(255 - t * 155), 255
            return f"background-color: rgb({r},{g},{b})"

        CATEGORY_BG = {"Growth": "#eef4fb", "Labor": "#eefaf0", "Inflation": "#fdf3ec", "Rates": "#f5f0fb"}

        display_df = grid_df.drop(columns=["SeriesID"]).copy()
        display_df["Flag"] = display_df["Zscore"].apply(
            lambda z: "outsized" if pd.notna(z) and abs(z) >= 2 else ("elevated" if pd.notna(z) and abs(z) >= 1.5 else "")
        )
        display_df = display_df.rename(columns={
            "LatestRelease": "Latest Release", "PriorValue": "Prior", "LatestValue": "Latest", "Zscore": "Z-score (10y)",
        })

        def row_style(row):
            base = f"background-color: {CATEGORY_BG.get(row['Category'], 'white')}"
            styles = [base] * len(row)
            styles[row.index.get_loc("Z-score (10y)")] = zscore_style(row["Z-score (10y)"]) or base
            return styles

        styled = display_df.style.apply(row_style, axis=1).format(
            {"Prior": "{:.2f}", "Latest": "{:.2f}", "Change": "{:+.2f}", "Z-score (10y)": "{:+.2f}"}, na_rep="n/a"
        )
        st.dataframe(styled, use_container_width=True, hide_index=True)

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
    st.caption(
        "Data runs back to 2007, but charts open on 2022+ by default -- April 2020's ~20M "
        "one-month payroll drop and claims' ~6.9M one-week spike are so far outside every other "
        "value that including them by default flattens the rest of each chart. Click the "
        "autoscale icon in a chart's toolbar (or drag to zoom out) to see the full history."
    )
    col1, col2 = st.columns(2)
    chart("labor_unemployment", col1)
    chart("labor_participation", col2)
    chart("labor_payrolls")
    chart("labor_claims")
    chart("labor_demand")

with tab_nowcast:
    NOWCAST_URL = "https://www.newyorkfed.org/research/policy/nowcast/#/nowcast"
    st.subheader("NY Fed Staff Nowcast")
    st.caption(
        "The New York Fed's own real-time GDP nowcast, including its **Data Flow** breakdown of "
        "how each quarterly GDP estimate has moved as new data releases came in -- embedded live "
        "from newyorkfed.org, not reproduced here. This is the tool itself (their JS renders it), "
        "so it always reflects whichever quarter(s) the Fed currently tracks -- nothing to update "
        "on our end as quarters roll over. Considered pulling the underlying data in instead (the "
        "house pattern every other tab uses), but couldn't find a stable, documented public data "
        "endpoint behind their interactive -- it's loaded by a minified in-page app, not a plain "
        "API or a linked CSV/XLSX. Confirmed newyorkfed.org sends no `X-Frame-Options` or CSP "
        "`frame-ancestors` header, so embedding is actually permitted (many .gov/.org sites block "
        "this outright). If it renders blank for you, your browser/extensions may still be "
        f"blocking the frame -- [open it directly]({NOWCAST_URL}) instead."
    )
    components.iframe(NOWCAST_URL, height=900, scrolling=True)

st.divider()
st.caption(
    "Data: FRED, NY Fed (Laubach-Williams r*, Nowcast, MCT inflation), Cleveland Fed, U.S. Treasury. "
    "Refreshed daily by GitHub Actions. Not investment advice."
)
