"""
Equities section pipeline: S&P 500 fundamental valuation yields plotted
against the rest of the capital structure (risk-free Treasuries,
investment-grade credit, high-yield credit).

Pulls live data (Robert Shiller's public U.S. stock market dataset, FRED)
and writes:

  data/equities/figures/*.json  -- one Plotly figure per chart
  data/equities/meta.json       -- last-updated timestamp + headline stats

Run standalone: `python pipelines/equities.py` (from the repo root).

Measures used, and why these four and not others:
  - Trailing earnings yield (E/P) -- the simplest, most-cited valuation
    yield, built straight from Shiller's reported trailing EPS.
  - CAPE yield (1/CAPE) -- Shiller's cyclically-adjusted P/E (10-year real
    average EPS) inverted to a yield, so unlike raw E/P it isn't distorted
    by a single depressed or inflated earnings quarter (see the 2008-09
    dip in raw earnings yield below -- CAPE yield barely moves).
  - Dividend yield (D/P) -- the cash actually paid out, independent of
    accounting earnings entirely.
  - A true *forward* (analyst-consensus) earnings yield was deliberately
    left out: there is no free, continuously-updated public time series
    of S&P 500 forward EPS. S&P's own published estimate file requires a
    login (spglobal.com returns 403 to anonymous/scripted requests) and
    sell-side consensus (I/B/E/S-style) data is licensed. Faking a
    "forward" yield by extrapolating trailing EPS growth would be a model
    output dressed up as a data series, so it's excluded rather than
    approximated.

Capital-structure comparison uses:
  - DGS10 (FRED) -- 10-year Treasury constant-maturity yield, risk-free.
  - BAA (FRED) -- Moody's Seasoned Baa Corporate Bond Yield, investment-
    grade credit. Chosen over Aaa because Baa is the lowest investment-grade
    tier and the more common "corporate credit" benchmark in this kind of
    comparison; also has the longest history of any public IG credit series
    (back to 1919), matching Shiller's own data span well.
  - BAMLH0A0HYM2EY (FRED) -- ICE BofA US High Yield Index *effective yield*,
    junk-grade credit. NOTE THE SUFFIX: the much more commonly-charted
    series `BAMLH0A0HYM2` (no "EY") is the index's *option-adjusted
    spread*, not a yield -- it reads as a plausible-looking ~2-3% number
    but is actually the spread over Treasuries, and would have silently
    plotted junk credit as yielding *less* than investment-grade Baa debt,
    which is never true. Confirmed the "EY" series against BAMLC0A0CMEY
    (investment-grade effective yield, same naming convention) reading
    sensibly above Baa's own yield before shipping this.
    KNOWN LIMITATION: FRED's public feed for ICE-sourced series (both
    effective-yield series above included) is licensed to only expose a
    trailing ~3-year window to unauthenticated requests -- the full high-
    yield series historically ran back to 1996, but a `cosd=1996-01-01`
    request to FRED's own CSV endpoint is silently truncated to ~3 years
    back regardless. Included anyway for the recent-regime comparison it's
    still useful for; the line will simply appear only over the last few
    years rather than the full chart.
"""
import datetime as dt
import io
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import plotly.graph_objects as go
import plotly.io as pio
import requests

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "equities"
FIGURES_DIR = DATA_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FIGURES = {}
META = {}

# --- shared chart styling (same palette/chrome as pipelines/rates_macro.py
# and pipelines/currencies.py; duplicated rather than imported so sections
# stay fully independent) ---
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDLINE, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
RECESSION_FILL = "rgba(137,135,129,0.16)"

SERIES_COLOR = {
    "Trailing Earnings Yield": CAT[0],
    "CAPE Yield": CAT[6],
    "Dividend Yield": CAT[4],
    "10-Year Treasury (risk-free)": CAT[2],
    "Baa Corporate (investment-grade)": CAT[3],
    "ICE BofA High Yield (junk credit)": CAT[7],
}


def style_fig(fig, title, yaxis_title=None, height=460, legend=True):
    fig.update_layout(
        title=dict(text=title, font=dict(size=15, color=INK_PRIMARY)),
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
        font=dict(color=INK_SECONDARY, size=12),
        hovermode="x unified",
        height=height,
        margin=dict(l=60, r=30, t=60, b=40),
        showlegend=legend,
        legend=dict(orientation="h", yanchor="bottom", y=1.0, xanchor="left", x=0, font=dict(size=11)),
    )
    fig.update_xaxes(showgrid=False, showline=True, linecolor=BASELINE, ticks="outside", tickcolor=BASELINE, tickfont=dict(color=INK_MUTED))
    fig.update_yaxes(showgrid=True, gridcolor=GRIDLINE, gridwidth=1, zeroline=False, showline=False, tickfont=dict(color=INK_MUTED), title=dict(text=yaxis_title, font=dict(size=11, color=INK_MUTED)))
    return fig


def get_recession_bands(start_date, end_date):
    usrec = web.DataReader("USREC", "fred", start_date, end_date).squeeze()
    in_rec = usrec > 0
    rec_bands, rstart = [], None
    for date, val in in_rec.items():
        if val and rstart is None:
            rstart = date
        elif not val and rstart is not None:
            rec_bands.append((rstart, date))
            rstart = None
    if rstart is not None:
        rec_bands.append((rstart, in_rec.index[-1]))
    return rec_bands


def add_recession_bands(fig, rec_bands, xmin=None):
    xmin_ts = pd.Timestamp(xmin) if xmin else None
    for s, e in rec_bands:
        if xmin_ts is not None and e < xmin_ts:
            continue
        s = max(s, xmin_ts) if xmin_ts is not None else s
        fig.add_vrect(x0=s.strftime("%Y-%m-%d"), x1=e.strftime("%Y-%m-%d"), fillcolor=RECESSION_FILL, line_width=0, layer="below")
    return fig


def get_shiller_data():
    """Robert Shiller's public U.S. stock market dataset (price, dividends,
    trailing EPS, CAPE ratio -- monthly, back to 1871).

    The download link's own path includes a random-looking GUID that
    changes whenever Shiller's team re-uploads the file, so the URL is
    scraped off shillerdata.com's own "Download" button each run rather
    than hardcoded -- a hardcoded snapshot of that URL was confirmed stale
    (served a copy frozen as of Sept 2024) while the live button link
    served data current to this month.
    """
    home = requests.get("https://shillerdata.com/", timeout=30)
    home.raise_for_status()
    m = re.search(r'href="(//img1\.wsimg\.com/[^"]*ie_data\.xls[^"]*)"', home.text)
    if not m:
        raise RuntimeError("Couldn't find the ie_data.xls download link on shillerdata.com -- page layout may have changed.")
    xls_url = "https:" + m.group(1)

    resp = requests.get(xls_url, timeout=30)
    resp.raise_for_status()
    df = pd.read_excel(io.BytesIO(resp.content), sheet_name="Data", header=7)

    # Trailing rows are footnotes ("Sept price is Sept 4th close...", source
    # notes, etc.) with a non-numeric Date -- drop them.
    date_num = pd.to_numeric(df["Date"], errors="coerce")
    df = df.loc[date_num.notna()].copy()
    date_num = date_num.loc[df.index]

    # BUG-PRONE QUIRK: Shiller's Date column encodes month as a fraction of
    # year written as e.g. 2025.09 for September but 2025.1 (not "2025.10")
    # for October -- as a float, 2025.1 == 2025.10 exactly (trailing zeros
    # aren't preserved), so this round-trip actually recovers the right
    # month either way. Confirmed against the raw file: 2025.1 sits between
    # 2025.09 and 2025.11 in row order, i.e. October.
    year = np.floor(date_num).astype(int)
    month = ((date_num - year) * 100).round().astype(int)
    df.index = pd.to_datetime({"year": year, "month": month, "day": 1})

    out = df.loc[:, ["P", "D", "E", "CAPE"]].astype(float)
    out.columns = ["price", "dividend", "eps", "cape"]
    return out


def main():
    print("Pulling Shiller S&P 500 dataset (price, dividends, trailing EPS, CAPE)...")
    shiller = get_shiller_data()

    start_date = "1919-01-01"  # matches BAA's own start -- the longest-history credit series used here
    end_date = dt.date.today().strftime("%Y-%m-%d")

    print("Pulling Treasury, investment-grade, and high-yield credit yields from FRED...")
    treasury_10y = web.DataReader("DGS10", "fred", start_date, end_date).squeeze().resample("MS").mean()
    baa_yield = web.DataReader("BAA", "fred", start_date, end_date).squeeze()
    # NB: "EY" suffix = effective yield, NOT the same as BAMLH0A0HYM2 (that
    # series is the option-adjusted spread) -- see the module docstring.
    # FRED's public feed also only serves ~3 years of history for this
    # ICE-sourced series regardless of the requested start date. Requesting
    # the full range anyway costs nothing and means this line automatically
    # gets longer if that restriction is ever lifted, with no code change
    # needed here.
    hy_yield = web.DataReader("BAMLH0A0HYM2EY", "fred", start_date, end_date).squeeze().resample("MS").mean()

    earnings_yield = (shiller["eps"] / shiller["price"] * 100).rename("Trailing Earnings Yield")
    cape_yield = (1.0 / shiller["cape"] * 100).rename("CAPE Yield")
    dividend_yield = (shiller["dividend"] / shiller["price"] * 100).rename("Dividend Yield")
    treasury_10y = treasury_10y.rename("10-Year Treasury (risk-free)")
    baa_yield = baa_yield.rename("Baa Corporate (investment-grade)")
    hy_yield = hy_yield.rename("ICE BofA High Yield (junk credit)")

    recession_bands = get_recession_bands(start_date, end_date)

    # =========================================================================
    # Single comparison chart: every yield on one %-axis, full available
    # history. Never dual-axis and never a hard-clipped y-range here --
    # earnings yield swings hard in a real recession (e.g. ~0.8% in mid-2009
    # as trailing GAAP earnings collapsed) and clipping would read as a data
    # gap rather than what actually happened.
    # =========================================================================
    fig = go.Figure()
    for series in [treasury_10y, baa_yield, hy_yield, dividend_yield, earnings_yield, cape_yield]:
        d = series.dropna()
        fig.add_trace(go.Scatter(x=d.index, y=d.values, name=series.name, line=dict(color=SERIES_COLOR[series.name], width=2)))
    add_recession_bands(fig, recession_bands)
    FIGURES["yield_comparison"] = style_fig(
        fig, "S&P 500 Fundamental Yields vs. the Rest of the Capital Structure", yaxis_title="%"
    )

    # =========================================================================
    # Save everything
    # =========================================================================
    print(f"Saving {len(FIGURES)} figures + meta to {DATA_DIR}...")
    for key, f in FIGURES.items():
        pio.write_json(f, FIGURES_DIR / f"{key}.json")

    latest_month = shiller.index.max()
    META.update(
        last_updated=dt.datetime.now(dt.timezone.utc).isoformat(),
        data_asof=latest_month.strftime("%Y-%m-%d"),
        earnings_yield=round(float(earnings_yield.dropna().iloc[-1]), 2),
        cape_yield=round(float(cape_yield.dropna().iloc[-1]), 2),
        dividend_yield=round(float(dividend_yield.dropna().iloc[-1]), 2),
        treasury_10y=round(float(treasury_10y.dropna().iloc[-1]), 2),
        baa_yield=round(float(baa_yield.dropna().iloc[-1]), 2),
        hy_yield=round(float(hy_yield.dropna().iloc[-1]), 2) if hy_yield.dropna().size else None,
        earnings_yield_vs_baa=round(float(earnings_yield.dropna().iloc[-1] - baa_yield.dropna().iloc[-1]), 2),
    )
    with open(DATA_DIR / "meta.json", "w") as fh:
        json.dump(META, fh, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
