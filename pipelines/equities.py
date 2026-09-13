"""
Equities section pipeline: S&P 500 fundamental valuation yields plotted
against the rest of the capital structure (risk-free Treasuries,
investment-grade credit).

Pulls live data (Robert Shiller's public U.S. stock market dataset, FRED)
and writes:

  data/equities/figures/*.json  -- one Plotly figure per chart
  data/equities/meta.json       -- last-updated timestamp + headline stats

Run standalone: `python pipelines/equities.py` (from the repo root).

Measures used, and why these three and not others:
  - Trailing earnings yield (E/P) -- the simplest, most-cited valuation
    yield, built straight from Shiller's reported trailing EPS.
  - CAPE yield (1/CAPE) -- Shiller's cyclically-adjusted P/E (10-year real
    average EPS) inverted to a yield, so unlike raw E/P it isn't distorted
    by a single depressed or inflated earnings quarter (see the 2008-09
    dip in raw earnings yield below -- CAPE yield barely moves).
  - Dividend yield (D/P) -- the cash actually paid out, independent of
    accounting earnings entirely.
  - A true *forward* (analyst-consensus) earnings yield is deliberately
    left out -- checked four candidate sources, all dead ends for a free,
    scriptable, continuously-updated time series:
      1. S&P Dow Jones Indices' own published EPS-estimate file
         (spglobal.com) -- returns 403 to any scripted request (tried a
         browser User-Agent + Referer too), Akamai/Cloudflare-level bot
         protection, not a simple auth wall.
      2. Yardeni Research -- publishes plenty of free forward-P/E *charts*,
         but they're rendered from a paid Refinitiv/LSEG Datastream feed
         (product.datastream.com links, subscription-gated); no underlying
         CSV/XLS.
      3. multpl.com -- confirmed trailing-only ("Price to earnings ratio,
         based on trailing twelve month 'as reported' earnings"); no
         forward-P/E page exists on the site.
      4. Sell-side consensus (I/B/E/S-style) data is licensed outright.
    Extrapolating trailing EPS growth to fake a "forward" number would be a
    model output dressed up as a data series, so it's excluded rather than
    approximated.

Daily resolution: Shiller updates ie_data.xls once a month, days after
month-end, so a pure Shiller-only chart would show earnings/CAPE/dividend
yield frozen for 3-5 weeks at a time -- out of step with how these
actually move (with the market, daily) between fundamentals updates and
visually inconsistent with the Treasury line's genuine daily resolution.
get_daily_tail() below extends the monthly series past Shiller's last
published month using FRED's daily S&P 500 close, holding EPS/dividend/the
CAPE ratio's implied earnings base flat at their last known value -- this
is exactly how "today's P/E" or "today's dividend yield" is computed by
any real-time source: the numerator doesn't move until new fundamentals
are actually reported, only price does.

Capital-structure comparison uses:
  - DGS10 (FRED) -- 10-year Treasury constant-maturity yield, risk-free,
    genuinely daily, plotted at native resolution (not resampled).
  - BAA + DBAA (FRED) -- Moody's Seasoned Baa Corporate Bond Yield,
    investment-grade credit. Chosen over Aaa because Baa is the lowest
    investment-grade tier and the more common "corporate credit" benchmark
    in this kind of comparison. Spliced from two FRED series rather than
    one: monthly BAA has the longest history of any public IG credit
    series (back to 1919, matching Shiller's own data span), but FRED also
    carries a genuinely daily version, DBAA, starting 1986-01-02 -- an
    earlier pass at this pipeline claimed no free daily version existed,
    which was wrong (just hadn't checked for a "D"-prefixed daily variant).
    get_baa_yield() below uses monthly BAA before DBAA's start and DBAA
    (its native daily resolution, not resampled) from 1986 on, so the
    modern era of the chart is genuinely daily instead of monthly-stepped,
    without sacrificing the pre-1986 history.

A high-yield/junk-credit line (ICE BofA effective yield, FRED
BAMLH0A0HYM2EY) was tried and then dropped. It only cluttered the chart for
what it was worth: FRED's own series notes confirm that, starting April
2026, ICE's license caps *every* ICE-sourced series on FRED (OAS or
effective-yield, IG or HY alike) to a trailing ~3-year window -- "go to the
source" for the fuller history FRED used to carry back to 1996. Checked
whether a different series ID or an explicit `cosd=1996-01-01` sidesteps
it -- neither does; the restriction is at the licensing level, not
per-request. No free public source has a comparable continuous high-yield
series, so a 3-year sliver of one more line wasn't worth the extra legend
entry.

Three derived (modeled, not observed) series, added after the above:
  - Implied Earnings Growth -- back-solves the constant annual earnings
    growth rate that would make the *average* trailing-earnings-yield path
    over the next 20 years equal today's Baa yield. See
    implied_earnings_growth()'s docstring for the exact assumption and why
    it's plotted hidden-by-default on the yield_comparison chart rather
    than folded into meta only.
  - Implied Forward Earnings Yield -- today's trailing yield compounded one
    year at that same solved growth rate (year 1 of the path whose 20-year
    average was set equal to Baa). Not a real forward yield in the sense
    of an analyst-consensus estimate (see the "true forward" discussion
    above) -- it's one year of the same model output, expressed in yield
    terms instead of a growth rate.
  - Curve Bias -- a second, unrelated chart (own FIGURES key
    "curve_bias"): S&P 500 price's short-term extension from its 21-day
    average vs. its medium-term (21-day) trend strength, both scaled by
    trailing daily volatility. See get_curve_bias_data()'s docstring.
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
from scipy.optimize import brentq

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
    "10-Year Treasury": CAT[2],
    "Baa Corporate Credit": CAT[3],
    "Implied Earnings Growth": CAT[7],
    "Implied Fwd Yield": CAT[1],
}


def style_fig(fig, title, yaxis_title=None, height=440, legend=True, hovermode="x unified"):
    fig.update_layout(
        # BUG FIX: caused a site-wide outage the day this section shipped.
        # go.Figure() defaults layout.template to plotly's full built-in
        # theme object, and pio.write_json serializes that whole object
        # (every trace type's default properties, not just the ones used
        # here) into the JSON unless told otherwise. Streamlit Cloud's next
        # dependency rebuild (triggered by this section's own requirements.txt
        # change) installed plotly 7.0, which renamed 'scattermapbox' to
        # 'scattermap' in that same template -- so every chart's JSON,
        # written under plotly 6.9, failed to parse under plotly>=7 with
        # "Invalid property specified ... 'scattermapbox'", taking down every
        # section's charts, not just this one. template=None means nothing
        # here depends on the default theme anyway -- every visual property
        # is already set explicitly, here and in update_xaxes/update_yaxes.
        template=None,
        title=dict(text=title, font=dict(size=15, color=INK_PRIMARY)),
        plot_bgcolor=SURFACE, paper_bgcolor=SURFACE,
        font=dict(color=INK_SECONDARY, size=12),
        hovermode=hovermode,
        height=height,
        margin=dict(l=60, r=30, t=70, b=40),
        showlegend=legend,
        # Short labels (below) keep these 5 entries on one row at normal
        # container widths -- a 6-entry version with longer names (see the
        # HY-series removal note in the module docstring) used to wrap to 2
        # lines and collide with the title above; dropping HY and shortening
        # labels fixed that at the source instead of just carving out more
        # margin for a 2-line legend.
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


def add_daily_tail(shiller):
    """Extend Shiller's monthly series with a daily tail using FRED's daily
    S&P 500 close (series SP500, available back to 2016 -- comfortably
    covers however many weeks/months it's been since Shiller's last fully
    reported month).

    BUG CAUGHT WHILE BUILDING THIS: Shiller's price/CAPE columns are
    sometimes populated 2-3 months ahead of dividend/EPS, which need actual
    reported figures and lag more -- confirmed against the live file, which
    had real price+CAPE for the trailing 3 months but NaN dividend/EPS for
    all three. Anchoring on the literal last row (`shiller.iloc[-1]`) would
    have propagated those NaNs through every "held flat" value below,
    silently blanking the earnings-yield and dividend-yield lines for
    however many months Shiller's own reporting lag runs. Anchoring on the
    last row where dividend/EPS/CAPE are ALL present, and rebuilding
    everything after that from daily price, avoids the gap entirely -- and
    is a strictly better reconstruction of those partial months than
    Shiller's own placeholder price-only rows anyway.

    Holds trailing EPS and dividend flat at the anchor's last known value,
    and scales the anchor's CAPE ratio by the live/anchor price ratio
    (equivalent to holding CAPE's 10-year-average-real-earnings denominator
    flat) -- see the module docstring for why this is the standard way to
    compute a "live" P/E-style figure, not an approximation dressed up as
    real data.
    """
    complete = shiller.dropna(subset=["dividend", "eps", "cape"])
    anchor_date, anchor = complete.index[-1], complete.iloc[-1]

    daily_price = web.DataReader("SP500", "fred", anchor_date, dt.date.today()).squeeze().dropna()
    daily_price = daily_price.loc[daily_price.index > anchor_date]
    if daily_price.empty:
        return shiller.loc[:anchor_date]

    tail = pd.DataFrame(index=daily_price.index)
    tail["price"] = daily_price
    tail["dividend"] = anchor["dividend"]
    tail["eps"] = anchor["eps"]
    tail["cape"] = anchor["cape"] * (daily_price / anchor["price"])
    return pd.concat([shiller.loc[:anchor_date], tail])


def get_baa_yield(start_date, end_date):
    """Moody's Seasoned Baa Corporate Bond Yield, spliced for maximum
    resolution: monthly `BAA` (back to 1919) before DBAA's start, genuinely
    daily `DBAA` (native resolution, not resampled) from 1986-01-02 on. See
    the module docstring for why an earlier version of this pipeline used
    monthly-only.
    """
    baa_monthly = web.DataReader("BAA", "fred", start_date, end_date).squeeze()
    baa_daily = web.DataReader("DBAA", "fred", start_date, end_date).squeeze().dropna()
    if baa_daily.empty:
        return baa_monthly
    return pd.concat([baa_monthly.loc[baa_monthly.index < baa_daily.index.min()], baa_daily])


def implied_earnings_growth(spot_yield, target_avg_yield, years=20):
    """Back into the constant annual earnings-growth rate g that reconciles
    today's trailing earnings yield with Baa credit, under one specific
    assumption: g solves

        mean_{t=0..years-1}[ spot_yield * (1+g)^t ] == target_avg_yield

    i.e. hold today's price fixed, let earnings compound at g for `years`
    years starting from latest TTM earnings ("spot"), and require the
    *average* of that path of future earnings yields to equal the Baa
    yield -- read as "the equity risk premium, averaged over the next 20
    years, works out to exactly zero vs. investment-grade credit." This is
    a assumption to hang a number on, not a forecast -- a genuinely
    priced equity risk premium would put the average yield *above* Baa,
    so treat the result as "the growth rate needed for stocks to be
    breakeven against Baa credit," not "the market's actual growth
    expectation."

    Closed form for the average of a geometric path:
        mean = spot_yield * [(1+g)^years - 1] / (years * g)   (g != 0)
    which is strictly increasing in g for spot_yield > 0 (g=-1 collapses
    every future term after t=0 to zero, mean -> spot_yield/years; g=0
    holds earnings flat, mean -> spot_yield; g>0 grows without bound) --
    so there's exactly one root, found by bracketing + brentq rather than
    a closed-form solve (no closed form exists for g here).

    Returns NaN where no root exists in the search range (spot_yield <= 0,
    e.g. an aggregate trailing-earnings collapse, or target_avg_yield <= 0)
    rather than raising -- this runs once per historical date and a few
    crisis-quarter gaps shouldn't break the whole series.
    """
    if not (spot_yield > 0 and target_avg_yield > 0):
        return np.nan

    def avg_path_yield(g):
        if abs(g) < 1e-9:
            return spot_yield
        return spot_yield * ((1 + g) ** years - 1) / (years * g)

    def f(g):
        return avg_path_yield(g) - target_avg_yield

    lo, hi = -0.99, 0.5
    flo, fhi = f(lo), f(hi)
    expansions = 0
    while flo * fhi > 0 and hi < 5.0 and expansions < 20:
        hi *= 1.5
        fhi = f(hi)
        expansions += 1
    if flo * fhi > 0:
        return np.nan
    return brentq(f, lo, hi, xtol=1e-8)


def get_curve_bias_data(start_date, end_date):
    """"Curve bias": a phase-space view of the S&P 500's own price action --
    how far price currently sits above/below its 21-day (~1 trading month)
    average, plotted against how hard it has trended to get there, both
    scaled by the *same* trailing-21-day volatility so the two axes are
    directly comparable in "how many typical daily moves" units:

      y = (price - 21d SMA) / (21d stdev of daily point changes)
          short-term extension: how stretched price is from its own recent
          average, in daily-vol units.
      x = (price - price[21 sessions ago]) / (21d stdev of daily point
          changes * sqrt(21))
          medium-term trend strength: the trailing-month return scaled to
          a *21-day-horizon* expected move (stdev * sqrt(21), the standard
          square-root-of-time scaling), so x is "how many expected-sized
          21-day moves did the market actually make."

    Both use raw daily point changes (not % returns) for the vol estimate,
    matching Bollinger-Band-style technical measures -- the 21-day rolling
    window renormalizes locally, so this stays meaningful even though the
    index's own level (and so its $-point volatility) has roughly tripled
    over the history pulled here.

    Data source: FRED series SP500 (S&P 500, daily close). FRED caps this
    specific series to a trailing ~10 years regardless of the requested
    start date -- a Dow Jones/S&P licensing restriction on FRED's end, the
    same shape of limitation as the ICE high-yield series documented above
    (a `cosd` override doesn't help there either; not re-tested here since
    the FRED series notes state the same license-level cap). Shiller's own
    price series would reach back further but is only monthly -- too coarse
    for a 21-day-window daily-volatility measure -- so this chart's history
    is shorter than the rest of the page's by construction, not a bug.
    """
    price = web.DataReader("SP500", "fred", start_date, end_date).squeeze().dropna()
    daily_move = price.diff()
    std21 = daily_move.rolling(21).std()
    sma21 = price.rolling(21).mean()
    df = pd.DataFrame({
        "price": price,
        "extension": (price - sma21) / std21,
        "trend": (price - price.shift(21)) / (std21 * np.sqrt(21)),
    }).dropna()
    return df


def main():
    print("Pulling Shiller S&P 500 dataset (price, dividends, trailing EPS, CAPE)...")
    shiller = get_shiller_data()
    # Last month with real (non-NaN) dividend/EPS/CAPE -- see add_daily_tail's
    # docstring for why this can trail shiller.index.max() by a couple months.
    shiller_asof = shiller.dropna(subset=["dividend", "eps", "cape"]).index.max()
    print(f"Shiller fundamentals complete through {shiller_asof:%Y-%m}; extending with a daily price tail...")
    shiller = add_daily_tail(shiller)

    start_date = "1919-01-01"  # matches BAA's own start -- the longest-history credit series used here
    end_date = dt.date.today().strftime("%Y-%m-%d")

    print("Pulling Treasury and investment-grade credit yields from FRED...")
    treasury_10y = web.DataReader("DGS10", "fred", start_date, end_date).squeeze()  # genuinely daily, plotted as-is
    baa_yield = get_baa_yield(start_date, end_date)  # monthly pre-1986, daily (DBAA) from 1986 on -- see get_baa_yield()

    earnings_yield = (shiller["eps"] / shiller["price"] * 100).rename("Trailing Earnings Yield")
    cape_yield = (1.0 / shiller["cape"] * 100).rename("CAPE Yield")
    dividend_yield = (shiller["dividend"] / shiller["price"] * 100).rename("Dividend Yield")
    treasury_10y = treasury_10y.rename("10-Year Treasury")
    baa_yield = baa_yield.rename("Baa Corporate Credit")

    # Baa is monthly pre-1986 and only trades on bond-market business days
    # even post-1986 (which don't perfectly coincide with equity trading
    # days) -- ffill onto earnings_yield's own index (same "hold flat
    # between updates" rule the daily tail itself uses for dividend/EPS/
    # CAPE above) so every date has a target to solve
    # implied_earnings_growth() against.
    print("Backing into implied earnings growth (spot yield vs. Baa, 20yr average)...")
    baa_aligned = baa_yield.reindex(earnings_yield.index, method="ffill")
    implied_growth = pd.Series(
        [
            implied_earnings_growth(e / 100.0, b / 100.0) * 100.0 if pd.notna(e) and pd.notna(b) else np.nan
            for e, b in zip(earnings_yield, baa_aligned)
        ],
        index=earnings_yield.index, name="Implied Earnings Growth",
    )
    # Implied Fwd Yield = today's trailing yield compounded one year at the
    # growth rate solved above, i.e. year-1 of the same path whose 20-year
    # average was set equal to Baa. Unlike implied_growth itself (a rate,
    # unbounded and wide-ranging), this is expressed in the same yield
    # units as the other five series and stays in a comparable range (a
    # single year of compounding at a plausible g moves the yield only
    # modestly off its spot value) -- see the module docstring. Short name
    # (vs. the more precise "Implied Forward Earnings Yield") to keep the
    # now-6-entry legend on one row -- see the layout note below.
    implied_fwd_yield = (earnings_yield * (1.0 + implied_growth / 100.0)).rename("Implied Fwd Yield")

    recession_bands = get_recession_bands(start_date, end_date)

    # =========================================================================
    # Single comparison chart: every yield on one %-axis, full available
    # history. Never dual-axis and never a hard-clipped y-range here --
    # earnings yield swings hard in a real recession (e.g. ~0.8% in mid-2009
    # as trailing GAAP earnings collapsed) and clipping would read as a data
    # gap rather than what actually happened.
    #
    # Implied Earnings Growth and Implied Forward Earnings Yield are both
    # MODELED series, not observed -- see implied_earnings_growth()'s
    # docstring. Only the growth-RATE line is hidden by default
    # (visible="legendonly"): its crisis-quarter spikes (e.g. >20% during
    # 2008-09, when trailing earnings briefly collapsed) are real outputs
    # of the assumption, not noise, but they'd swamp the autoranged axis for
    # the other five directly-observed series if shown at the same time.
    # The forward-YIELD line stays in a comparable range to the others (one
    # year of compounding moves a yield only modestly), so it's drawn like
    # any other line, just dashed as a visual cue that it's derived.
    # =========================================================================
    fig = go.Figure()
    for series in [treasury_10y, baa_yield, dividend_yield, earnings_yield, cape_yield, implied_fwd_yield]:
        d = series.dropna()
        fig.add_trace(go.Scatter(
            x=d.index, y=d.values, name=series.name,
            line=dict(color=SERIES_COLOR[series.name], width=2, dash="dot" if series is implied_fwd_yield else "solid"),
        ))
    d = implied_growth.dropna()
    fig.add_trace(go.Scatter(
        x=d.index, y=d.values, name=implied_growth.name,
        line=dict(color=SERIES_COLOR[implied_growth.name], width=2, dash="dot"),
        visible="legendonly",
    ))
    add_recession_bands(fig, recession_bands)
    FIGURES["yield_comparison"] = style_fig(
        fig, "S&P 500 Fundamental Yields vs. the Rest of the Capital Structure", yaxis_title="%", height=520
    )
    # BUG FIX (same shape as the legend/title collision this chart hit
    # before dropping the HY line -- see that commit): back up to 7 legend
    # entries now (6 visible + Implied Earnings Growth hidden-but-still-
    # occupying-a-legend-slot). Even with short labels this wraps to 2
    # lines below ~1000px container width and the default t=70/
    # yanchor="bottom" gives a wrapped legend nowhere to go but on top of
    # the title. t=110 (matches height=520 above) plus yanchor="top"
    # (anchors the legend's TOP at y=1.0, hanging downward) fixes it the
    # same way as before. Verified by rendering at 800px/1200px with
    # kaleido: no collision at either width now.
    FIGURES["yield_comparison"].update_layout(
        margin=dict(l=60, r=30, t=110, b=40),
        legend=dict(orientation="h", yanchor="top", y=1.0, xanchor="left", x=0, font=dict(size=11)),
    )

    # =========================================================================
    # Curve bias: S&P 500 price action's own short-term extension vs.
    # medium-term trend strength, both in trailing-21-day-vol units. See
    # get_curve_bias_data()'s docstring for the exact construction. Rendered
    # as three layers rather than one flat scatter: the full (~10yr, FRED's
    # license cap) history as light context, the last 60 sessions as a
    # visible path so the *direction* of travel through this space is
    # readable (a single snapshot point can't show that), and today singled
    # out. hovermode="closest" overrides style_fig's default "x unified" --
    # unified-on-x only makes sense for a time series sharing one x-axis,
    # and this chart's x-axis is a value, not a date.
    # =========================================================================
    print("Computing curve bias (price extension vs. trend, in trailing-vol units)...")
    cb = get_curve_bias_data(start_date, end_date)
    trail = cb.iloc[-60:]
    latest = cb.iloc[-1]

    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=cb["trend"], y=cb["extension"], mode="markers", name="Full history",
        marker=dict(size=4, color=INK_MUTED, opacity=0.30), hoverinfo="skip",
    ))
    fig.add_trace(go.Scatter(
        x=trail["trend"], y=trail["extension"], mode="lines+markers", name="Last 60 sessions",
        line=dict(color=CAT[0], width=1.5), marker=dict(size=5, color=CAT[0]),
        text=[d.strftime("%Y-%m-%d") for d in trail.index], hovertemplate="%{text}<br>x=%{x:.2f}  y=%{y:.2f}<extra></extra>",
    ))
    fig.add_trace(go.Scatter(
        x=[latest["trend"]], y=[latest["extension"]], mode="markers", name="Latest",
        marker=dict(size=12, color=CAT[7], line=dict(color=SURFACE, width=1.5)),
        text=[cb.index[-1].strftime("%Y-%m-%d")], hovertemplate="%{text}<br>x=%{x:.2f}  y=%{y:.2f}<extra></extra>",
    ))
    fig.add_hline(y=0, line=dict(color=BASELINE, width=1, dash="dot"))
    fig.add_vline(x=0, line=dict(color=BASELINE, width=1, dash="dot"))
    FIGURES["curve_bias"] = style_fig(
        fig, "Curve Bias: Short-Term Extension vs. Medium-Term Trend",
        yaxis_title="Extension vs. 21d avg (σ of daily moves)", height=520, hovermode="closest",
    )
    FIGURES["curve_bias"].update_xaxes(title=dict(text="21d trend (σ of daily moves, √21-scaled)", font=dict(size=11, color=INK_MUTED)))

    # =========================================================================
    # Curve bias components, plotted individually over time. The scatter
    # above shows the two axes against EACH OTHER but drops the date
    # dimension entirely (x/y position only, no sense of when) -- these two
    # single-series time-series charts are the same y_extension/x_trend
    # values cb already has, just each plotted against its own date so the
    # standalone behavior of each axis (not just their joint position) is
    # readable. Single series each -> no legend needed (the title names it).
    # =========================================================================
    ext_fig = go.Figure()
    ext_fig.add_trace(go.Scatter(x=cb.index, y=cb["extension"], mode="lines", line=dict(color=CAT[0], width=1.5)))
    ext_fig.add_hline(y=0, line=dict(color=BASELINE, width=1, dash="dot"))
    add_recession_bands(ext_fig, recession_bands, xmin=cb.index[0])
    FIGURES["curve_bias_extension"] = style_fig(
        ext_fig, "Curve Bias -- Y: Short-Term Extension", yaxis_title="σ of daily moves", height=280, legend=False,
    )

    trend_fig = go.Figure()
    trend_fig.add_trace(go.Scatter(x=cb.index, y=cb["trend"], mode="lines", line=dict(color=CAT[1], width=1.5)))
    trend_fig.add_hline(y=0, line=dict(color=BASELINE, width=1, dash="dot"))
    add_recession_bands(trend_fig, recession_bands, xmin=cb.index[0])
    FIGURES["curve_bias_trend"] = style_fig(
        trend_fig, "Curve Bias -- X: Medium-Term Trend", yaxis_title="σ of daily moves, √21-scaled", height=280, legend=False,
    )

    # =========================================================================
    # Save everything
    # =========================================================================
    print(f"Saving {len(FIGURES)} figures + meta to {DATA_DIR}...")
    for key, f in FIGURES.items():
        pio.write_json(f, FIGURES_DIR / f"{key}.json")

    META.update(
        last_updated=dt.datetime.now(dt.timezone.utc).isoformat(),
        data_asof=shiller.index.max().strftime("%Y-%m-%d"),
        shiller_asof=shiller_asof.strftime("%Y-%m-%d"),
        earnings_yield=round(float(earnings_yield.dropna().iloc[-1]), 2),
        cape_yield=round(float(cape_yield.dropna().iloc[-1]), 2),
        dividend_yield=round(float(dividend_yield.dropna().iloc[-1]), 2),
        treasury_10y=round(float(treasury_10y.dropna().iloc[-1]), 2),
        baa_yield=round(float(baa_yield.dropna().iloc[-1]), 2),
        earnings_yield_vs_baa=round(float(earnings_yield.dropna().iloc[-1] - baa_yield.dropna().iloc[-1]), 2),
        implied_earnings_growth=round(float(implied_growth.dropna().iloc[-1]), 2) if implied_growth.dropna().size else None,
        implied_fwd_earnings_yield=round(float(implied_fwd_yield.dropna().iloc[-1]), 2) if implied_fwd_yield.dropna().size else None,
        curve_bias_extension=round(float(latest["extension"]), 2),
        curve_bias_trend=round(float(latest["trend"]), 2),
        curve_bias_asof=cb.index[-1].strftime("%Y-%m-%d"),
    )
    with open(DATA_DIR / "meta.json", "w") as fh:
        json.dump(META, fh, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
