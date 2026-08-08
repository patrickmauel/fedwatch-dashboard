"""
Rates & Macro data pipeline (one section of the larger dashboard).

Pulls live data (FRED, NY Fed, Cleveland Fed, U.S. Treasury), runs the
Laubach-Williams-based rate/growth/inflation model and Monte Carlo
simulation, builds every dashboard chart, and writes:

  data/rates_macro/figures/*.json  -- one Plotly figure per chart (plotly.io JSON)
  data/rates_macro/fomc_grid.csv   -- the FOMC data-release grid as a plain table
  data/rates_macro/meta.json       -- last-updated timestamp + headline stats

Run standalone: `python pipelines/rates_macro.py` (from the repo root).
Intended to be run once a day by the GitHub Actions workflow in
.github/workflows/daily_refresh.yml; pages/rates_macro.py only ever reads
the files this script writes, never the network.
"""
import datetime as dt
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pandas_datareader.data as web
import plotly.graph_objects as go
import plotly.io as pio
import statsmodels.formula.api as smf
from plotly.subplots import make_subplots
from scipy.interpolate import PchipInterpolator

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "rates_macro"
FIGURES_DIR = DATA_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FIGURES = {}  # key -> plotly Figure, saved to data/figures/{key}.json at the end
META = {}  # headline stats, saved to data/meta.json at the end


# =============================================================================
# 1. Data helper functions
# =============================================================================
def get_gdpnowcast():
    """New York Fed Staff Nowcast: real GDP growth nowcast by forecast horizon."""
    url = (
        "https://www.newyorkfed.org/medialibrary/Research/Interactives/Data/NowCast/"
        "Downloads/New-York-Fed-Staff-Nowcast_download_data.xlsx"
    )
    df = pd.read_excel(url, sheet_name="Forecasts By Horizon", index_col=0, header=5)
    df = df.drop(columns="Reference quarter")
    return df.astype(float)


def get_gdpgrowth(start_date, end_date):
    """Real GDP level, YoY growth, and QoQ annualized growth from FRED (GDPC1)."""
    rGDP_level = web.DataReader("GDPC1", "fred", start_date, end_date)
    rGDP = rGDP_level.pct_change(4).squeeze() * 100.0
    rGDP_qoq = rGDP_level.pct_change().squeeze() * 4 * 100.0
    return rGDP, rGDP_qoq, rGDP_level


def pull_gdpgrowth_proj():
    """Latest FOMC Summary of Economic Projections for real GDP growth, from FRED."""
    series_ids = {
        "Median": "GDPC1MD",
        "CentralTendencyLow": "GDPC1CTL",
        "CentralTendencyHigh": "GDPC1CTH",
        "RangeLow": "GDPC1RL",
        "RangeHigh": "GDPC1RH",
    }
    start = dt.datetime(2020, 1, 1)
    end = dt.datetime.now() + pd.offsets.DateOffset(years=3)
    gdp_proj = pd.DataFrame()
    for label, series in series_ids.items():
        gdp_proj[label] = web.DataReader(series, "fred", start, end).squeeze()
    gdp_proj = gdp_proj.dropna().iloc[-4:]
    gdp_proj.index = gdp_proj.index.year.astype(str)
    gdp_proj.index = pd.to_datetime(gdp_proj.index + "-12-31")
    gdp_proj = gdp_proj.astype(float)
    gdp_proj = gdp_proj.reindex(pd.date_range(dt.date.today(), gdp_proj.index.max())).interpolate().bfill()
    return gdp_proj


def pull_mct():
    """NY Fed Multivariate Core Trend (MCT) inflation."""
    excel_path = (
        "https://www.newyorkfed.org/medialibrary/Research/Interactives/mct/downloads/"
        "NYFed_MCT-Inflation_data"
    )
    xls = pd.ExcelFile(excel_path)
    last_sheet = xls.sheet_names[-1]
    df = pd.read_excel(xls, sheet_name=last_sheet, index_col=[0], header=[4, 5])
    df.index = pd.to_datetime(df.index) + pd.offsets.MonthEnd(0)
    return df.astype(float)


def get_pce(start_date, end_date):
    """Headline and core PCE inflation, YoY and annualized MoM, from FRED."""
    pcepi = web.DataReader("PCEPI", "fred", start_date, end_date).squeeze()
    pce_mom = (np.power(1 + pcepi.pct_change(), 12) - 1.0) * 100.0
    pce = pcepi.pct_change(12) * 100.0

    pcepilfe = web.DataReader("PCEPILFE", "fred", start_date, end_date).squeeze()
    pce_core_mom = (np.power(1 + pcepilfe.pct_change(), 12) - 1.0) * 100.0
    pce_core = pcepilfe.pct_change(12) * 100.0

    for s in (pce, pce_mom, pce_core, pce_core_mom):
        s.index = s.index + pd.offsets.MonthEnd(0)

    return pce, pce_mom, pce_core, pce_core_mom


def pull_inflation_fcst():
    """Cleveland Fed model-based inflation expectations."""
    url = (
        "https://www.clevelandfed.org/-/media/files/webcharts/inflationexpectations/"
        "inflation-expectations.xlsx"
    )
    df = pd.read_excel(url, sheet_name="Expected Inflation", index_col=0)
    df.index = pd.to_datetime(df.index)
    df.columns = df.columns.str.split(" ").str[1].astype(float)
    df = df.tail(1).T * 100.0
    df.index = [df.columns[0] + pd.offsets.DateOffset(years=i) for i in df.index]
    df.index = pd.to_datetime(df.index)
    df.columns = ["Inflation_exp"]
    return df


TREASURY_TENOR_YEARS = {
    "1 Mo": 1 / 12, "1.5 Month": 1.5 / 12, "2 Mo": 2 / 12, "3 Mo": 3 / 12,
    "4 Mo": 4 / 12, "6 Mo": 6 / 12, "1 Yr": 1.0, "2 Yr": 2.0, "3 Yr": 3.0,
    "5 Yr": 5.0, "7 Yr": 7.0, "10 Yr": 10.0, "20 Yr": 20.0, "30 Yr": 30.0,
}


def get_treasury_curve(asof=None):
    """
    Pull the most recent U.S. Treasury daily par yield curve and bootstrap it
    into a zero (spot) curve and a continuously-compounded forward curve.
    """
    if asof is None:
        asof = dt.date.today()
    year = asof.year

    url = (
        "https://home.treasury.gov/resource-center/data-chart-center/interest-rates/"
        "daily-treasury-rates.csv/{year}/all?type=daily_treasury_yield_curve&"
        "field_tdr_date_value={year}&page&_format=csv"
    ).format(year=year)

    df = pd.read_csv(url)
    df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
    df = df.set_index("Date").sort_index()

    if df.dropna(how="all").empty and year == dt.date.today().year:
        year -= 1
        url = url.replace(str(asof.year), str(year))
        df = pd.read_csv(url)
        df["Date"] = pd.to_datetime(df["Date"], format="%m/%d/%Y")
        df = df.set_index("Date").sort_index()

    latest = df.dropna(how="all").iloc[-1]
    latest_date = df.dropna(how="all").index[-1]

    par = pd.Series(
        {TREASURY_TENOR_YEARS[k]: v for k, v in latest.items() if k in TREASURY_TENOR_YEARS}
    ).sort_index()
    par = par.dropna() / 100.0

    return _bootstrap_par_curve(par, latest_date)


def _bootstrap_par_curve(par, asof_date, step=0.5, max_maturity=30.0):
    """Bootstrap a semiannual-pay par yield curve into discount factors, spot
    rates, and continuously-compounded forward rates."""
    grid = np.round(np.arange(step, max_maturity + 1e-9, step), 4)

    x = par.index.values
    y = par.values
    interp = PchipInterpolator(x, y, extrapolate=False)
    par_grid = interp(grid)
    par_grid[grid <= x.min()] = y[0]
    par_grid[grid >= x.max()] = y[-1]

    discount = {}
    cum_coupon_df = 0.0
    for t, c in zip(grid, par_grid):
        if t <= 1.0:
            df_t = 1.0 / (1.0 + c / 2.0) ** (2 * t)
        else:
            df_t = (1.0 - (c / 2.0) * cum_coupon_df) / (1.0 + c / 2.0)
        discount[t] = df_t
        cum_coupon_df += df_t

    discount = pd.Series(discount).sort_index()
    spot = -np.log(discount) / discount.index.values

    fwd = {}
    idx = discount.index.values
    dfs = discount.values
    for i in range(len(idx) - 1):
        t1, t2 = idx[i], idx[i + 1]
        d1, d2 = dfs[i], dfs[i + 1]
        fwd[t2] = (np.log(d1) - np.log(d2)) / (t2 - t1)
    fwd[idx[0]] = spot.iloc[0]
    fwd = pd.Series(fwd).sort_index()

    out = pd.DataFrame(
        {
            "tenor_years": discount.index,
            "discount_factor": discount.values,
            "spot_rate": spot.values * 100.0,
            "forward_rate": fwd.reindex(discount.index).values * 100.0,
        }
    )
    out["maturity_date"] = [asof_date + pd.DateOffset(days=round(t * 365.25)) for t in out["tenor_years"]]
    out = out.set_index("maturity_date")
    out.attrs["asof_date"] = asof_date
    return out


# =============================================================================
# Shared chart styling (dataviz house method)
# =============================================================================
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
STATUS_GOOD, STATUS_CRITICAL = "#0ca30c", "#d03b3b"
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDLINE, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
RECESSION_FILL = "rgba(137,135,129,0.16)"


def style_fig(fig, title, yaxis_title=None, height=380, legend=True):
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


def main():
    start_date = "1990-01-01"
    end_date = dt.date.today().strftime("%Y-%m-%d")

    # =========================================================================
    # 2. Real GDP: nowcast, actuals, FOMC projections
    # =========================================================================
    print("Pulling GDP data...")
    df_gdpnowcast = get_gdpnowcast()["Nowcast (current quarter)"]
    df_gdpnowcast.columns = ["rGDPg"]
    df_gdpnowcast_annual = (
        np.power(1 + df_gdpnowcast.resample("D").bfill() / 100.0, 1 / 365.0)
        .rolling(window=365)
        .apply(np.prod, raw=True)
        - 1.0
    ) * 100.0

    rGDP, rGDP_qoq, rGDP_level = get_gdpgrowth(start_date, end_date)
    df_gdpgrowth = pull_gdpgrowth_proj()

    rGDP.index = rGDP.index + pd.offsets.QuarterEnd()
    rGDP_qoq.index = rGDP_qoq.index + pd.offsets.QuarterEnd()
    rGDP_level.index = rGDP_level.index + pd.offsets.QuarterEnd()

    # =========================================================================
    # 3. Inflation: trend measures, PCE, expectations
    # =========================================================================
    print("Pulling inflation data...")
    df_mct = pull_mct()
    df_mct_total = df_mct["MCT Inflation"]
    pce, pce_mom, pce_core, pce_core_mom = get_pce(start_date, end_date)

    df_pcefcst = pull_inflation_fcst()
    df_pcefcst = df_pcefcst.loc[: df_gdpgrowth.index.max() + pd.offsets.DateOffset(years=1)].resample("D").interpolate()
    df_pcefcst = df_pcefcst.reindex(pd.date_range(dt.date.today(), df_gdpgrowth.index.max())).bfill()

    # =========================================================================
    # 4. Laubach-Williams natural rate model
    # =========================================================================
    print("Pulling Laubach-Williams r* model...")
    lw_url = (
        "https://www.newyorkfed.org/medialibrary/media/research/economists/williams/data/"
        "Laubach_Williams_current_estimates.xlsx"
    )
    df_lwrstar = pd.read_excel(lw_url, sheet_name="data", header=[4, 5], index_col=0)["One-Sided Estimates"].drop(columns=["Output gap.1"])
    df_lwrstar.index = df_lwrstar.index + pd.offsets.QuarterEnd()

    df_lwrstar_input = pd.read_excel(lw_url, sheet_name="input data", header=[0], index_col=0)
    df_lwrstar_input.index = df_lwrstar_input.index + pd.offsets.QuarterEnd()

    df_parameters = pd.read_excel(lw_url, sheet_name="parameters", header=[2], index_col=0).loc["Estimate", :]
    df_parameters_secondary = pd.read_excel(lw_url, sheet_name="parameters", header=[7], index_col=0).loc[:, "Unnamed: 1"]
    # NB: the source spreadsheet's own column labels have trailing spaces on a
    # couple of parameters ('b_3 ', 'sigma_3 ') -- kept verbatim since the
    # simulation below indexes df_parameters by these exact (quirky) labels.
    df_parameters = pd.concat([df_parameters, df_parameters_secondary.loc[["sigma_3 ", "sigma_5", "sigma_r*"]]])

    df_rGDP_potential_level = np.exp(np.log(rGDP_level)["GDPC1"] - df_lwrstar.loc[:, "Output gap"] / 100.0)

    # =========================================================================
    # 5. Monetary policy reaction function
    # =========================================================================
    print("Fitting policy rule...")
    ff = web.DataReader("DFF", "fred", "1960-01-01", end_date).squeeze()
    ff_q = ff.resample("QE").mean()

    df_ff = pd.concat(
        {"FF_0": ff_q, "FF_1": ff_q.shift(1), "FF_2": ff_q.shift(2), "FF_3": ff_q.shift(3), "FF_4": ff_q.shift(4)},
        axis=1,
    ).iloc[:-1]
    df_ff.loc[:, "pi_0"] = df_mct_total.loc[:, "Median"].resample("QE").mean().shift(1)
    df_ff.loc[:, "ybar_0"] = df_lwrstar.loc[:, "Output gap"].shift(1)
    df_ff.loc[:, "rstar"] = df_lwrstar.loc[:, "rstar"]
    df_ff.loc[:, "pibar_0"] = df_ff.loc[:, "pi_0"] - 2.0

    coeff_set = 0.75
    df_ff.loc[:, "FF_0_delta"] = df_ff.loc[:, "FF_0"] - coeff_set * df_ff.loc[:, "FF_1"]
    fit_policymodel = smf.ols("FF_0_delta ~ rstar + pibar_0 + ybar_0 -1", data=df_ff).fit()

    # =========================================================================
    # 6. Bridge LW's last estimate to the most recent actual data
    # =========================================================================
    STATE_COLS = [
        "ystar", "g", "rstar_0", "rstar_1", "rstar_2", "rstar_3", "rstar_4",
        "ybar_0", "ybar_1", "ybar_2", "z",
        "pi_0", "pi_1", "pi_2", "pi_3", "pi_4", "pibar_0",
        "FF_0", "FF_1", "FF_2", "FF_3", "FF_4",
    ]

    fcst_start_lw = df_lwrstar.dropna().index[-1]

    today = dt.date.today()
    current_quarter_start = dt.date(today.year, 3 * ((today.month - 1) // 3) + 1, 1)
    last_full_calendar_quarter = pd.Timestamp(current_quarter_start) - pd.DateOffset(days=1)
    last_actual_quarter = min(rGDP_level.index.max(), df_mct_total.index.max(), last_full_calendar_quarter)
    last_actual_quarter = max(last_actual_quarter, fcst_start_lw)

    fcst_end = last_actual_quarter + pd.DateOffset(years=3)
    print(f"LW's last complete quarter:        {fcst_start_lw:%Y-%m-%d}")
    print(f"Last quarter with all actual data: {last_actual_quarter:%Y-%m-%d}")

    df_lwtemplate = pd.DataFrame(index=pd.date_range(fcst_start_lw, fcst_end, freq="QE"), columns=STATE_COLS)

    df_lwtemplate.loc[fcst_start_lw, "g"] = df_lwrstar.loc[fcst_start_lw, "g"]
    df_lwtemplate.loc[fcst_start_lw, "rstar_0"] = df_lwrstar.loc[fcst_start_lw, "rstar"]
    df_lwtemplate.loc[fcst_start_lw, "rstar_1"] = df_lwrstar.loc[:fcst_start_lw, "rstar"].iloc[-2]
    df_lwtemplate.loc[fcst_start_lw, "rstar_2"] = df_lwrstar.loc[:fcst_start_lw, "rstar"].iloc[-3]
    df_lwtemplate.loc[fcst_start_lw, "rstar_3"] = df_lwrstar.loc[:fcst_start_lw, "rstar"].iloc[-4]
    df_lwtemplate.loc[fcst_start_lw, "rstar_4"] = df_lwrstar.loc[:fcst_start_lw, "rstar"].iloc[-5]

    df_lwtemplate.loc[fcst_start_lw, "ybar_0"] = df_lwrstar.loc[fcst_start_lw, "Output gap"]
    df_lwtemplate.loc[fcst_start_lw, "ybar_1"] = df_lwrstar.loc[:fcst_start_lw, "Output gap"].iloc[-2]
    df_lwtemplate.loc[fcst_start_lw, "ybar_2"] = df_lwrstar.loc[:fcst_start_lw, "Output gap"].iloc[-3]

    df_lwtemplate.loc[fcst_start_lw, "z"] = df_lwrstar.loc[fcst_start_lw, "z"]
    df_lwtemplate.loc[fcst_start_lw, "ystar"] = np.log(df_rGDP_potential_level.loc[fcst_start_lw])

    df_lwtemplate.loc[fcst_start_lw, "pi_0"] = df_mct_total.loc[fcst_start_lw, "Median"]
    df_lwtemplate.loc[fcst_start_lw, "pi_1"] = df_mct_total.loc[:fcst_start_lw, "Median"].iloc[-2]
    df_lwtemplate.loc[fcst_start_lw, "pi_2"] = df_mct_total.loc[:fcst_start_lw, "Median"].iloc[-3]
    df_lwtemplate.loc[fcst_start_lw, "pi_3"] = df_mct_total.loc[:fcst_start_lw, "Median"].iloc[-4]
    df_lwtemplate.loc[fcst_start_lw, "pi_4"] = df_mct_total.loc[:fcst_start_lw, "Median"].iloc[-5]

    df_lwtemplate.loc[fcst_start_lw, "pibar_0"] = df_lwrstar_input.loc[fcst_start_lw, "inflation"] - 2.0

    df_lwtemplate.loc[fcst_start_lw, "FF_0"] = df_ff.loc[fcst_start_lw, "FF_0"]
    df_lwtemplate.loc[fcst_start_lw, "FF_1"] = df_ff.loc[fcst_start_lw, "FF_1"]
    df_lwtemplate.loc[fcst_start_lw, "FF_2"] = df_ff.loc[fcst_start_lw, "FF_2"]
    df_lwtemplate.loc[fcst_start_lw, "FF_3"] = df_ff.loc[fcst_start_lw, "FF_3"]
    df_lwtemplate.loc[fcst_start_lw, "FF_4"] = df_ff.loc[fcst_start_lw, "FF_4"]

    for dt_prev, dt_next in zip(df_lwtemplate.index, df_lwtemplate.index[1:]):
        if dt_next > last_actual_quarter:
            break
        df_lwtemplate.loc[dt_next, "g"] = df_lwtemplate.loc[dt_prev, "g"]
        df_lwtemplate.loc[dt_next, "z"] = df_lwtemplate.loc[dt_prev, "z"]
        df_lwtemplate.loc[dt_next, "rstar_0"] = df_parameters["c"] * df_lwtemplate.loc[dt_next, "g"] + df_lwtemplate.loc[dt_next, "z"]
        df_lwtemplate.loc[dt_next, "ystar"] = df_lwtemplate.loc[dt_prev, "ystar"] + df_lwtemplate.loc[dt_prev, "g"] / 400.0

        df_lwtemplate.loc[dt_next, "ybar_0"] = 100.0 * (np.log(rGDP_level.loc[dt_next, "GDPC1"]) - df_lwtemplate.loc[dt_next, "ystar"])
        df_lwtemplate.loc[dt_next, "pi_0"] = df_mct_total.loc[dt_next, "Median"]
        df_lwtemplate.loc[dt_next, "FF_0"] = ff_q.loc[dt_next]

        df_lwtemplate.loc[dt_next, "rstar_1"] = df_lwtemplate.loc[dt_prev, "rstar_0"]
        df_lwtemplate.loc[dt_next, "rstar_2"] = df_lwtemplate.loc[dt_prev, "rstar_1"]
        df_lwtemplate.loc[dt_next, "rstar_3"] = df_lwtemplate.loc[dt_prev, "rstar_2"]
        df_lwtemplate.loc[dt_next, "rstar_4"] = df_lwtemplate.loc[dt_prev, "rstar_3"]

        df_lwtemplate.loc[dt_next, "ybar_1"] = df_lwtemplate.loc[dt_prev, "ybar_0"]
        df_lwtemplate.loc[dt_next, "ybar_2"] = df_lwtemplate.loc[dt_prev, "ybar_1"]

        df_lwtemplate.loc[dt_next, "pi_1"] = df_lwtemplate.loc[dt_prev, "pi_0"]
        df_lwtemplate.loc[dt_next, "pi_2"] = df_lwtemplate.loc[dt_prev, "pi_1"]
        df_lwtemplate.loc[dt_next, "pi_3"] = df_lwtemplate.loc[dt_prev, "pi_2"]
        df_lwtemplate.loc[dt_next, "pi_4"] = df_lwtemplate.loc[dt_prev, "pi_3"]

        df_lwtemplate.loc[dt_next, "FF_1"] = df_lwtemplate.loc[dt_prev, "FF_0"]
        df_lwtemplate.loc[dt_next, "FF_2"] = df_lwtemplate.loc[dt_prev, "FF_1"]
        df_lwtemplate.loc[dt_next, "FF_3"] = df_lwtemplate.loc[dt_prev, "FF_2"]
        df_lwtemplate.loc[dt_next, "FF_4"] = df_lwtemplate.loc[dt_prev, "FF_3"]

        df_lwtemplate.loc[dt_next, "pibar_0"] = df_lwtemplate.loc[dt_prev, "pi_0"] - 2.0

    fcst_start = last_actual_quarter

    # =========================================================================
    # 7. Monte Carlo simulation
    # =========================================================================
    print("Running Monte Carlo simulation (1000 paths)...")
    nsims = 1000
    rng = np.random.default_rng()

    dict_sims = {}
    for n in range(nsims):
        loop = df_lwtemplate.loc[fcst_start:].copy()
        for i in range(len(loop.index) - 1):
            dt_prev = loop.index[i]
            dt_next = loop.index[i + 1]

            loop.loc[dt_next, "g"] = loop.loc[dt_prev, "g"] + df_parameters["sigma_5"] * 4 * rng.normal()
            loop.loc[dt_next, "z"] = loop.loc[dt_prev, "z"] + df_parameters["sigma_3 "] * rng.normal()
            loop.loc[dt_next, "rstar_0"] = df_parameters["c"] * loop.loc[dt_next, "g"] + loop.loc[dt_next, "z"]
            loop.loc[dt_next, "ystar"] = (
                loop.loc[dt_prev, "ystar"] + loop.loc[dt_prev, "g"] / 400 + df_parameters["sigma_4"] / 400.0 * rng.normal()
            )

            loop.loc[dt_next, "FF_0"] = (
                fit_policymodel.params["rstar"] * loop.loc[dt_next, "rstar_0"]
                + fit_policymodel.params["pibar_0"] * loop.loc[dt_prev, "pibar_0"]
                + fit_policymodel.params["ybar_0"] * loop.loc[dt_prev, "ybar_0"]
            ) + coeff_set * loop.loc[dt_prev, "FF_0"]
            loop.loc[dt_next, "FF_0"] = max(loop.loc[dt_next, "FF_0"], 0.0)

            loop.loc[dt_next, "ybar_0"] = (
                df_parameters["a_1"] * loop.loc[dt_prev, "ybar_0"]
                + df_parameters["a_2"] * loop.loc[dt_prev, "ybar_1"]
                + df_parameters["a_3"] * loop.loc[dt_prev, "ybar_2"]
                + df_parameters["b_1"] * (loop.loc[dt_prev, "FF_0"] - loop.loc[dt_prev, "pi_0"] - loop.loc[dt_prev, "rstar_0"]) / 100.0
                + df_parameters["b_2"] * (loop.loc[dt_prev, "FF_1"] - loop.loc[dt_prev, "pi_1"] - loop.loc[dt_prev, "rstar_1"]) / 100.0
                + df_parameters["b_3 "] * (loop.loc[dt_prev, "FF_2"] - loop.loc[dt_prev, "pi_2"] - loop.loc[dt_prev, "rstar_2"]) / 100.0
                + df_parameters["b_4"] * (loop.loc[dt_prev, "FF_3"] - loop.loc[dt_prev, "pi_3"] - loop.loc[dt_prev, "rstar_3"]) / 100.0
                + df_parameters["b_5"] * (loop.loc[dt_prev, "FF_4"] - loop.loc[dt_prev, "pi_4"] - loop.loc[dt_prev, "rstar_4"]) / 100.0
                + df_parameters["sigma_1"] * rng.normal()
            )

            loop.loc[dt_next, "pi_0"] = (
                loop.loc[dt_prev, "pi_0"] + df_parameters["phi"] * loop.loc[dt_prev, "ybar_0"] + df_parameters["sigma_2"] * rng.normal()
            )

            loop.loc[dt_next, "rstar_1"] = loop.loc[dt_prev, "rstar_0"]
            loop.loc[dt_next, "rstar_2"] = loop.loc[dt_prev, "rstar_1"]
            loop.loc[dt_next, "rstar_3"] = loop.loc[dt_prev, "rstar_2"]
            loop.loc[dt_next, "rstar_4"] = loop.loc[dt_prev, "rstar_3"]

            loop.loc[dt_next, "ybar_1"] = loop.loc[dt_prev, "ybar_0"]
            loop.loc[dt_next, "ybar_2"] = loop.loc[dt_prev, "ybar_1"]

            loop.loc[dt_next, "pi_1"] = loop.loc[dt_prev, "pi_0"]
            loop.loc[dt_next, "pi_2"] = loop.loc[dt_prev, "pi_1"]
            loop.loc[dt_next, "pi_3"] = loop.loc[dt_prev, "pi_2"]
            loop.loc[dt_next, "pi_4"] = loop.loc[dt_prev, "pi_3"]

            loop.loc[dt_next, "FF_1"] = loop.loc[dt_prev, "FF_0"]
            loop.loc[dt_next, "FF_2"] = loop.loc[dt_prev, "FF_1"]
            loop.loc[dt_next, "FF_3"] = loop.loc[dt_prev, "FF_2"]
            loop.loc[dt_next, "FF_4"] = loop.loc[dt_prev, "FF_3"]

            loop.loc[dt_next, "pibar_0"] = loop.loc[dt_prev, "pi_0"] - 2.0

        dict_sims[n] = loop

    df_sims = pd.concat(dict_sims, axis=1).swaplevel(0, 1, axis=1).sort_index(axis=1)
    df_sims = df_sims.apply(pd.to_numeric, errors="coerce")

    # =========================================================================
    # 8. Aggregate simulated paths
    # =========================================================================
    rGDP_potential_fcst = np.e ** (df_sims["ystar"])
    rGDP_fcst = np.e ** (df_sims["ybar_0"] / 100.0 + df_sims["ystar"])

    rGDP_potential_fcst = rGDP_potential_fcst.reindex(
        pd.date_range(df_rGDP_potential_level.index.min(), rGDP_potential_fcst.index.max(), freq="QE")
    )
    rGDP_potential_fcst = rGDP_potential_fcst.apply(lambda row: row.combine_first(df_rGDP_potential_level), axis=0)

    rGDP_fcst = rGDP_fcst.reindex(pd.date_range(rGDP_level.index.min(), rGDP_fcst.index.max(), freq="QE"))
    rGDP_fcst = rGDP_fcst.apply(lambda row: row.combine_first(rGDP_level["GDPC1"]), axis=0)
    rGDP_g_fcst = rGDP_fcst.pct_change(4) * 100.0

    pi_fcst = df_sims["pi_0"]
    ff_fcst = df_sims["FF_0"]

    # =========================================================================
    # 9. Term structure
    # =========================================================================
    print("Bootstrapping Treasury curve...")
    df_treasury_curve = get_treasury_curve()
    asof_date = df_treasury_curve.attrs["asof_date"]

    fwd_daily = df_treasury_curve["forward_rate"].copy()
    fwd_daily.index = pd.to_datetime(fwd_daily.index)
    fwd_daily = fwd_daily.reindex(pd.date_range(asof_date, fwd_daily.index.max())).bfill()

    # =========================================================================
    # 10. Main dashboard (2x2)
    # =========================================================================
    print("Building charts...")

    def add_fan(fig, row, col, legend, df_fcst, color, name, y_start):
        d = df_fcst.loc[y_start:]
        lo = pd.to_numeric(d.quantile(0.16, axis=1))
        hi = pd.to_numeric(d.quantile(0.84, axis=1))
        mean = pd.to_numeric(d.mean(axis=1))
        fig.add_trace(go.Scatter(x=lo.index, y=lo.values, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=row, col=col)
        fig.add_trace(
            go.Scatter(x=hi.index, y=hi.values, mode="lines", line=dict(width=0), fill="tonexty",
                       fillcolor=color.replace("rgb", "rgba").replace(")", ",0.25)"), name=f"{name} 1-sigma", hoverinfo="skip", legend=legend),
            row=row, col=col,
        )
        fig.add_trace(go.Scatter(x=mean.index, y=mean.values, mode="lines", line=dict(color=color, dash="dash"), name=f"{name} mean", legend=legend), row=row, col=col)

    y_start = "2023"
    fig = make_subplots(
        rows=2, cols=2,
        subplot_titles=("rGDP Realizations and Projections", "PCE Realizations and Projections", "GDP Modeled", "Fed Funds: Model Projection vs. Treasury-Implied Forward Path"),
        vertical_spacing=0.12, horizontal_spacing=0.08,
    )
    L1, L2, L3, L4 = "legend", "legend2", "legend3", "legend4"

    d = df_gdpnowcast.loc[y_start:]
    fig.add_trace(go.Scatter(x=d.index, y=d.values, name="rGDP Nowcast", line=dict(width=2), legend=L1), row=1, col=1)
    d = df_gdpnowcast_annual.loc[y_start:]
    fig.add_trace(go.Scatter(x=d.index, y=d.values, name="rGDP Nowcast Annual", line=dict(width=2), legend=L1), row=1, col=1)
    d = rGDP.loc[y_start:]
    fig.add_trace(go.Scatter(x=d.index, y=d.values, name="rGDP Actual YoY", line=dict(width=2, color="green"), legend=L1), row=1, col=1)
    add_fan(fig, 1, 1, L1, rGDP_g_fcst, "rgb(44,160,44)", "rGDP growth", y_start)

    d = df_mct_total.loc[y_start:]
    fig.add_trace(go.Scatter(x=d.index, y=d["16th percentile"], mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"), row=1, col=2)
    fig.add_trace(go.Scatter(x=d.index, y=d["84th percentile"], mode="lines", line=dict(width=0), fill="tonexty", fillcolor="rgba(128,128,128,0.3)", name="MCT range", legend=L2), row=1, col=2)
    fig.add_trace(go.Scatter(x=d.index, y=d["Median"], name="MCT", line=dict(color="green"), legend=L2), row=1, col=2)
    d = pce.loc[y_start:]
    fig.add_trace(go.Scatter(x=d.index, y=d.values, name="PCE", line=dict(color="blue"), legend=L2), row=1, col=2)
    d = pce_core.loc[y_start:]
    fig.add_trace(go.Scatter(x=d.index, y=d.values, name="PCE Core", line=dict(color="orange"), legend=L2), row=1, col=2)
    add_fan(fig, 1, 2, L2, pi_fcst, "rgb(44,160,44)", "inflation", y_start)

    d = df_rGDP_potential_level.loc[y_start:]
    fig.add_trace(go.Scatter(x=d.index, y=d.values, name="rGDP potential", line=dict(color="blue"), legend=L3), row=2, col=1)
    add_fan(fig, 2, 1, L3, rGDP_potential_fcst, "rgb(31,119,180)", "potential GDP", y_start)
    d = rGDP_level.loc[y_start:, "GDPC1"]
    fig.add_trace(go.Scatter(x=d.index, y=d.values, name="rGDP", line=dict(color="green"), legend=L3), row=2, col=1)
    add_fan(fig, 2, 1, L3, rGDP_fcst, "rgb(44,160,44)", "GDP", y_start)

    d = ff.loc[y_start:]
    fig.add_trace(go.Scatter(x=d.index, y=d.values, name="Fed Funds (effective)", line=dict(color="blue"), legend=L4), row=2, col=2)
    add_fan(fig, 2, 2, L4, ff_fcst, "rgb(44,160,44)", "FF", y_start)
    d = fwd_daily.loc[y_start:pi_fcst.index.max()]
    fig.add_trace(go.Scatter(x=d.index, y=d.values, name="Treasury-implied forward", line=dict(color="black", dash="dash"), legend=L4), row=2, col=2)

    legend_style = dict(font=dict(size=10), bgcolor="rgba(255,255,255,0.6)", bordercolor="rgba(0,0,0,0.2)", borderwidth=1)
    fig.update_layout(
        height=1000, title_text="FedWatch Dashboard", hovermode="closest",
        legend=dict(x=0.0, y=1.0, xanchor="left", yanchor="top", **legend_style),
        legend2=dict(x=0.56, y=1.0, xanchor="left", yanchor="top", **legend_style),
        legend3=dict(x=0.0, y=0.42, xanchor="left", yanchor="top", **legend_style),
        legend4=dict(x=0.56, y=0.42, xanchor="left", yanchor="top", **legend_style),
    )
    FIGURES["main_dashboard"] = fig

    # =========================================================================
    # 11. Credit market context
    # =========================================================================
    gdp = web.DataReader("GDP", "fred", start_date, end_date).squeeze()
    gdp_d = gdp.resample("D").ffill()
    recession_bands = get_recession_bands(start_date, end_date)
    credit_ystart = "2000"

    household_total = web.DataReader("CMDEBT", "fred", start_date, end_date).squeeze() / 1000.0
    household_total.index = household_total.index + pd.offsets.QuarterEnd()
    household_mortgage = web.DataReader("HHMSDODNS", "fred", start_date, end_date).squeeze() / 1000.0
    household_mortgage.index = household_mortgage.index + pd.offsets.QuarterEnd()
    household_consumer = household_total - household_mortgage
    household_consumer_2 = web.DataReader("TOTALSL", "fred", start_date, end_date).squeeze() / 1000.0
    household_consumer_2.index = household_consumer_2.index + pd.offsets.MonthEnd()

    fig = go.Figure()
    for name, s, color, dash in [
        ("Total household debt", household_total, CAT[0], "solid"),
        ("Mortgage debt", household_mortgage, CAT[1], "solid"),
        ("Non-mortgage (consumer) debt", household_consumer, CAT[2], "solid"),
        ("Consumer credit, alt. source (G.19)", household_consumer_2, CAT[3], "dot"),
    ]:
        ratio = (s / gdp_d.reindex(s.index) * 100.0).loc[credit_ystart:]
        fig.add_trace(go.Scatter(x=ratio.index, y=ratio.values, name=name, line=dict(color=color, width=2, dash=dash)))
    add_recession_bands(fig, recession_bands, xmin=credit_ystart)
    FIGURES["household_debt"] = style_fig(fig, "Household Debt (% of GDP)", yaxis_title="% of GDP")

    household_dsr = web.DataReader("TDSP", "fred", start_date, end_date).squeeze()
    household_dsr.index = household_dsr.index + pd.offsets.QuarterEnd()
    household_sr = web.DataReader("PSAVERT", "fred", start_date, end_date).squeeze()
    household_sr.index = household_sr.index + pd.offsets.MonthEnd()

    fig = go.Figure()
    d = household_dsr.loc[credit_ystart:]
    fig.add_trace(go.Scatter(x=d.index, y=d.values, line=dict(color=CAT[0], width=2)))
    add_recession_bands(fig, recession_bands, xmin=credit_ystart)
    FIGURES["household_dsr"] = style_fig(fig, "Household Debt Service Ratio", yaxis_title="% of disposable income", legend=False, height=320)

    fig = go.Figure()
    d = household_sr.loc[credit_ystart:]
    fig.add_trace(go.Scatter(x=d.index, y=d.values, line=dict(color=CAT[0], width=2)))
    add_recession_bands(fig, recession_bands, xmin=credit_ystart)
    FIGURES["household_sr"] = style_fig(fig, "Personal Saving Rate", yaxis_title="% of disposable income", legend=False, height=320)

    nonfinancial_total = web.DataReader("TCMDO", "fred", start_date, end_date).squeeze() / 1000.0
    nonfinancial_total.index = nonfinancial_total.index + pd.offsets.QuarterEnd()
    fig = go.Figure()
    ratio = (nonfinancial_total / gdp_d.reindex(nonfinancial_total.index) * 100.0).loc[credit_ystart:]
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio.values, line=dict(color=CAT[0], width=2)))
    add_recession_bands(fig, recession_bands, xmin=credit_ystart)
    FIGURES["total_leverage"] = style_fig(fig, "Total Credit Market Debt, All Sectors", yaxis_title="% of GDP", legend=False, height=320)

    nonfinancial_corporate = web.DataReader("NCBDBIQ027S", "fred", start_date, end_date).squeeze() / 1000.0
    nonfinancial_corporate.index = nonfinancial_corporate.index + pd.offsets.QuarterEnd()
    cni_loans = web.DataReader("BUSLOANS", "fred", start_date, end_date).squeeze()
    cni_loans.index = cni_loans.index + pd.offsets.MonthEnd()
    corporate_profits = web.DataReader("A053RC1Q027SBEA", "fred", start_date, end_date).squeeze()
    corporate_profits.index = corporate_profits.index + pd.offsets.QuarterEnd()
    net_interest = web.DataReader("W255RC1Q027SBEA", "fred", start_date, end_date).squeeze()
    net_interest.index = net_interest.index + pd.offsets.QuarterEnd()
    corporate_dsr = net_interest / corporate_profits * 100.0

    fig = go.Figure()
    for name, s, color in [("Nonfinancial corporate debt", nonfinancial_corporate, CAT[0]), ("C&I loans (bank lending)", cni_loans, CAT[1])]:
        ratio = (s / gdp_d.reindex(s.index) * 100.0).loc[credit_ystart:]
        fig.add_trace(go.Scatter(x=ratio.index, y=ratio.values, name=name, line=dict(color=color, width=2)))
    add_recession_bands(fig, recession_bands, xmin=credit_ystart)
    FIGURES["corporate_debt"] = style_fig(fig, "Corporate & Business Debt (% of GDP)", yaxis_title="% of GDP")

    fig = go.Figure()
    d = corporate_dsr.loc[credit_ystart:]
    fig.add_trace(go.Scatter(x=d.index, y=d.values, line=dict(color=CAT[0], width=2)))
    add_recession_bands(fig, recession_bands, xmin=credit_ystart)
    FIGURES["corporate_dsr"] = style_fig(fig, "Corporate Net Interest Burden", yaxis_title="Net interest / corporate profits (%)", legend=False, height=320)

    federal = web.DataReader("GFDEBTN", "fred", start_date, end_date).squeeze() / 1000.0
    federal.index = federal.index + pd.offsets.QuarterEnd()
    federal_public = web.DataReader("FYGFDPUN", "fred", start_date, end_date).squeeze() / 1000.0
    federal_public.index = federal_public.index + pd.offsets.QuarterEnd()
    state = web.DataReader("SLGSDODNS", "fred", start_date, end_date).squeeze() / 1000.0
    state.index = state.index + pd.offsets.QuarterEnd()

    fig = go.Figure()
    for name, s, color in [("Federal debt, total", federal, CAT[0]), ("Federal debt, held by public", federal_public, CAT[1]), ("State & local debt", state, CAT[2])]:
        ratio = (s / gdp_d.reindex(s.index) * 100.0).loc[credit_ystart:]
        fig.add_trace(go.Scatter(x=ratio.index, y=ratio.values, name=name, line=dict(color=color, width=2)))
    add_recession_bands(fig, recession_bands, xmin=credit_ystart)
    FIGURES["government_debt"] = style_fig(fig, "Government Debt (% of GDP)", yaxis_title="% of GDP")

    gov_interest = web.DataReader("A180RC1Q027SBEA", "fred", start_date, end_date).squeeze()
    gov_interest.index = gov_interest.index + pd.offsets.QuarterEnd()
    gov_revenue = web.DataReader("W066RC1Q027SBEA", "fred", start_date, end_date).squeeze()
    gov_revenue.index = gov_revenue.index + pd.offsets.QuarterEnd()
    gov_deficit = web.DataReader("AD01RC1Q027SBEA", "fred", start_date, end_date).squeeze()
    gov_deficit.index = gov_deficit.index + pd.offsets.QuarterEnd()

    fig = go.Figure()
    ratio = (gov_interest / gdp_d.reindex(gov_interest.index) * 100.0).loc[credit_ystart:]
    fig.add_trace(go.Scatter(x=ratio.index, y=ratio.values, line=dict(color=CAT[0], width=2)))
    add_recession_bands(fig, recession_bands, xmin=credit_ystart)
    FIGURES["gov_interest_gdp"] = style_fig(fig, "Federal Interest Payments", yaxis_title="% of GDP", legend=False, height=320)

    fig = go.Figure()
    for name, s, color in [("Interest / revenue", (gov_interest / gov_revenue * 100.0), CAT[0]), ("Deficit / revenue", (-gov_deficit / gov_revenue * 100.0), CAT[1])]:
        d = s.loc[credit_ystart:]
        fig.add_trace(go.Scatter(x=d.index, y=d.values, name=name, line=dict(color=color, width=2)))
    add_recession_bands(fig, recession_bands, xmin=credit_ystart)
    FIGURES["gov_int_deficit"] = style_fig(fig, "Federal Interest & Deficit Burden", yaxis_title="% of federal revenue")

    fig = go.Figure()
    interest_to_gdp_growth = gov_interest.rolling(4).mean() / gdp.diff(4).resample("D").ffill().reindex(gov_interest.index)
    d = interest_to_gdp_growth.loc[credit_ystart:]
    fig.add_trace(go.Scatter(x=d.index, y=d.values, line=dict(color=CAT[0], width=2)))
    add_recession_bands(fig, recession_bands, xmin=credit_ystart)
    FIGURES["gov_int_vs_growth"] = style_fig(fig, "Federal Interest Payments vs. Nominal GDP Growth", yaxis_title="Interest / annual $ GDP growth (x)", legend=False, height=320)

    # =========================================================================
    # 12. Labor market context
    # =========================================================================
    labor_ystart = "2022"

    unemployment = web.DataReader("UNRATE", "fred", start_date, end_date).squeeze()
    unemployment.index = unemployment.index + pd.offsets.MonthEnd()
    fig = go.Figure()
    d = unemployment.loc[labor_ystart:]
    fig.add_trace(go.Scatter(x=d.index, y=d.values, line=dict(color=CAT[0], width=2)))
    add_recession_bands(fig, recession_bands, xmin=labor_ystart)
    FIGURES["labor_unemployment"] = style_fig(fig, "Unemployment Rate", yaxis_title="%", legend=False, height=320)

    laborpart = web.DataReader("CIVPART", "fred", start_date, end_date).squeeze()
    laborpart.index = laborpart.index + pd.offsets.MonthEnd()
    emp_pop = web.DataReader("EMRATIO", "fred", start_date, end_date).squeeze()
    emp_pop.index = emp_pop.index + pd.offsets.MonthEnd()
    fig = go.Figure()
    for name, s, color in [("Labor force participation", laborpart, CAT[0]), ("Employment-population ratio", emp_pop, CAT[1])]:
        d = s.loc[labor_ystart:]
        fig.add_trace(go.Scatter(x=d.index, y=d.values, name=name, line=dict(color=color, width=2)))
    add_recession_bands(fig, recession_bands, xmin=labor_ystart)
    FIGURES["labor_participation"] = style_fig(fig, "Labor Force Participation & Employment-Population Ratio", yaxis_title="%")

    nonfarm = web.DataReader("PAYEMS", "fred", start_date, end_date).squeeze().diff()
    nonfarm.index = nonfarm.index + pd.offsets.MonthEnd()
    fig = go.Figure()
    d = nonfarm.loc[labor_ystart:]
    colors = [STATUS_GOOD if v >= 0 else STATUS_CRITICAL for v in d.values]
    fig.add_trace(go.Bar(x=d.index, y=d.values, marker_color=colors, marker_line_width=0, showlegend=False))
    x0 = d.index[0].strftime("%Y-%m-%d")
    fig.add_trace(go.Scatter(x=[x0], y=[None], mode="markers", marker=dict(color=STATUS_GOOD, size=10), name="Gain"))
    fig.add_trace(go.Scatter(x=[x0], y=[None], mode="markers", marker=dict(color=STATUS_CRITICAL, size=10), name="Loss"))
    add_recession_bands(fig, recession_bands, xmin=labor_ystart)
    FIGURES["labor_payrolls"] = style_fig(fig, "Nonfarm Payrolls, Monthly Change", yaxis_title="Thousands of jobs")

    # ICSA is weekly -- unlike the other labor series above, do NOT shift its
    # index to MonthEnd: that collapses ~4 weekly observations per month onto
    # the same duplicate date, which renders as a zigzag. Keep its native
    # (already meaningful) week-ending date.
    claims = web.DataReader("ICSA", "fred", start_date, end_date).squeeze() / 1000.0
    fig = go.Figure()
    d = claims.loc[labor_ystart:]
    fig.add_trace(go.Scatter(x=d.index, y=d.values, line=dict(color=CAT[0], width=2)))
    add_recession_bands(fig, recession_bands, xmin=labor_ystart)
    FIGURES["labor_claims"] = style_fig(fig, "Initial Jobless Claims (weekly)", yaxis_title="Thousands", legend=False, height=320)

    jolts = web.DataReader("JTSJOL", "fred", start_date, end_date).squeeze()
    jolts.index = jolts.index + pd.offsets.MonthEnd()
    quitsrate = web.DataReader("JTSQUR", "fred", start_date, end_date).squeeze()
    quitsrate.index = quitsrate.index + pd.offsets.MonthEnd()
    fig = go.Figure()
    base_date = jolts.loc[labor_ystart:].index[0]
    for name, s, color in [("Job openings (JOLTS)", jolts, CAT[0]), ("Quits rate", quitsrate, CAT[1])]:
        d = s.loc[labor_ystart:]
        idx = d / d.loc[base_date] * 100.0
        fig.add_trace(go.Scatter(x=idx.index, y=idx.values, name=name, line=dict(color=color, width=2)))
    add_recession_bands(fig, recession_bands, xmin=labor_ystart)
    FIGURES["labor_demand"] = style_fig(fig, "Labor Demand", yaxis_title=f"Index ({base_date:%b %Y} = 100)")

    # =========================================================================
    # 13. FOMC data release grid
    # =========================================================================
    print("Building FOMC data grid...")
    ZSCORE_WINDOW_YEARS = 10

    def _period_zscore(change_series):
        change_series = change_series.dropna()
        cutoff = change_series.index.max() - pd.DateOffset(years=ZSCORE_WINDOW_YEARS)
        window = change_series.loc[change_series.index >= cutoff]
        mu, sigma = window.mean(), window.std()
        latest = change_series.iloc[-1]
        z = (latest - mu) / sigma if sigma and not np.isnan(sigma) else np.nan
        return latest, z

    def level_row(category, label, series_id, transform="level", fmt="{:.2f}", divisor=1.0):
        raw = web.DataReader(series_id, "fred", start_date, end_date).squeeze().dropna() / divisor
        if transform == "yoy":
            raw = (raw.pct_change(12) * 100.0).dropna()
        elif transform == "qoq_saar":
            raw = (raw.pct_change(1) * 4 * 100.0).dropna()
        change_series = raw.diff().dropna()
        latest_date = raw.index[-1]
        latest_val, prior_val = raw.iloc[-1], raw.iloc[-2]
        change, z = _period_zscore(change_series)
        return dict(Category=category, Series=label, SeriesID=series_id, LatestRelease=latest_date, LatestValue=fmt.format(latest_val), PriorValue=fmt.format(prior_val), Change=fmt.format(change), Zscore=z)

    def flow_row(category, label, series_id, transform="diff", fmt="{:.1f}"):
        raw = web.DataReader(series_id, "fred", start_date, end_date).squeeze().dropna()
        if transform == "diff":
            flow = raw.diff().dropna()
        elif transform == "pct_mom":
            flow = raw.pct_change(1).dropna() * 100.0
        latest_date = flow.index[-1]
        latest_val, prior_val = flow.iloc[-1], flow.iloc[-2]
        _, z = _period_zscore(flow)
        return dict(Category=category, Series=label, SeriesID=series_id, LatestRelease=latest_date, LatestValue=fmt.format(latest_val), PriorValue=fmt.format(prior_val), Change=fmt.format(latest_val), Zscore=z)

    rows = [
        level_row("Growth", "Real GDP (QoQ SAAR %)", "GDPC1", transform="qoq_saar"),
        flow_row("Growth", "Retail Sales (MoM %)", "RSAFS", transform="pct_mom"),
        flow_row("Growth", "Industrial Production (MoM %)", "INDPRO", transform="pct_mom"),
        flow_row("Growth", "Housing Starts (MoM %)", "HOUST", transform="pct_mom"),
        flow_row("Labor", "Nonfarm Payrolls (k)", "PAYEMS", transform="diff", fmt="{:.0f}"),
        level_row("Labor", "Unemployment Rate (%)", "UNRATE"),
        level_row("Labor", "Labor Force Participation (%)", "CIVPART"),
        level_row("Labor", "Initial Claims (k)", "ICSA", fmt="{:.0f}", divisor=1000.0),
        level_row("Labor", "Job Openings, JOLTS (k)", "JTSJOL", fmt="{:.0f}"),
        level_row("Labor", "Avg Hourly Earnings (YoY %)", "CES0500000003", transform="yoy"),
        level_row("Inflation", "CPI (YoY %)", "CPIAUCSL", transform="yoy"),
        level_row("Inflation", "Core CPI (YoY %)", "CPILFESL", transform="yoy"),
        level_row("Inflation", "PCE (YoY %)", "PCEPI", transform="yoy"),
        level_row("Inflation", "Core PCE (YoY %)", "PCEPILFE", transform="yoy"),
        level_row("Rates", "Fed Funds (Effective, %)", "DFF"),
        level_row("Rates", "10Y Treasury Yield (%)", "DGS10"),
        level_row("Rates", "10Y-2Y Spread (%)", "T10Y2Y"),
    ]
    df_fomc_grid = pd.DataFrame(rows)

    def _zscore_color(z, vmax=3.0):
        if pd.isna(z):
            return "rgba(255,255,255,1)"
        t = max(-1.0, min(1.0, z / vmax))
        if t >= 0:
            r, g, b = 255, int(255 - t * 155), int(255 - t * 155)
        else:
            t = -t
            r, g, b = int(255 - t * 155), int(255 - t * 155), 255
        return f"rgba({r},{g},{b},1)"

    grid = df_fomc_grid.copy()
    grid["LatestRelease"] = grid["LatestRelease"].dt.strftime("%Y-%m-%d")
    grid["ZscoreDisplay"] = grid["Zscore"].apply(lambda z: f"{z:+.2f}" if pd.notna(z) else "n/a")
    grid["Flag"] = grid["Zscore"].apply(lambda z: "outsized" if pd.notna(z) and abs(z) >= 2 else ("elevated" if pd.notna(z) and abs(z) >= 1.5 else ""))

    category_bg = {"Growth": "#eef4fb", "Labor": "#eefaf0", "Inflation": "#fdf3ec", "Rates": "#f5f0fb"}
    cat_col_colors = [category_bg.get(c, "white") for c in grid["Category"]]
    zscore_colors = [_zscore_color(z) for z in grid["Zscore"]]

    columns = ["Category", "Series", "LatestRelease", "PriorValue", "LatestValue", "Change", "ZscoreDisplay", "Flag"]
    headers = ["Category", "Series", "Latest Release", "Prior", "Latest", "Change", f"Z-score ({ZSCORE_WINDOW_YEARS}y)", "Flag"]
    cell_colors = [zscore_colors if col == "ZscoreDisplay" else cat_col_colors for col in columns]

    fig = go.Figure(data=[go.Table(
        header=dict(values=headers, fill_color="#2c3e50", font=dict(color="white", size=12), align="left"),
        cells=dict(values=[grid[c] for c in columns], fill_color=cell_colors, align="left", font=dict(size=11), height=26),
    )])
    fig.update_layout(title=f"FOMC-Relevant Data Releases ({ZSCORE_WINDOW_YEARS}y z-score of period-over-period change)", height=140 + 38 * len(grid))
    FIGURES["fomc_grid"] = fig

    # =========================================================================
    # 14. Save everything
    # =========================================================================
    print(f"Saving {len(FIGURES)} figures + grid + meta to {DATA_DIR}...")
    for key, f in FIGURES.items():
        pio.write_json(f, FIGURES_DIR / f"{key}.json")

    df_fomc_grid.to_csv(DATA_DIR / "fomc_grid.csv", index=False)

    outsized = df_fomc_grid["Zscore"].apply(lambda z: pd.notna(z) and abs(z) >= 2).sum()
    META.update(
        last_updated=dt.datetime.now(dt.timezone.utc).isoformat(),
        fcst_start_lw=fcst_start_lw.strftime("%Y-%m-%d"),
        last_actual_quarter=last_actual_quarter.strftime("%Y-%m-%d"),
        fcst_end=fcst_end.strftime("%Y-%m-%d"),
        treasury_asof_date=asof_date.strftime("%Y-%m-%d"),
        fed_funds_effective=round(float(ff.iloc[-1]), 2),
        unemployment_rate=round(float(unemployment.iloc[-1]), 2),
        core_pce_yoy=round(float(pce_core.iloc[-1]), 2),
        core_cpi_yoy=float(df_fomc_grid.loc[df_fomc_grid["Series"] == "Core CPI (YoY %)", "LatestValue"].iloc[0]),
        ff_model_median_1yr=round(float(pd.to_numeric(ff_fcst.iloc[min(4, len(ff_fcst) - 1)]).median()), 2) if len(ff_fcst) else None,
        outsized_release_count=int(outsized),
        num_fomc_grid_rows=len(df_fomc_grid),
    )
    with open(DATA_DIR / "meta.json", "w") as fh:
        json.dump(META, fh, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
