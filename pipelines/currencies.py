"""
Currencies section pipeline: Holston-Laubach-Williams (HLW) multi-country
natural-rate model (US / Euro Area / Canada), FX forecasts from interest-rate
differentials, and trade-balance context.

Pulls live data (NY Fed HLW estimates, FRED, Eurostat), runs a 1,000-path
Monte Carlo simulation per country, and writes:

  data/currencies/figures/*.json  -- one Plotly figure per chart
  data/currencies/meta.json       -- last-updated timestamp + headline stats

Run standalone: `python pipelines/currencies.py` (from the repo root).
Ported from examples/HLW_Currency.ipynb -- see that notebook for the
original matplotlib/Gmail version. Two real bugs were found and fixed
while porting (see comments at their fix sites):

  1. Eurostat's EU trade dataset row keys changed from EA20 to EA21
     (Eurozone gained a member, changing the aggregate code) -- the
     notebook's hardcoded 'EXT_EA20' keys no longer exist.
  2. The FX-forecast anchor date is always a HLW quarter-start date
     (Jan/Apr/Jul/Oct 1st) -- Jan 1st is always a market holiday, so
     looking up that exact date in the daily FX series returned NaN,
     which propagated through the whole recursive forecast, making it
     entirely empty whenever the anchor happened to be a Q1 date (as it
     currently is). Fixed by forward-filling before the lookup.

Known limitation carried over from the original notebook (not fixed here):
the HLW model's own last published quarter typically lags the current
quarter by one quarter (same characteristic documented for the domestic
model in pipelines/rates_macro.py), so the simulation's first quarter is
sometimes already-known rather than genuinely future. Fixing this properly
would need each country's real-time GDP/inflation/policy-rate data sourced
under HLW's own precise definitions (which differ by country and aren't
safely inferable from generic FRED series) -- flagged rather than
silently worked around.
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

DATA_DIR = Path(__file__).resolve().parent.parent / "data" / "currencies"
FIGURES_DIR = DATA_DIR / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

FIGURES = {}
META = {}

COUNTRIES = ["US", "Euro Area", "Canada"]

# --- shared chart styling (same palette/chrome as pipelines/rates_macro.py;
# duplicated rather than imported so sections stay fully independent) ---
CAT = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#4a3aa7", "#e34948"]
INK_PRIMARY, INK_SECONDARY, INK_MUTED = "#0b0b0b", "#52514e", "#898781"
GRIDLINE, BASELINE, SURFACE = "#e1e0d9", "#c3c2b7", "#fcfcfb"
RECESSION_FILL = "rgba(137,135,129,0.16)"
COUNTRY_COLOR = {"US": CAT[0], "Euro Area": CAT[1], "Canada": CAT[2]}


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


def add_fan(fig, df_fcst, color, name, y_start=None):
    d = df_fcst.loc[y_start:] if y_start else df_fcst
    lo = pd.to_numeric(d.quantile(0.16, axis=1))
    hi = pd.to_numeric(d.quantile(0.84, axis=1))
    mean = pd.to_numeric(d.mean(axis=1))
    fig.add_trace(go.Scatter(x=lo.index, y=lo.values, mode="lines", line=dict(width=0), showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=hi.index, y=hi.values, mode="lines", line=dict(width=0), fill="tonexty", fillcolor=color.replace("rgb", "rgba").replace(")", ",0.25)"), name=f"{name} 1-sigma", hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=mean.index, y=mean.values, mode="lines", line=dict(color=color, dash="dash"), name=f"{name} mean"))


def run_hlw(country, df_combined, df_parameters, end_date, nsims, rng):
    """Simulate the HLW state-space system forward for one country. Ported
    verbatim from HLW_Currency.ipynb's run_hlw (only np.random.normal ->
    rng.normal, for a seedable/non-global RNG)."""
    df_sub = df_combined.loc[:, country].copy()
    df_sub.loc[:, "r"] = df_sub.loc[:, "interest"] - df_sub.loc[:, "inflation.expectations"]

    df_sub.loc[:, "interest_1"] = df_sub.loc[:, "interest"].shift(1)
    df_sub.loc[:, "inflation.expectations_1"] = df_sub.loc[:, "inflation.expectations"].shift(1)
    df_sub.loc[:, "r_1"] = df_sub.loc[:, "interest_1"] - df_sub.loc[:, "inflation.expectations_1"]

    df_sub.loc[:, "ybar_1"] = df_sub.loc[:, "ybar"].shift(1)
    df_sub.loc[:, "rstar_1"] = df_sub.loc[:, "rstar"].shift(1)
    df_sub.loc[:, "pibar"] = df_sub.loc[:, "inflation"] - 2.0

    df_sub.loc[:, "inflation_1"] = df_sub.loc[:, "inflation"].shift(1)
    df_sub.loc[:, "inflation_2"] = df_sub.loc[:, "inflation"].shift(2)
    df_sub.loc[:, "inflation_3"] = df_sub.loc[:, "inflation"].shift(3)

    df_param_sub = df_parameters.loc[:, country]
    df_template = df_sub.tail(1).reindex(pd.date_range(df_sub.tail(1).index[0], end_date, freq="QS"))

    coeff_set = 0.75
    df_sub.loc[:, "interest_delta"] = df_sub.loc[:, "interest"] - coeff_set * df_sub.loc[:, "interest_1"]
    fit_policymodel = smf.ols("interest_delta ~ rstar + pibar + ybar -1", data=df_sub).fit()

    dict_loops = {}
    for n in range(nsims):
        loop = df_template.copy()
        for i in range(len(loop.index) - 1):
            t_next, t_curr = loop.index[i + 1], loop.index[i]

            loop.loc[t_next, "g"] = loop.loc[t_curr, "g"] + rng.normal() * df_param_sub.loc["sigma_g"]
            loop.loc[t_next, "ystar"] = loop.loc[t_curr, "ystar"] + loop.loc[t_curr, "g"] / 400 + rng.normal() * df_param_sub.loc["sigma_y*"] / 100.0
            loop.loc[t_next, "z"] = loop.loc[t_curr, "z"] + rng.normal() * df_param_sub.loc["sigma_z"]
            loop.loc[t_next, "rstar"] = df_param_sub["c"] * loop.loc[t_next, "g"] + loop.loc[t_next, "z"]

            loop.loc[t_next, "ybar"] = (
                df_param_sub["a_y,1"] * loop.loc[t_curr, "ybar"]
                + df_param_sub["a_y,2"] * loop.loc[t_curr, "ybar_1"]
                + df_param_sub["a_r"] / 2.0 * (loop.loc[t_curr, "r"] - loop.loc[t_curr, "rstar"])
                + df_param_sub["a_r"] / 2.0 * (loop.loc[t_curr, "r_1"] - loop.loc[t_curr, "rstar_1"])
                + rng.normal() * df_param_sub.loc["sigma_y~"]
            )
            loop.loc[t_next, "inflation"] = (
                df_param_sub["b_pi"] * loop.loc[t_curr, "inflation"]
                + (1.0 - df_param_sub["b_pi"]) * (loop.loc[t_curr, "inflation_1"] + loop.loc[t_curr, "inflation_3"]) / 2.0
                + df_param_sub["b_y"] * loop.loc[t_curr, "ybar"]
                + rng.normal() * df_param_sub.loc["sigma_pi"]
            )

            loop.loc[t_next, "inflation_1"] = loop.loc[t_curr, "inflation"]
            loop.loc[t_next, "inflation_2"] = loop.loc[t_curr, "inflation_1"]
            loop.loc[t_next, "inflation_3"] = loop.loc[t_curr, "inflation_2"]

            loop.loc[t_next, "inflation.expectations"] = (
                df_param_sub["b_pi"] * loop.loc[t_next, "inflation"]
                + (1.0 - df_param_sub["b_pi"]) * (loop.loc[t_next, "inflation_1"] + loop.loc[t_next, "inflation_3"]) / 2.0
                + df_param_sub["b_y"] * loop.loc[t_next, "ybar"]
            )

            loop.loc[t_next, "pibar"] = loop.loc[t_next, "inflation"] - 2.0
            loop.loc[t_next, "interest"] = (
                fit_policymodel.params["rstar"] * loop.loc[t_next, "rstar"]
                + fit_policymodel.params["pibar"] * loop.loc[t_next, "pibar"]
                + fit_policymodel.params["ybar"] * loop.loc[t_next, "ybar"]
                + coeff_set * loop.loc[t_curr, "interest"]
            )
            loop.loc[t_next, "interest"] = max(loop.loc[t_next, "interest"], 0.0)
            loop.loc[t_next, "gdp.log"] = loop.loc[t_next, "ystar"] + loop.loc[t_next, "ybar"] / 100.0

            loop.loc[t_next, "interest_1"] = loop.loc[t_curr, "interest"]
            loop.loc[t_next, "r_1"] = loop.loc[t_curr, "r"]
            loop.loc[t_next, "rstar_1"] = loop.loc[t_curr, "rstar"]
            loop.loc[t_next, "inflation.expectations_1"] = loop.loc[t_curr, "inflation.expectations"]
            loop.loc[t_next, "ybar_1"] = loop.loc[t_curr, "ybar"]

            loop.loc[t_next, "r"] = loop.loc[t_next, "interest"] - loop.loc[t_next, "inflation.expectations"]
        dict_loops[n] = loop

    return pd.concat(dict_loops, axis=1).swaplevel(0, 1, axis=1).sort_index(axis=1)


def create_currencyfcsts(country_cross, currency, df_combined, df_countries):
    """FX forecast from the interest-rate differential implied by the two
    countries' simulated policy-rate paths (partial-adjustment regression,
    same style as the domestic Taylor-rule fit)."""
    df_relint_fcst = (1 + df_countries.loc[:, (country_cross, "interest")] / 100.0) / (1 + df_countries.loc[:, ("US", "interest")] / 100.0)
    df_relint_hist = (1 + df_combined.loc[:, (country_cross, "interest")] / 100.0) / (1 + df_combined.loc[:, ("US", "interest")] / 100.0)

    start = df_combined.loc[:, (country_cross, "interest")].dropna().index.min().strftime("%Y-%m-%d")
    end = df_combined.loc[:, (country_cross, "interest")].dropna().index.max().strftime("%Y-%m-%d")
    fx = web.DataReader(currency, "fred", start, end)

    if currency == "DEXCAUS":
        fx = 1.0 / fx  # normalize to USD-per-foreign-currency-unit, matching DEXUSEU's convention

    df_train = pd.concat({"rel_int": df_relint_hist, "currency": fx.resample("QS").mean().loc[:, currency]}, axis=1).dropna()
    df_train.loc[:, "currency1"] = df_train.loc[:, "currency"].shift(1)

    coeff = 0.5
    df_train.loc[:, "currencydiff"] = df_train.loc[:, "currency"] - coeff * df_train.loc[:, "currency1"]
    fit_fx = smf.ols("currencydiff ~ rel_int", data=df_train.loc["2005":]).fit()

    df_relint_fcst = df_relint_fcst.reindex(pd.date_range(df_relint_hist.tail(5).index.max(), df_relint_fcst.index.max(), freq="QS")).copy()
    for c in df_relint_fcst.columns:
        df_relint_fcst.loc[:, c] = df_relint_hist.tail(1).combine_first(df_relint_fcst.loc[:, c])

    df_modeled = df_relint_fcst.copy()
    df_modeled.loc[:, :] = np.nan
    for c in df_modeled.columns:
        df_modeled.loc[:, c] = fit_fx.predict(df_relint_fcst.loc[:, [c]].rename(columns={c: "rel_int"}))

    # BUG FIX: the anchor date `end` is always a HLW quarter-start date
    # (Jan/Apr/Jul/Oct 1) -- Jan 1 is always a market holiday, so `fx.loc[end]`
    # returns NaN on any Q1 anchor (which then poisons the whole recursive
    # forecast below via +=). Forward-fill first so a holiday resolves to the
    # last actual trading day's rate.
    df_modeled.loc[pd.to_datetime(end), :] = fx.ffill().loc[end].values[0]

    for i in range(len(df_modeled.index) - 1):
        t_next, t_curr = df_modeled.index[i + 1], df_modeled.index[i]
        df_modeled.loc[t_next, :] = df_modeled.loc[t_next, :] + coeff * df_modeled.loc[t_curr, :]

    return fx, df_modeled


def main():
    start_date = "1990-01-01"
    end_date = dt.date.today().strftime("%Y-%m-%d")

    # =========================================================================
    # 1. Load HLW multi-country data
    # =========================================================================
    print("Pulling Holston-Laubach-Williams multi-country estimates...")
    hlw_url = "https://www.newyorkfed.org/medialibrary/media/research/economists/williams/data/Holston_Laubach_Williams_current_estimates.xlsx"
    df_estimates = pd.read_excel(hlw_url, sheet_name="HLW Estimates", index_col=0, header=[4, 5])
    df_estimates = df_estimates.loc[:, df_estimates.columns[~(df_estimates.columns.get_level_values(1).isin(["Euro Area.1", "Date"]))]]
    df_estimates.columns = pd.MultiIndex.from_tuples([
        ({"Trend Growth (g), Annualized": "g", "Other Determinants (z)": "z", "Natural Rate (r*)": "rstar", "Output Gap": "ybar"}[c[0]], c[1])
        for c in df_estimates.columns
    ])
    df_estimates = df_estimates.swaplevel(0, 1, axis=1).sort_index(axis=1)

    df_inputs = pd.concat({
        "US": pd.read_excel(hlw_url, sheet_name="US input data", index_col=0),
        "Euro Area": pd.read_excel(hlw_url, sheet_name="EA input data", index_col=0),
        "Canada": pd.read_excel(hlw_url, sheet_name="CA input data", index_col=0),
    }, axis=1)

    df_combined = pd.concat([df_inputs, df_estimates], axis=1).sort_index(axis=1)
    # NB: 'Canada ' has a trailing space in the source spreadsheet's own column
    # label -- matched verbatim by the rename dict, same quirk pattern as the
    # domestic model's 'b_3 '/'sigma_3 ' parameters.
    df_parameters = pd.read_excel(hlw_url, sheet_name="Parameters", index_col=0, header=[2, 3]).dropna().droplevel(1, axis=1).rename(columns={"United States": "US", "Canada ": "Canada"})

    df_potential = pd.concat(
        {"ystar": (df_combined.loc[:, pd.IndexSlice[:, "gdp.log"]].droplevel(1, axis=1) - df_combined.loc[:, pd.IndexSlice[:, "ybar"]].droplevel(1, axis=1) / 100.0)},
        axis=1,
    ).swaplevel(0, 1, axis=1)
    df_combined = pd.concat([df_combined, df_potential], axis=1).sort_index(axis=1)

    # =========================================================================
    # 2. Monte Carlo simulate each country forward
    # =========================================================================
    anchor = df_combined.loc[:, ("US", "interest")].dropna().index.max()
    fcst_end = anchor + pd.DateOffset(years=3)
    print(f"HLW anchor quarter: {anchor:%Y-%m-%d}  ->  simulating to {fcst_end:%Y-%m-%d}")

    nsims = 1000
    rng = np.random.default_rng()
    dict_countries = {}
    for country in COUNTRIES:
        print(f"Simulating {country}...")
        dict_countries[country] = run_hlw(country, df_combined, df_parameters, fcst_end, nsims, rng)
    df_countries = pd.concat(dict_countries, axis=1)

    recession_bands = get_recession_bands(start_date, end_date)
    view_ystart = "2015"

    # =========================================================================
    # 3. Per-country charts: GDP growth, inflation, policy rate
    # =========================================================================
    for country in COUNTRIES:
        color = COUNTRY_COLOR[country]
        key_prefix = country.lower().replace(" ", "_")

        potential_growth_hist = (np.exp(df_combined.loc[:, (country, "ystar")]).pct_change(4) * 100).loc[view_ystart:]
        actual_growth_hist = (np.exp(df_combined.loc[:, (country, "gdp.log")]).pct_change(4) * 100).loc[view_ystart:]

        potential_growth_fcst = np.exp(df_countries.loc[:, (country, "ystar")]).reindex(
            pd.date_range(df_combined.index.min(), df_countries.index.max(), freq="QS")
        )
        for c in potential_growth_fcst.columns:
            potential_growth_fcst.loc[:, c] = np.exp(df_combined.loc[:, (country, "ystar")]).combine_first(potential_growth_fcst.loc[:, c])
        potential_growth_fcst = (potential_growth_fcst.pct_change(4) * 100).loc[view_ystart:]

        actual_growth_fcst = np.exp(df_countries.loc[:, (country, "gdp.log")]).reindex(
            pd.date_range(df_combined.index.min(), df_countries.index.max(), freq="QS")
        )
        for c in actual_growth_fcst.columns:
            actual_growth_fcst.loc[:, c] = np.exp(df_combined.loc[:, (country, "gdp.log")]).combine_first(actual_growth_fcst.loc[:, c])
        actual_growth_fcst = (actual_growth_fcst.pct_change(4) * 100).loc[view_ystart:]

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=potential_growth_hist.index, y=potential_growth_hist.values, name="Potential GDP growth", line=dict(color=CAT[0], width=2)))
        add_fan(fig, potential_growth_fcst, CAT[0], "Potential")
        fig.add_trace(go.Scatter(x=actual_growth_hist.index, y=actual_growth_hist.values, name="Actual GDP growth", line=dict(color=CAT[2], width=2)))
        add_fan(fig, actual_growth_fcst, "rgb(27,175,122)", "Actual")
        add_recession_bands(fig, recession_bands, xmin=view_ystart)
        FIGURES[f"{key_prefix}_gdp"] = style_fig(fig, f"{country}: GDP Growth, Actual vs. Potential", yaxis_title="YoY %")

        infl_hist = df_combined.loc[view_ystart:, (country, "inflation")]
        infl_fcst = df_countries.loc[:, (country, "inflation")]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=infl_hist.index, y=infl_hist.values, line=dict(color=color, width=2)))
        add_fan(fig, infl_fcst, color, "Inflation", y_start=view_ystart)
        add_recession_bands(fig, recession_bands, xmin=view_ystart)
        FIGURES[f"{key_prefix}_inflation"] = style_fig(fig, f"{country}: Inflation", yaxis_title="%", legend=False)

        rate_hist = df_combined.loc[view_ystart:, (country, "interest")]
        rate_fcst = df_countries.loc[:, (country, "interest")]
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=rate_hist.index, y=rate_hist.values, line=dict(color=color, width=2)))
        add_fan(fig, rate_fcst, color, "Policy rate", y_start=view_ystart)
        add_recession_bands(fig, recession_bands, xmin=view_ystart)
        FIGURES[f"{key_prefix}_rate"] = style_fig(fig, f"{country}: Policy Rate", yaxis_title="%", legend=False)

    # =========================================================================
    # 4. Cross-country policy rate comparison
    # =========================================================================
    fig = go.Figure()
    for country in COUNTRIES:
        color = COUNTRY_COLOR[country]
        hist = df_combined.loc[view_ystart:, (country, "interest")]
        fcst_mean = pd.to_numeric(df_countries.loc[:, (country, "interest")].mean(axis=1))
        fig.add_trace(go.Scatter(x=hist.index, y=hist.values, name=f"{country} (actual)", line=dict(color=color, width=2)))
        fig.add_trace(go.Scatter(x=fcst_mean.index, y=fcst_mean.values, name=f"{country} (model mean)", line=dict(color=color, width=2, dash="dash")))
    add_recession_bands(fig, recession_bands, xmin=view_ystart)
    FIGURES["rates_comparison"] = style_fig(fig, "Policy Rates: US vs. Euro Area vs. Canada", yaxis_title="%", height=440)

    # =========================================================================
    # 5. FX forecasts
    # =========================================================================
    print("Forecasting FX pairs...")
    eurusd, eurusd_modeled = create_currencyfcsts("Euro Area", "DEXUSEU", df_combined, df_countries)
    cadusd, cadusd_modeled = create_currencyfcsts("Canada", "DEXCAUS", df_combined, df_countries)

    for name, hist, hist_col, modeled, title in [
        ("eurusd", eurusd, "DEXUSEU", eurusd_modeled, "EUR/USD"),
        ("cadusd", cadusd, "DEXCAUS", cadusd_modeled, "CAD/USD (USD per CAD)"),
    ]:
        d = hist.loc[view_ystart:, hist_col].dropna()
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=d.index, y=d.values, line=dict(color=CAT[0], width=2)))
        add_fan(fig, modeled, CAT[0], title, y_start=view_ystart)
        add_recession_bands(fig, recession_bands, xmin=view_ystart)
        FIGURES[f"fx_{name}"] = style_fig(fig, f"{title} Forecast (from rate differential)", yaxis_title=title, legend=False)

    # =========================================================================
    # 6. Trade balance (% of GDP, comparable scale so combinable on one chart)
    # =========================================================================
    print("Pulling trade balance data...")
    us_exports = web.DataReader("BOPTEXP", "fred", start_date, end_date).squeeze()
    us_imports = web.DataReader("BOPTIMP", "fred", start_date, end_date).squeeze()
    us_nex = us_exports - us_imports  # millions of USD, monthly

    # BUG FIX: the Eurostat aggregate code changed from EA20 to EA21 (the
    # Eurozone gained a member) -- the notebook's hardcoded 'EXT_EA20' row
    # keys no longer exist in the dataset; using the current 'EXT_EA21' keys.
    eurostat_url = "https://ec.europa.eu/eurostat/api/dissemination/sdmx/2.1/data/ext_st_eabec?format=TSV"
    df_raw = pd.read_csv(eurostat_url, sep="\t", index_col=[0])
    df_raw = df_raw.loc[
        ["M,EXP,TRD_VAL,EXT_EA21,TOTAL,EA21", "M,IMP,TRD_VAL,EXT_EA21,TOTAL,EA21"], :
    ].T.rename(columns={"M,EXP,TRD_VAL,EXT_EA21,TOTAL,EA21": "Exports", "M,IMP,TRD_VAL,EXT_EA21,TOTAL,EA21": "Imports"})
    df_raw.index = pd.to_datetime(df_raw.index.str.strip(), format="%Y-%m")
    df_raw.index.names = ["date"]
    df_raw = df_raw.astype(float)
    eu_nex = df_raw.loc[:, "Exports"] - df_raw.loc[:, "Imports"]  # millions of EUR, monthly

    us_gdp = web.DataReader("GDP", "fred", start_date, end_date).squeeze()
    us_gdp_m = us_gdp.resample("QE").mean().resample("ME").interpolate().resample("MS").mean() / 4 / 3  # quarterly $bn -> monthly $bn

    eu_gdp = web.DataReader("EUNNGDP", "fred", start_date, end_date).squeeze()
    eu_gdp_m = eu_gdp.resample("QE").mean().resample("ME").interpolate().resample("MS").mean() / 3  # quarterly EUR bn -> monthly EUR bn

    # NB: BOPTEXP/BOPTIMP (US trade) are in millions of USD but FRED's GDP
    # series is in billions, hence /1000 below -- but EUNNGDP (Euro Area GDP)
    # and Eurostat's TRD_VAL (EU trade) are BOTH already in millions of EUR,
    # so no conversion is needed on that side (dividing eu_nex by 1000 too
    # would silently make the ratio ~1000x too small, i.e. an invisible flat
    # line at ~0 instead of the real ~2-4% surplus).
    us_nex_gdp = (us_nex / 1000.0) / us_gdp_m.reindex(pd.date_range(us_gdp_m.index.min(), us_nex.index.max(), freq="MS")).ffill() * 100.0
    eu_nex_gdp = eu_nex / eu_gdp_m.reindex(pd.date_range(eu_gdp_m.index.min(), eu_nex.index.max(), freq="MS")).ffill() * 100.0

    fig = go.Figure()
    for name, s, color in [("US", us_nex_gdp, CAT[0]), ("Euro Area", eu_nex_gdp, CAT[1])]:
        d = s.loc[view_ystart:]
        fig.add_trace(go.Scatter(x=d.index, y=d.values, name=name, line=dict(color=color, width=2)))
    add_recession_bands(fig, recession_bands, xmin=view_ystart)
    FIGURES["trade_balance"] = style_fig(fig, "Net Exports (% of GDP): US vs. Euro Area", yaxis_title="% of GDP")

    # =========================================================================
    # 7. Save everything
    # =========================================================================
    print(f"Saving {len(FIGURES)} figures + meta to {DATA_DIR}...")
    for key, f in FIGURES.items():
        pio.write_json(f, FIGURES_DIR / f"{key}.json")

    META.update(
        last_updated=dt.datetime.now(dt.timezone.utc).isoformat(),
        hlw_anchor_quarter=anchor.strftime("%Y-%m-%d"),
        fcst_end=fcst_end.strftime("%Y-%m-%d"),
        us_policy_rate=round(float(df_combined.loc[:, ("US", "interest")].dropna().iloc[-1]), 2),
        ea_policy_rate=round(float(df_combined.loc[:, ("Euro Area", "interest")].dropna().iloc[-1]), 2),
        ca_policy_rate=round(float(df_combined.loc[:, ("Canada", "interest")].dropna().iloc[-1]), 2),
        eurusd_spot=round(float(eurusd["DEXUSEU"].ffill().iloc[-1]), 4),
        cadusd_spot=round(float(cadusd["DEXCAUS"].ffill().iloc[-1]), 4),
    )
    with open(DATA_DIR / "meta.json", "w") as fh:
        json.dump(META, fh, indent=2)

    print("Done.")


if __name__ == "__main__":
    main()
