# Macro Dashboard

A multi-section Streamlit dashboard, deployed at
[fedwatch-dashboard](https://github.com/patrickmauel/fedwatch-dashboard) ->
Streamlit Community Cloud. Currently one live section, built to add more:

- **Rates & Macro** (live) -- GDP growth, inflation, the fed funds rate, and
  the Treasury term structure, built on the NY Fed's Laubach-Williams r*
  model, a fitted Taylor-rule policy reaction function, and a 1,000-path
  Monte Carlo simulation. Plus credit-market and labor-market context
  charts and a z-score-flagged grid of FOMC-relevant data releases. Deployable
  counterpart to the `FedWatch_v2.ipynb` notebook in `../new_jupyter/`.
- **Equities** -- placeholder, not built yet.
- **Currencies** -- placeholder, not built yet.

## How a section is structured

Each section is three independent pieces, so sections never touch each
other's code or data:

```
pipelines/<name>.py   pulls data + computes, writes to data/<name>/
pages/<name>.py        reads only data/<name>/, renders it -- no network calls
data/<name>/            figures/*.json (Plotly), meta.json, any CSVs
```

`streamlit_app.py` at the repo root is just the navigation router (`st.navigation`)
listing every page -- it's the file Streamlit Cloud is configured to run, and
stays put so adding a section never means touching deployment settings.

`.github/workflows/daily_refresh.yml` loops over every file in `pipelines/`
and commits whatever changed in `data/` -- a new pipeline script is picked
up automatically, nothing to add to the workflow.

## Adding a new section (e.g. Equities)

1. Write `pipelines/equities.py` following the pattern in
   `pipelines/rates_macro.py`: pull data, build Plotly figures into a
   `FIGURES` dict, `pio.write_json` each one to `data/equities/figures/`,
   write a `data/equities/meta.json`.
2. Write `pages/equities.py` following `pages/rates_macro.py`: load from
   `data/equities/`, render with `st.plotly_chart`. No `st.set_page_config`
   -- that's only ever called once, in `streamlit_app.py`.
3. In `streamlit_app.py`, the `equities` page is already registered in the
   nav list (it currently points at the placeholder) -- nothing to add
   there once the real page replaces the placeholder content.
4. Run `python pipelines/equities.py` once locally to populate
   `data/equities/`, commit, push. The daily workflow picks it up from then on.

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python pipelines/rates_macro.py     # populates data/rates_macro/ (~1 min)
streamlit run streamlit_app.py
```

## Deployment status & how it's wired

Already deployed -- GitHub Actions refreshes `data/` daily at 11:00 UTC
(adjust the `cron` line in `.github/workflows/daily_refresh.yml`, times are
UTC) and Streamlit Community Cloud auto-redeploys on every push to `main`.
No secrets to configure; the workflow uses GitHub's automatic `GITHUB_TOKEN`.

**If deploying a fresh copy of this repo elsewhere:**

```bash
git init && git add -A && git commit -m "Initial commit"
gh repo create <name> --private --source=. --remote=origin --push
```
(or skip `gh` -- create an empty repo at github.com/new, then
`git remote add origin <url> && git push -u origin main`)

Then on [share.streamlit.io](https://share.streamlit.io): sign in with
GitHub -> New app -> pick the repo, branch `main`, main file path
`streamlit_app.py`.

**Gotcha we hit standing this up**: Streamlit Cloud authenticates via a
GitHub **OAuth App** (not the newer GitHub App model -- check
`github.com/settings/applications`, not `.../installations`). If it was
first authorized without the private-repo scope, it can't see private repos
at all (fails as "repository doesn't exist," not a permissions error) --
easiest fix is making the repo public; the alternative is revoking
Streamlit's access at `github.com/settings/applications` and re-authorizing
so it prompts for the `repo` scope this time.

## Adjusting things

- **Refresh time**: edit the `cron` line in `.github/workflows/daily_refresh.yml`.
- **Add/remove a chart in an existing section**: the pipeline builds each
  chart into `FIGURES["some_key"]`; add/remove the matching `chart("some_key")`
  call in that section's `pages/<name>.py`.
- **FOMC grid series**: edit the `rows = [...]` list in
  `pipelines/rates_macro.py`'s FOMC grid section.
