# FedWatch Dashboard

A macro dashboard for GDP growth, inflation, the fed funds rate, and the
Treasury term structure -- built on the NY Fed's Laubach-Williams r* model,
a fitted Taylor-rule policy reaction function, and a 1,000-path Monte Carlo
simulation. Includes credit-market and labor-market context charts and a
z-score-flagged grid of FOMC-relevant data releases.

This is the deployable counterpart to the `FedWatch_v2.ipynb` notebook in
`../new_jupyter/` -- same model, same charts, restructured to run
unattended:

- **`pipeline.py`** -- pulls all data (FRED, NY Fed, Cleveland Fed, U.S.
  Treasury), runs the model, and writes results to `data/` (one JSON per
  chart, a CSV of the FOMC grid, and `meta.json` with headline stats).
  Takes about a minute to run.
- **`streamlit_app.py`** -- reads `data/` and renders the dashboard. Never
  touches the network itself, so it loads fast and never trips over FRED
  being slow.
- **`.github/workflows/daily_refresh.yml`** -- runs `pipeline.py` once a
  day (11:00 UTC by default) and commits the refreshed `data/` folder back
  to the repo.

## Run it locally

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python pipeline.py          # populates data/ (~1 min)
streamlit run streamlit_app.py
```

## Deploy it (one-time setup)

### 1. Create the GitHub repo and push

From this directory:

```bash
git init                          # already done if you're reading this from a fresh clone
git add -A
git commit -m "Initial commit"
```

Then create the repo on GitHub (either the web UI at github.com/new, or
the `gh` CLI if you have it installed):

```bash
gh repo create fedwatch-dashboard --private --source=. --remote=origin --push
```

Or without `gh`: create an empty repo named `fedwatch-dashboard` at
github.com/new (don't initialize it with a README), then:

```bash
git remote add origin https://github.com/<your-username>/fedwatch-dashboard.git
git branch -M main
git push -u origin main
```

### 2. Let the daily job run once

The workflow runs automatically on the schedule, but for the first deploy
trigger it manually so `data/` gets populated right away: go to the repo's
**Actions** tab -> **Daily FedWatch refresh** -> **Run workflow**. Takes
about a minute; it'll commit a `data/` folder back to `main` when done.

(No secrets to configure -- the workflow uses GitHub's automatic
`GITHUB_TOKEN`, which already has permission to push to the repo it runs in.)

### 3. Connect Streamlit Community Cloud

1. Go to [share.streamlit.io](https://share.streamlit.io) and sign in with
   GitHub.
2. **New app** -> pick the `fedwatch-dashboard` repo, branch `main`, main
   file path `streamlit_app.py`.
3. Deploy. Streamlit auto-redeploys on every push to `main` -- so each
   morning's data-refresh commit from step 2 automatically updates the
   live dashboard within a minute or two.

That's it -- from then on the whole thing runs unattended: GitHub Actions
refreshes the data daily, commits it, and Streamlit Cloud picks up the new
commit and redeploys.

## Adjusting things

- **Refresh time**: edit the `cron` line in
  `.github/workflows/daily_refresh.yml` (times are UTC).
- **Add/remove a chart**: `pipeline.py` builds each chart into `FIGURES["some_key"]`;
  add a `chart("some_key")` call in `streamlit_app.py` to display it.
- **FOMC grid series**: edit the `rows = [...]` list in `pipeline.py`'s FOMC
  grid section.
