"""
Dashboard entry point / navigation router.

This stays at the repo root under this exact filename because it's what
Streamlit Community Cloud is configured to run -- adding a new section
means adding a page below and an entry in the `pages` list, not touching
any deployment settings.

Each section is self-contained: pages/<name>.py renders it, reading only
from data/<name>/, which pipelines/<name>.py populates (see
pipelines/rates_macro.py for the pattern). Wire a new pipeline into
.github/workflows/daily_refresh.yml alongside the existing one so it also
refreshes daily.
"""
import streamlit as st

st.set_page_config(page_title="Macro Dashboard", page_icon="\U0001F4C8", layout="wide")

rates_macro = st.Page("pages/rates_macro.py", title="Rates & Macro", icon="\U0001F4C8", default=True)
equities = st.Page("pages/equities.py", title="Equities", icon="\U0001F4C9")
currencies = st.Page("pages/currencies.py", title="Currencies", icon="\U0001F4B1")

pg = st.navigation([rates_macro, equities, currencies])
pg.run()
