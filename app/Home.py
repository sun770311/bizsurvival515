from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components


st.set_page_config(page_title="NYC Business Map", layout="wide")

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.8rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }

    h1 {
        margin-top: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("NYC Business Map")
st.caption("Interactive business map embedded in Streamlit using Mapbox GL JS.")

# Base app directory
APP_DIR = Path(__file__).resolve().parent

# Template directory (HTML + JS)
TEMPLATE_DIR = APP_DIR / "map_templates"

# Artifact directory (geojson, models, etc.)
ARTIFACT_DIR = APP_DIR / "artifacts"

# GeoJSON location
GEOJSON_PATH = ARTIFACT_DIR / "geojson" / "businesses.geojson"

# ---- checks ----

if "MAPBOX_PUBLIC_TOKEN" not in st.secrets:
    st.error("Missing MAPBOX_PUBLIC_TOKEN in Streamlit secrets.")
    st.stop()

if not GEOJSON_PATH.exists():
    st.error(f"GeoJSON file not found: {GEOJSON_PATH}")
    st.stop()

# ---- load assets ----

mapbox_token = st.secrets["MAPBOX_PUBLIC_TOKEN"]

index_html = (TEMPLATE_DIR / "index.html").read_text(encoding="utf-8")
app_js = (TEMPLATE_DIR / "app.js").read_text(encoding="utf-8")
geojson_text = GEOJSON_PATH.read_text(encoding="utf-8")

# Inject token + GeoJSON
html = index_html.replace("YOUR_MAPBOX_PUBLIC_TOKEN", mapbox_token)

html = html.replace(
    '<script src="/app.js"></script>',
    f"""
    <script>
      const EMBEDDED_GEOJSON = {geojson_text};
    </script>
    <script>
      {app_js}
    </script>
    """,
)

components.html(html, height=550, scrolling=False)

st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
