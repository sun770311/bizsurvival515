from __future__ import annotations

from pathlib import Path

import streamlit as st
import streamlit.components.v1 as components
from bizsurvival515.app.utils.ui_styles import apply_shared_styles

st.set_page_config(
    page_title="NYC Business Map",
    layout="wide",
)
apply_shared_styles()

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 0.8rem;
        padding-bottom: 0rem;
        max-width: 100%;
    }

    .glass-banner {
        margin-top: 12px;
        margin-bottom: 18px;
        padding: 14px 18px;
        border-radius: 14px;

        background: linear-gradient(
            135deg,
            rgba(127,255,212,0.30) 0%,
            rgba(255,255,140,0.25) 50%,
            rgba(11,102,35,0.30) 100%
        );

        border: 1px solid rgba(255,255,255,0.35);

        backdrop-filter: blur(18px);
        -webkit-backdrop-filter: blur(18px);

        color: white;
        font-size: 15px;
        font-weight: 500;

        box-shadow: 0 10px 25px rgba(0,0,0,0.18);
    }

    h1 {
        margin-top: 0.4rem;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.title("Business Landscape Explorer")
st.markdown(
    """
    <p style="font-size:16px; color:#6b7280; margin-top:-6px;">
    Visualize business locations, licensing history, and complaint activity.
    </p>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    """
    <div class="glass-banner">
        Want to estimate survival probability for a hypothetical business? 
        Go to the <b>Logistic Regression</b> or <b>Cox Models</b> page.
    </div>
    """,
    unsafe_allow_html=True,
)

APP_DIR = Path(__file__).resolve().parents[1]
TEMPLATE_DIR = APP_DIR / "map_templates"
ARTIFACT_DIR = APP_DIR / "artifacts"
GEOJSON_PATH = ARTIFACT_DIR / "geojson" / "businesses.geojson"

if "MAPBOX_PUBLIC_TOKEN" not in st.secrets:
    st.error("Missing MAPBOX_PUBLIC_TOKEN in Streamlit secrets.")
    st.stop()

if not GEOJSON_PATH.exists():
    st.error(f"GeoJSON file not found: {GEOJSON_PATH}")
    st.stop()

mapbox_token = st.secrets["MAPBOX_PUBLIC_TOKEN"]

index_html = (TEMPLATE_DIR / "index.html").read_text(encoding="utf-8")
app_js = (TEMPLATE_DIR / "app.js").read_text(encoding="utf-8")
geojson_text = GEOJSON_PATH.read_text(encoding="utf-8")

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