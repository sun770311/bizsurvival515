from __future__ import annotations

import base64
from pathlib import Path

import streamlit as st

from utils.ui_styles import apply_shared_styles


st.set_page_config(
    page_title="NYC Business Survival",
    layout="wide",
    initial_sidebar_state="collapsed",
)
apply_shared_styles()

APP_DIR = Path(__file__).resolve().parent
VIDEO_PATH = APP_DIR / "assets" / "nyc_drone_shot.mp4"

if not VIDEO_PATH.exists():
    st.error(f"Missing video file: {VIDEO_PATH}")
    st.stop()

video_bytes = VIDEO_PATH.read_bytes()
video_b64 = base64.b64encode(video_bytes).decode("utf-8")

st.markdown(
    """
    <style>
    html, body, [class*="css"] {
        margin: 0 !important;
        padding: 0 !important;
    }

    body {
        overflow: hidden;
    }

    .stApp,
    [data-testid="stAppViewContainer"],
    [data-testid="stMain"] {
        background: transparent !important;
        margin: 0 !important;
        padding: 0 !important;
    }

    [data-testid="stHeader"] {
        background: transparent !important;
        height: 0 !important;
    }

    [data-testid="stToolbar"] {
        right: 1rem;
    }

    [data-testid="stDecoration"] {
        display: none !important;
    }

    [data-testid="stSidebar"] {
        background: rgba(20, 20, 20, 0.92);
    }

    .block-container {
        padding: 0 !important;
        max-width: 100% !important;
    }

    .landing-wrapper {
        position: fixed;
        inset: 0;
        width: 100vw;
        height: 100vh;
        overflow: hidden;
        margin: 0;
        padding: 0;
        z-index: 0;
    }

    .landing-video {
        position: absolute;
        inset: 0;
        width: 100%;
        height: 100%;
        object-fit: cover;
        z-index: 0;
    }

    .landing-overlay {
        position: absolute;
        inset: 0;
        background: rgba(0, 0, 0, 0.35);
        z-index: 1;
    }

    .landing-content {
        position: absolute;
        inset: 0;
        z-index: 2;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        color: white;
        flex-direction: column;
        padding: 2rem;
        gap: 1.5rem;
        pointer-events: none;
    }

    .landing-title {
        font-size: clamp(2.5rem, 6vw, 5rem);
        font-weight: 700;
        letter-spacing: 0.04em;
        margin: 0;
    }

    div[data-testid="stButton"] {
        position: fixed;
        top: 50%;
        left: 50%;
        transform: translate(-50%, calc(-50% + 5.5rem));
        z-index: 10;
        width: auto;
    }

    div[data-testid="stButton"] > button {
        border-radius: 999px;
        padding: 0.95rem 1.6rem;
        font-size: 1.05rem;
        font-weight: 600;
        color: white;
        background: rgba(255, 255, 255, 0.12);
        border: 1px solid rgba(255, 255, 255, 0.35);
        backdrop-filter: blur(8px);
    }

    div[data-testid="stButton"] > button:hover {
        border-color: rgba(255, 255, 255, 0.6);
        background: rgba(255, 255, 255, 0.2);
        color: white;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

st.markdown(
    f"""
    <div class="landing-wrapper">
        <video class="landing-video" autoplay muted loop playsinline>
            <source src="data:video/mp4;base64,{video_b64}" type="video/mp4">
        </video>
        <div class="landing-overlay"></div>
        <div class="landing-content">
            <div class="landing-title">NYC Business Survival</div>
        </div>
    </div>
    """,
    unsafe_allow_html=True,
)

clicked = st.button("Explore Now →")

if clicked:
    st.switch_page("pages/1_Map.py")