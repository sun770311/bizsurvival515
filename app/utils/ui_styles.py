"""
Shared UI styles for the NYC Business Survival project Streamlit app.
"""

import streamlit as st


def apply_shared_styles() -> None:
    """
    Apply the shared UI styles to the Streamlit app.
    """
    st.markdown(
        """
        <style>

        /* Glass sidebar */
        [data-testid="stSidebar"] {
            background: rgba(25, 25, 25, 0.22) !important;
            backdrop-filter: blur(16px) saturate(160%) !important;
            -webkit-backdrop-filter: blur(16px) saturate(160%) !important;
            border-right: 1px solid rgba(255,255,255,0.15) !important;
        }

        [data-testid="stSidebar"] > div:first-child {
            background: transparent !important;
        }

        /* Sidebar navigation links */
        [data-testid="stSidebarNav"] {
            background: transparent !important;
        }

        [data-testid="stSidebarNav"] * {
            color: white !important;
        }

        </style>
        """,
        unsafe_allow_html=True,
    )
