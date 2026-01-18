from __future__ import annotations

import streamlit as st


def apply_base_styles() -> None:
    """Apply global CSS styles.

    Mobile-first note:
    - Streamlit is inherently responsive-ish, but wide column layouts can suck on phones.
    - We keep controls large and avoid huge tables.
    """

    st.set_page_config(
        page_title="PerryPicks 🕵️‍♂️",
        page_icon="🕵️‍♂️",
        layout="wide",
    )

    st.markdown(
        """
        <style>
          [data-testid="stSidebar"] { display: none !important; }
          [data-testid="stSidebarNav"] { display: none !important; }

          .block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; }

          .pp-card {
            border-radius: 18px;
            padding: 16px 18px;
            background: rgba(255,255,255,0.04);
            border: 1px solid rgba(255,255,255,0.10);
            box-shadow: 0 8px 26px rgba(0,0,0,0.15);
          }
          .pp-title { font-size: 26px; font-weight: 800; margin: 0 0 2px 0; }
          .pp-sub { opacity: 0.85; margin: 0 0 10px 0; }

          .pp-kpi {
            border-radius: 16px;
            padding: 12px 14px;
            background: rgba(0,0,0,0.18);
            border: 1px solid rgba(255,255,255,0.10);
          }
          .pp-muted { opacity: 0.8; font-size: 13px; }

          .stButton>button { border-radius: 14px; font-weight: 650; padding: 0.55rem 0.9rem; }
          .stTextInput>div>div>input { border-radius: 14px; }
          .stNumberInput>div>div>input { border-radius: 14px; }

          /* Make metrics a bit more compact on mobile */
          @media (max-width: 600px) {
            .pp-title { font-size: 22px; }
            .block-container { padding-left: 0.8rem; padding-right: 0.8rem; }
          }
        </style>
        """,
        unsafe_allow_html=True,
    )
