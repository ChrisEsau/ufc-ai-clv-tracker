from __future__ import annotations

import streamlit as st


def inject_styles() -> None:
    """Inject styles scoped to the new clean Model Setup workspace."""

    st.markdown(
        """
        <style>
        .model-setup-shell {
            border: 1px solid rgba(43,60,82,.95);
            border-radius: 10px;
            padding: 1rem;
            background: linear-gradient(180deg, rgba(17,31,49,.95), rgba(9,19,32,.98));
            box-shadow: 0 20px 42px rgba(0,0,0,.24);
        }
        .model-setup-title {
            color: #f5f7fb;
            font-size: 1.35rem;
            font-weight: 950;
            letter-spacing: -.03em;
            margin-bottom: .25rem;
        }
        .model-setup-subtitle {
            color: #dbe7f5;
            font-size: .9rem;
            margin-bottom: .75rem;
        }
        .model-setup-note {
            color: #9fb0c4;
            font-size: .78rem;
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
