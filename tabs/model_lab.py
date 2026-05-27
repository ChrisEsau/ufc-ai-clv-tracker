import streamlit as st

def render_model_lab():
    st.markdown(
        '<div class="section-header">Model Lab</div>',
        unsafe_allow_html=True,
    )

    st.info(
        "Future area for model diagnostics, calibration, feature importance, training runs, and backtesting."
    )
