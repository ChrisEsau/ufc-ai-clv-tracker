from __future__ import annotations

import streamlit as st


def render_model_status() -> None:
    st.html(
        """
        <div class="ops-card ops-model-strip">
            <div>
                <div class="ops-strip-label">Production Model</div>
                <div class="ops-strip-value">moneyline_xgb_v5 <span class="ops-badge success">PRODUCTION</span></div>
            </div>
            <div>
                <div class="ops-strip-label">Last Trained</div>
                <div class="ops-strip-value">Model registry</div>
            </div>
            <div>
                <div class="ops-strip-label">Next Scheduled Train</div>
                <div class="ops-strip-value">Not scheduled</div>
            </div>
            <div>
                <div class="ops-strip-label">Uptime</div>
                <div class="ops-strip-value"><span class="ops-green">Operational</span></div>
            </div>
        </div>
        """
    )
