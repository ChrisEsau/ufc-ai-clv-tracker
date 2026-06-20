from __future__ import annotations

from datetime import datetime, timezone

import pandas as pd
import streamlit as st

from pipeline.common.paths import CLV_RESULTS_PATH, MODEL_CANDIDATE_CLV_PATH, MODEL_CANDIDATE_TRACKER_PATH


def render_header(candidate_clv: pd.DataFrame, official_clv: pd.DataFrame) -> None:
    now = datetime.now(timezone.utc).strftime("%b %-d, %Y %I:%M %p UTC")
    candidate_rows = 0 if candidate_clv is None else len(candidate_clv)
    official_rows = 0 if official_clv is None else len(official_clv)
    st.markdown(
        f"""
        <div class="clvi-hero">
            <div class="clvi-hero-grid">
                <div>
                    <div class="clvi-kicker">Model Validation Workspace</div>
                    <div class="clvi-title">CLV Intelligence Center <span class="clvi-pill">Candidate CLV</span></div>
                    <div class="clvi-subtitle">
                        Track every model candidate from first qualifying signal through close. Use this page to see
                        whether model edge, confidence, timing, and model version are validated by market movement.
                    </div>
                </div>
                <div class="clvi-artifacts">
                    ◷ Last loaded: {now}<br/>
                    Candidates: {candidate_rows:,} · Official Bets: {official_rows:,}<br/>
                    <span style="color:#8fb6df;">{MODEL_CANDIDATE_CLV_PATH}</span><br/>
                    <span style="color:#8fb6df;">{MODEL_CANDIDATE_TRACKER_PATH}</span><br/>
                    <span style="color:#8fb6df;">{CLV_RESULTS_PATH}</span>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )
