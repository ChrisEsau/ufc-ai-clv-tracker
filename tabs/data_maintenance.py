import streamlit as st
import pandas as pd

from utils.data_loader import load_parquet
from utils.ui_components import render_metric
from utils.panels import render_section_header


def render_data_maintenance():

    render_section_header("Data Maintenance")

    st.caption(
        "Dataset freshness, ingestion readiness, feature audits, fighter matching, and pipeline health."
    )

    # ========================================================
    # LOAD ARTIFACTS
    # ========================================================

    dataset_status = load_parquet("ufc_dataset_status.parquet")
    event_status = load_parquet("ufc_dataset_event_status.parquet")
    feature_audit = load_parquet("ufc_live_feature_audit.parquet")
    match_audit = load_parquet("ufc_live_match_audit.parquet")
    market_audit = load_parquet("ufc_market_match_audit.parquet")
    event_check = load_parquet("ufc_ufcstats_event_check.parquet")
    missing_events = load_parquet("ufc_missing_events.parquet")

    # ========================================================
    # DATASET STATUS
    # ========================================================

    render_section_header("Dataset Status")

    if dataset_status.empty:
        st.warning(
            "No dataset status artifact found yet. Expected: ufc_dataset_status.parquet"
        )
    else:
        latest = dataset_status.iloc[0]

        c1, c2, c3, c4, c5, c6 = st.columns(6)

        with c1:
            render_metric("Rows", latest.get("row_count", "N/A"), accent="blue")

        with c2:
            render_metric("Columns", latest.get("column_count", "N/A"), accent="purple")

        with c3:
            render_metric("Unique Fighters", latest.get("unique_fighters", "N/A"), accent="green")

        with c4:
            render_metric("Events", latest.get("unique_events", "N/A"), accent="amber")

        with c5:
            render_metric("Duplicates", latest.get("duplicate_fight_count", "N/A"), accent="red")

        with c6:
            render_metric("Missing Results", latest.get("missing_result_count", "N/A"), accent="red")

        st.subheader("Latest Dataset Snapshot")

        display_cols = [
            "run_timestamp",
            "dataset_path",
            "earliest_fight_date",
            "latest_fight_date",
            "latest_event_name",
            "latest_event_fight_count",
            "missing_red_id_count",
            "missing_blue_id_count",
            "invalid_date_count",
        ]

        display_cols = [c for c in display_cols if c in dataset_status.columns]

        st.dataframe(
            dataset_status[display_cols],
            use_container_width=True,
            hide_index=True,
        )
    # ========================================================
    # UFCSTATS EVENT CHECK
    # ========================================================

    render_section_header("UFCStats Event Check")

    if event_check.empty:
        st.warning("No UFCStats event check artifact found yet.")
    else:
        missing_count = len(missing_events)
    
        latest_ufcstats = (
            event_check
            .sort_values("ufcstats_event_date", ascending=False)
            .iloc[0]
            .get("ufcstats_event_name", "N/A")
        )
    
        latest_local = (
            dataset_status.iloc[0].get("latest_event_name", "N/A")
            if not dataset_status.empty
            else "N/A"
        )
    
        status_text = "CURRENT" if missing_count == 0 else "OUTDATED"
    
        c1, c2, c3, c4 = st.columns(4)
    
        with c1:
            render_metric(
                "Dataset Status",
                status_text,
                accent="green" if missing_count == 0 else "red",
            )
    
        with c2:
            render_metric(
                "Missing Events",
                missing_count,
                accent="green" if missing_count == 0 else "red",
            )
    
        with c3:
            render_metric(
                "Latest Local",
                latest_local,
                accent="blue",
            )
    
        with c4:
            render_metric(
                "Latest UFCStats",
                latest_ufcstats,
                accent="amber",
            )
    
        if missing_count == 0:
            st.success("Local dataset appears current with UFCStats completed events.")
        else:
            st.error(
                f"{missing_count} completed UFCStats events are missing from the local dataset."
            )
    
            display_cols = [
                "ufcstats_event_name",
                "ufcstats_event_date",
                "status",
                "ufcstats_event_url",
            ]
    
            display_cols = [
                c for c in display_cols
                if c in missing_events.columns
            ]
    
            st.dataframe(
                missing_events[display_cols],
                use_container_width=True,
                hide_index=True,
            )
    # ========================================================
    # EVENT STATUS
    # ========================================================

    render_section_header("Recent Event Status")

    if event_status.empty:
        st.info(
            "No event status artifact found yet. Expected: ufc_dataset_event_status.parquet"
        )
    else:
        st.dataframe(
            event_status.head(25),
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # LIVE FEATURE AUDIT
    # ========================================================

    render_section_header("Live Feature Audit")

    if feature_audit.empty:
        st.info("No live feature audit found.")
    else:
        failed_feature = (
            (~feature_audit["passes_feature_validation"]).sum()
            if "passes_feature_validation" in feature_audit.columns
            else 0
        )

        failed_match = (
            (~feature_audit["passes_match_quality"]).sum()
            if "passes_match_quality" in feature_audit.columns
            else 0
        )

        avg_zero = (
            feature_audit["zero_feature_pct"].mean()
            if "zero_feature_pct" in feature_audit.columns
            else 0
        )

        f1, f2, f3, f4 = st.columns(4)

        with f1:
            render_metric("Fights Audited", len(feature_audit), accent="blue")

        with f2:
            render_metric("Failed Features", int(failed_feature), accent="red")

        with f3:
            render_metric("Failed Matches", int(failed_match), accent="red")

        with f4:
            render_metric("Avg Zero %", f"{avg_zero:.1f}%", accent="amber")

        feature_cols = [
            "event_name",
            "red_fighter",
            "blue_fighter",
            "red_feature_match",
            "blue_feature_match",
            "nonzero_feature_count",
            "zero_feature_pct",
            "passes_match_quality",
            "passes_feature_validation",
        ]

        feature_cols = [c for c in feature_cols if c in feature_audit.columns]

        st.dataframe(
            feature_audit[feature_cols].sort_values(
                "zero_feature_pct",
                ascending=False,
            ),
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # FIGHTER MATCH AUDIT
    # ========================================================

    render_section_header("Fighter Match Audit")

    if match_audit.empty:
        st.info("No fighter match audit found.")
    else:
        st.dataframe(
            match_audit,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # MARKET MATCH AUDIT
    # ========================================================

    render_section_header("Market Match Audit")

    if market_audit.empty:
        st.info("No market match audit found.")
    else:
        low_conf = (
            (market_audit["odds_match_type"] != "matched").sum()
            if "odds_match_type" in market_audit.columns
            else 0
        )

        m1, m2, m3 = st.columns(3)

        with m1:
            render_metric("Market Rows", len(market_audit), accent="blue")

        with m2:
            render_metric("Low Confidence", int(low_conf), accent="red")

        with m3:
            avg_score = (
                market_audit["odds_match_score"].mean()
                if "odds_match_score" in market_audit.columns
                else 0
            )
            render_metric("Avg Match Score", f"{avg_score:.1f}", accent="green")

        st.dataframe(
            market_audit,
            use_container_width=True,
            hide_index=True,
        )

    # ========================================================
    # MAINTENANCE ACTIONS
    # ========================================================

    render_section_header("Maintenance Actions")

    st.info(
        "Next phase: connect these actions to GitHub workflows for dataset status, UFCStats ingestion, rolling feature rebuild, and current fighter feature rebuild."
    )

    c1, c2, c3, c4 = st.columns(4)

    with c1:
        st.button("Check Dataset Status", use_container_width=True)

    with c2:
        st.button("Check New UFCStats Events", use_container_width=True)

    with c3:
        st.button("Rebuild Rolling Features", use_container_width=True)

    with c4:
        st.button("Rebuild Current Fighters", use_container_width=True)