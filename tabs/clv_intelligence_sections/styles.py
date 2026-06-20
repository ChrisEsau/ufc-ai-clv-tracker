from __future__ import annotations

import streamlit as st


def inject_styles() -> None:
    st.markdown(
        """
        <style>
        .clvi-hero {
            border: 1px solid rgba(43, 60, 82, .95);
            border-radius: 14px;
            padding: 1rem 1.15rem;
            margin: .72rem 0 .9rem;
            background: linear-gradient(180deg, rgba(14, 32, 55, .98), rgba(6, 17, 31, .99));
            box-shadow: 0 18px 42px rgba(0,0,0,.24);
        }
        .clvi-hero-grid {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto;
            gap: 1rem;
            align-items: start;
        }
        .clvi-kicker {
            color: #8fb6df;
            text-transform: uppercase;
            letter-spacing: .07em;
            font-size: .66rem;
            font-weight: 740;
            margin-bottom: .25rem;
        }
        .clvi-title {
            color: #f8fbff;
            font-size: 1.48rem;
            line-height: 1.08;
            font-weight: 780;
            letter-spacing: -.026em;
        }
        .clvi-subtitle {
            color: #c9d8ea;
            font-size: .84rem;
            line-height: 1.36;
            margin-top: .34rem;
            max-width: 760px;
        }
        .clvi-artifacts {
            color: #d9e8f8;
            font-size: .72rem;
            text-align: right;
            line-height: 1.42;
            white-space: nowrap;
        }
        .clvi-pill {
            display: inline-block;
            padding: .18rem .46rem;
            border-radius: 999px;
            border: 1px solid rgba(96, 165, 250, .45);
            background: rgba(37, 99, 235, .22);
            color: #dcecff;
            font-size: .66rem;
            font-weight: 680;
            margin-left: .35rem;
        }
        .clvi-card-title {
            color: #5fb7ff;
            font-size: .98rem;
            font-weight: 700;
            letter-spacing: -.015em;
            margin-bottom: .22rem;
        }
        .clvi-card-caption {
            color: #9fb0c4;
            font-size: .72rem;
            margin-bottom: .7rem;
        }
        .clvi-leader-row {
            display: grid;
            grid-template-columns: minmax(0, 1.2fr) .5fr .55fr .55fr;
            gap: .55rem;
            align-items: center;
            padding: .5rem 0;
            border-bottom: 1px solid rgba(43,60,82,.58);
            color: #f8fbff;
            font-size: .78rem;
        }
        .clvi-leader-head {
            color: #8fa3bb;
            font-size: .66rem;
            text-transform: uppercase;
            letter-spacing: .045em;
            font-weight: 720;
        }
        .clvi-steam-row {
            display: grid;
            grid-template-columns: minmax(0, 1fr) auto auto;
            gap: .7rem;
            align-items: center;
            padding: .56rem 0;
            border-bottom: 1px solid rgba(43,60,82,.58);
        }
        .clvi-steam-main {
            min-width: 0;
        }
        .clvi-steam-side {
            color: #f8fbff;
            font-size: .78rem;
            font-weight: 690;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .clvi-steam-meta {
            color: #8fa3bb;
            font-size: .66rem;
            margin-top: .12rem;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
        }
        .clvi-steam-odds {
            color: #dbeafe;
            font-size: .78rem;
            font-weight: 720;
            white-space: nowrap;
        }
        .clvi-steam-clv {
            color: #64d899;
            font-size: .82rem;
            font-weight: 780;
            text-align: right;
            white-space: nowrap;
        }
        .clvi-empty-note {
            color: #9fb0c4;
            font-size: .78rem;
            padding: .8rem 0;
        }
        .clvi-positive { color: #64d899; font-weight: 700; }
        .clvi-negative { color: #ff7b7b; font-weight: 700; }
        .clvi-neutral { color: #dbe7f5; font-weight: 650; }
        section.main div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlockBorderWrapper"] {
            border-color: rgba(43, 60, 82, 0.95) !important;
            border-radius: 12px !important;
            background: linear-gradient(180deg, rgba(12, 27, 47, 0.97), rgba(6, 17, 31, 0.99)) !important;
            box-shadow: 0 16px 34px rgba(0, 0, 0, 0.18) !important;
        }
        section.main div[data-testid="stMetric"] {
            border: 1px solid rgba(43, 60, 82, 0.95);
            border-radius: 10px;
            padding: .58rem .72rem;
            background: rgba(10, 28, 48, .86);
        }
        section.main label,
        section.main div[data-testid="stWidgetLabel"] p {
            color: #d7e2f0 !important;
            font-size: .72rem !important;
            font-weight: 560 !important;
        }
        @media (max-width: 1000px) {
            .clvi-hero-grid { grid-template-columns: 1fr; }
            .clvi-artifacts { text-align: left; white-space: normal; }
        }
        </style>
        """,
        unsafe_allow_html=True,
    )
