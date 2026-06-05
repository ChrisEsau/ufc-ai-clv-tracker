"""Page and section layout helpers."""

from __future__ import annotations

import html
import streamlit as st


def page_header(
    title: str,
    subtitle: str,
    kicker: str | None = None,
    updated_label: str | None = None,
) -> None:
    """Render the common workspace header used by all tabs.

    ``updated_label`` is intentionally opt-in so the UI does not fabricate a
    "Last Updated" timestamp when a tab has not loaded a real artifact time.
    """

    kicker_html = (
        f'<div class="ufc-kicker">{html.escape(kicker)}</div>' if kicker else ""
    )
    updated_html = (
        f'<div class="ufc-updated">{html.escape(updated_label)}</div>'
        if updated_label
        else ""
    )
    st.markdown(
        f"""
        <div class="ufc-page-header">
            <div>
                {kicker_html}
                <h1 class="ufc-title">{html.escape(title)}</h1>
                <div class="ufc-subtitle">{html.escape(subtitle)}</div>
            </div>
            {updated_html}
        </div>
        """,
        unsafe_allow_html=True,
    )


def section_heading(
    title: str, caption: str | None = None, icon: str | None = None
) -> None:
    label = f"{icon} {title}" if icon else title
    caption_html = (
        f'<div class="section-caption">{html.escape(caption)}</div>' if caption else ""
    )
    st.markdown(
        f'<div class="section-header">{html.escape(label)}</div>{caption_html}',
        unsafe_allow_html=True,
    )


def section_divider() -> None:
    st.markdown(
        '<div style="height:1px;background:rgba(38,54,74,.75);margin:1rem 0;"></div>',
        unsafe_allow_html=True,
    )
