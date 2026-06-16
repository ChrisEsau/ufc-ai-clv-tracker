from __future__ import annotations

import html

import streamlit as st


def _escape(value) -> str:
    return html.escape(str(value or ""))


def _status_card(label: str, value: str, caption: str, icon: str, tone: str = "success") -> str:
    return (
        '<div class="ops-card ops-auto-card">'
        f'<div class="ops-auto-icon {tone}">{_escape(icon)}</div>'
        '<div>'
        f'<div class="ops-auto-label">{_escape(label)}</div>'
        f'<div class="ops-auto-value {tone}">{_escape(value)}</div>'
        f'<div class="ops-auto-caption">{_escape(caption)}</div>'
        '</div></div>'
    )


def render_autopilot_summary() -> None:
    cards = [
        ("Autopilot Status", "Running", "All systems operational", "BOT", "success"),
        ("Current Runbook", "Daily Market Refresh", "Scheduled", "CAL", "info"),
        ("Next Scheduled Run", "Tomorrow 8:00 AM", "Daily Market Refresh", "CLK", "purple"),
        ("Last Completed Run", "Today 8:02 AM", "Daily Market Refresh", "OK", "success"),
        ("Alerts Requiring Review", "2", "Review items available", "ALR", "warning"),
    ]
    st.html('<div class="ops-auto-grid">' + ''.join(_status_card(*card) for card in cards) + '</div>')


def render_runbook_progress() -> None:
    steps = [
        (1, "Refresh Market Data", "Collect latest market data from configured sources", "Today 7:58 AM", "Complete", "complete"),
        (2, "Update Market Database", "Store and version latest market data", "Today 8:00 AM", "Complete", "complete"),
        (3, "Run Production Predictions", "Generate predictions for all available markets", "Running", "In Progress", "progress"),
        (4, "Update Action Board", "Recalculate edges and update review items", "—", "Waiting", "waiting"),
        (5, "Snapshot Market Lines", "Capture current lines for tracking", "—", "Waiting", "waiting"),
        (6, "Review Alert Check", "Identify items requiring human review", "—", "Waiting", "waiting"),
    ]
    rows = []
    for number, title, desc, stamp, state, tone in steps:
        rows.append(
            f'<div class="ops-runbook-row {tone}">'
            f'<div class="ops-runbook-num {tone}">{number}</div>'
            '<div class="ops-runbook-main">'
            f'<div class="ops-runbook-title">{_escape(title)}</div>'
            f'<div class="ops-runbook-desc">{_escape(desc)}</div>'
            '</div>'
            f'<div class="ops-runbook-time">{_escape(stamp)}</div>'
            f'<div class="ops-runbook-state {tone}">{_escape(state)}</div>'
            '</div>'
        )
    st.html(
        '<div class="ops-card ops-panel">'
        '<div class="ops-panel-header"><div><div class="ops-panel-title">Runbook Progress</div><div class="ops-panel-subtitle">Daily Market Refresh</div></div>'
        '<div class="ops-legend"><span><i class="complete"></i>Complete</span><span><i class="progress"></i>In Progress</span><span><i class="waiting"></i>Waiting</span><span><i class="failed"></i>Failed</span></div></div>'
        '<div class="ops-runbook-list">' + ''.join(rows) + '</div>'
        '<div class="ops-panel-note">This run is scheduled to complete around 8:15 AM CT</div>'
        '</div>'
    )


def render_upcoming_runs() -> None:
    runs = [
        ("Daily Market Refresh", "Tomorrow 8:00 AM CT", "Refresh data, run predictions, update board, snapshot lines", "Recurring"),
        ("Daily Market Refresh", "Jun 18, 2026 8:00 AM CT", "Refresh data, run predictions, update board, snapshot lines", "Recurring"),
        ("Daily Market Refresh", "Jun 19, 2026 8:00 AM CT", "Refresh data, run predictions, update board, snapshot lines", "Recurring"),
        ("Fight Day Monitor", "Jun 21, 2026 8:00 AM CT", "Increased monitoring and final snapshots", "Scheduled"),
        ("Monday Reset (Weekly)", "Jun 23, 2026 8:00 AM CT", "Ingest results, settle ledger, update data, refresh markets", "Recurring"),
    ]
    rows = []
    for name, when, desc, badge in runs:
        badge_class = "purple" if badge == "Scheduled" else "info"
        rows.append(
            '<div class="ops-upcoming-row">'
            '<div class="ops-upcoming-icon">▣</div>'
            '<div class="ops-upcoming-main">'
            f'<div class="ops-upcoming-title">{_escape(name)}</div>'
            f'<div class="ops-upcoming-desc">{_escape(desc)}</div>'
            '</div>'
            f'<div class="ops-upcoming-time">{_escape(when)}</div>'
            f'<div class="ops-mini-badge {badge_class}">{_escape(badge)}</div>'
            '</div>'
        )
    st.html(
        '<div class="ops-card ops-panel">'
        '<div class="ops-panel-header"><div class="ops-panel-title">Upcoming Runs</div><div class="ops-link-inline">View Full Schedule</div></div>'
        + ''.join(rows) + '</div>'
    )


def render_review_alerts() -> None:
    rows = [
        ("HIGH", "Review Item A", "Current event - board review", "Priority", "Above threshold", "Today 8:01 AM"),
        ("HIGH", "Review Item B", "Current event - board review", "Priority", "Above threshold", "Today 8:01 AM"),
    ]
    html_rows = []
    for level, title, detail, priority, edge, stamp in rows:
        html_rows.append(
            '<div class="ops-alert-row">'
            f'<div class="ops-alert-badge">{_escape(level)}</div>'
            '<div class="ops-alert-main">'
            f'<div class="ops-alert-title">{_escape(title)}</div>'
            f'<div class="ops-alert-detail">{_escape(detail)}</div>'
            '</div>'
            f'<div class="ops-alert-edge"><div>{_escape(priority)}</div><span>{_escape(edge)}</span></div>'
            f'<div class="ops-alert-time">{_escape(stamp)}</div>'
            '</div>'
        )
    st.html(
        '<div class="ops-card ops-panel">'
        '<div class="ops-panel-header"><div class="ops-panel-title">Review Alerts</div><div class="ops-link-inline">View All Alerts</div></div>'
        + ''.join(html_rows) + '<div class="ops-alert-footer">2 alerts require review</div></div>'
    )


def render_system_health_compact() -> None:
    rows = ["Data Ingestion", "Prediction Engine", "Database", "Notifications"]
    html_rows = ''.join(
        f'<div class="ops-health-row"><span>OK {_escape(row)}</span><span class="ops-green">Healthy</span><span>100%</span></div>'
        for row in rows
    )
    st.html(
        '<div class="ops-card ops-panel">'
        '<div class="ops-panel-header"><div class="ops-panel-title">System Health</div><div class="ops-link-inline">View Details</div></div>'
        + html_rows + '<div class="ops-health-footer">All systems operational</div></div>'
    )


def render_recent_activity_compact() -> None:
    rows = [
        ("OK", "Market data refreshed successfully", "8:00 AM"),
        ("OK", "Market database updated", "8:00 AM"),
        ("RUN", "Prediction run started", "8:00 AM"),
        ("INFO", "Scheduled run started: Daily Market Refresh", "8:00 AM"),
        ("NEXT", "Next run scheduled for tomorrow 8:00 AM", "7:59 AM"),
    ]
    html_rows = ''.join(
        '<div class="ops-activity-row">'
        f'<span>{_escape(icon)}</span><span>{_escape(text)}</span><span>{_escape(stamp)}</span>'
        '</div>'
        for icon, text, stamp in rows
    )
    st.html(
        '<div class="ops-card ops-panel">'
        '<div class="ops-panel-header"><div class="ops-panel-title">Recent Activity</div><div class="ops-link-inline">View All Logs</div></div>'
        + html_rows + '</div>'
    )


def render_autopilot_footer() -> None:
    st.html(
        '<div class="ops-card ops-footer">'
        '<div>Autopilot is running as scheduled. You will be notified when review items are available.</div>'
        '<button class="ops-settings-button">Autopilot Settings</button>'
        '</div>'
    )
