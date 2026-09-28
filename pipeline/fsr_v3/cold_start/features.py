"""Leakage-safe feature construction from objective pre-UFC evidence."""
from __future__ import annotations

from collections.abc import Iterable

import numpy as np
import pandas as pd

from .schema import evidence_bucket, normalize_name, require_columns, validate_external_bouts

BASE_FEATURE_COLUMNS = [
    "ext_bouts", "ext_wins", "ext_losses", "ext_draws", "ext_win_pct",
    "ext_finish_win_rate", "ext_ko_win_rate", "ext_sub_win_rate", "ext_dec_win_rate",
    "ext_finish_loss_rate", "ext_ko_loss_rate", "ext_sub_loss_rate",
    "ext_round1_win_rate", "ext_avg_elapsed_seconds", "ext_unique_opponents",
    "ext_days_since_last_fight", "ext_bouts_365", "ext_bouts_730",
    "ext_major_org_bouts", "ext_major_org_wins", "ext_major_org_win_pct",
    "ext_current_elo", "ext_avg_opponent_pre_elo", "ext_best_win_opponent_pre_elo",
    "ext_avg_win_opponent_pre_elo", "ext_avg_loss_opponent_pre_elo",
    "ext_height_cm", "ext_weight_kg", "ext_org_count",
    "ext_pfl_bouts", "ext_bellator_bouts", "ext_lfa_bouts", "ext_cage_warriors_bouts",
    "ext_rizin_bouts", "ext_aca_bouts", "ext_ksw_bouts", "ext_oktagon_bouts",
    "path_fights", "path_sig_attempts_per_15m", "path_sig_accuracy",
    "path_td_attempts_per_15m", "path_td_accuracy", "path_control_per_15m",
    "path_control_per_td_landed", "path_kd_per_100_sig_landed",
    "ped_wrestling", "ped_wrestling_elite", "ped_bjj", "ped_judo", "ped_sambo",
    "ped_achievements",
    "has_external_record", "has_opponent_quality", "has_pathway_stats", "has_pedigree",
]

PROMOTION_TOKENS = {
    "pfl": ("pfl", "professional fighters league"),
    "bellator": ("bellator",),
    "lfa": ("lfa", "legacy fighting alliance"),
    "cage_warriors": ("cage warriors", "cagewarriors"),
    "rizin": ("rizin",),
    "aca": ("aca", "absolute championship akhmat"),
    "ksw": ("ksw",),
    "oktagon": ("oktagon",),
}


def _safe_rate(num: float, den: float) -> float:
    return float(num / den) if den > 0 else np.nan


def _elapsed_seconds(history: pd.DataFrame) -> pd.Series:
    rounds = pd.to_numeric(history["round_num"], errors="coerce")
    seconds = pd.to_numeric(history["time_finish_seconds"], errors="coerce")
    return (rounds - 1.0) * 300.0 + seconds


def _promotion_count(history: pd.DataFrame, tokens: tuple[str, ...]) -> int:
    org = history["organization"].fillna("").astype(str).str.lower()
    return int(org.map(lambda x: any(token in x for token in tokens)).sum())


def _aggregate_bouts(history: pd.DataFrame, as_of: pd.Timestamp) -> dict[str, float | int | bool]:
    if history.empty:
        return {column: np.nan for column in BASE_FEATURE_COLUMNS if not column.startswith("has_")} | {
            "ext_bouts": 0,
            "ext_wins": 0,
            "ext_losses": 0,
            "ext_draws": 0,
            "ext_unique_opponents": 0,
            "ext_bouts_365": 0,
            "ext_bouts_730": 0,
            "ext_major_org_bouts": 0,
            "ext_major_org_wins": 0,
            "ext_org_count": 0,
            "ext_pfl_bouts": 0,
            "ext_bellator_bouts": 0,
            "ext_lfa_bouts": 0,
            "ext_cage_warriors_bouts": 0,
            "ext_rizin_bouts": 0,
            "ext_aca_bouts": 0,
            "ext_ksw_bouts": 0,
            "ext_oktagon_bouts": 0,
            "has_external_record": False,
            "has_opponent_quality": False,
            "has_pathway_stats": False,
            "has_pedigree": False,
            "path_fights": 0,
            "ped_achievements": 0,
        }

    h = history.sort_values(["event_date", "fight_id"]).copy()
    n = len(h)
    wins = int((h["result"] == "W").sum())
    losses = int((h["result"] == "L").sum())
    draws = int((h["result"] == "D").sum())
    completed = wins + losses
    win_h = h[h["result"] == "W"]
    loss_h = h[h["result"] == "L"]
    win_methods = win_h["method_class"]
    loss_methods = loss_h["method_class"]
    elapsed = _elapsed_seconds(h)
    days = (as_of - h["event_date"]).dt.days
    major = h[h["is_major_org"].fillna(False).astype(bool)]
    latest = h.iloc[-1]

    out: dict[str, float | int | bool] = {
        "ext_bouts": int(n),
        "ext_wins": wins,
        "ext_losses": losses,
        "ext_draws": draws,
        "ext_win_pct": _safe_rate(wins, completed),
        "ext_finish_win_rate": _safe_rate(int(win_methods.isin(["KO_TKO", "SUB"]).sum()), wins),
        "ext_ko_win_rate": _safe_rate(int((win_methods == "KO_TKO").sum()), wins),
        "ext_sub_win_rate": _safe_rate(int((win_methods == "SUB").sum()), wins),
        "ext_dec_win_rate": _safe_rate(int((win_methods == "DEC").sum()), wins),
        "ext_finish_loss_rate": _safe_rate(int(loss_methods.isin(["KO_TKO", "SUB"]).sum()), losses),
        "ext_ko_loss_rate": _safe_rate(int((loss_methods == "KO_TKO").sum()), losses),
        "ext_sub_loss_rate": _safe_rate(int((loss_methods == "SUB").sum()), losses),
        "ext_round1_win_rate": _safe_rate(int(((win_h["round_num"] == 1) & win_methods.isin(["KO_TKO", "SUB"])).sum()), wins),
        "ext_avg_elapsed_seconds": float(elapsed.mean()) if elapsed.notna().any() else np.nan,
        "ext_unique_opponents": int(h["opponent_key"].nunique()),
        "ext_days_since_last_fight": float(days.iloc[-1]) if len(days) else np.nan,
        "ext_bouts_365": int((days <= 365).sum()),
        "ext_bouts_730": int((days <= 730).sum()),
        "ext_major_org_bouts": int(len(major)),
        "ext_major_org_wins": int((major["result"] == "W").sum()),
        "ext_major_org_win_pct": _safe_rate(int((major["result"] == "W").sum()), int(major["result"].isin(["W", "L"]).sum())),
        "ext_current_elo": float(latest["fighter_post_elo"]) if pd.notna(latest.get("fighter_post_elo", np.nan)) else np.nan,
        "ext_avg_opponent_pre_elo": float(h["opponent_pre_elo"].mean()) if "opponent_pre_elo" in h and h["opponent_pre_elo"].notna().any() else np.nan,
        "ext_best_win_opponent_pre_elo": float(win_h["opponent_pre_elo"].max()) if "opponent_pre_elo" in win_h and win_h["opponent_pre_elo"].notna().any() else np.nan,
        "ext_avg_win_opponent_pre_elo": float(win_h["opponent_pre_elo"].mean()) if "opponent_pre_elo" in win_h and win_h["opponent_pre_elo"].notna().any() else np.nan,
        "ext_avg_loss_opponent_pre_elo": float(loss_h["opponent_pre_elo"].mean()) if "opponent_pre_elo" in loss_h and loss_h["opponent_pre_elo"].notna().any() else np.nan,
        "ext_height_cm": float(h["fighter_height_cm"].dropna().iloc[-1]) if h["fighter_height_cm"].notna().any() else np.nan,
        "ext_weight_kg": float(h["fighter_weight_kg"].dropna().iloc[-1]) if h["fighter_weight_kg"].notna().any() else np.nan,
        "ext_org_count": int(h["organization"].nunique()),
        "has_external_record": True,
        "has_opponent_quality": bool("opponent_pre_elo" in h and h["opponent_pre_elo"].notna().any()),
        "has_pathway_stats": False,
        "has_pedigree": False,
        "path_fights": 0,
        "ped_achievements": 0,
    }
    for label, tokens in PROMOTION_TOKENS.items():
        out[f"ext_{label}_bouts"] = _promotion_count(h, tokens)
    for column in BASE_FEATURE_COLUMNS:
        out.setdefault(column, np.nan)
    return out


def _aggregate_pathway(pathway: pd.DataFrame) -> dict[str, float | int | bool]:
    if pathway.empty:
        return {"path_fights": 0, "has_pathway_stats": False}
    p = pathway.copy()
    def total(col: str) -> float:
        return float(pd.to_numeric(p[col], errors="coerce").fillna(0.0).sum()) if col in p else 0.0
    elapsed = total("elapsed_seconds")
    sig_att = total("sig_attempted")
    sig_land = total("sig_landed")
    td_att = total("td_attempted")
    td_land = total("td_landed")
    ctrl = total("control_seconds")
    kd = total("knockdowns")
    return {
        "path_fights": int(p["fight_id"].nunique()) if "fight_id" in p else int(len(p)),
        "path_sig_attempts_per_15m": sig_att / elapsed * 900.0 if elapsed > 0 else np.nan,
        "path_sig_accuracy": _safe_rate(sig_land, sig_att),
        "path_td_attempts_per_15m": td_att / elapsed * 900.0 if elapsed > 0 else np.nan,
        "path_td_accuracy": _safe_rate(td_land, td_att),
        "path_control_per_15m": ctrl / elapsed * 900.0 if elapsed > 0 else np.nan,
        "path_control_per_td_landed": _safe_rate(ctrl, td_land),
        "path_kd_per_100_sig_landed": kd / sig_land * 100.0 if sig_land > 0 else np.nan,
        "has_pathway_stats": True,
    }


def _aggregate_pedigree(pedigree: pd.DataFrame) -> dict[str, float | int | bool]:
    if pedigree.empty:
        return {"ped_achievements": 0, "has_pedigree": False}
    text = (pedigree.get("discipline", pd.Series(index=pedigree.index, dtype=str)).fillna("").astype(str) + " " +
            pedigree.get("level", pd.Series(index=pedigree.index, dtype=str)).fillna("").astype(str)).str.lower()
    wrestling = text.str.contains("wrest").any()
    elite = text.str.contains(r"olympic|world|ncaa.*champ|ncaa.*all.?american", regex=True).any()
    return {
        "ped_wrestling": bool(wrestling),
        "ped_wrestling_elite": bool(wrestling and elite),
        "ped_bjj": bool(text.str.contains(r"bjj|jiu.?jitsu", regex=True).any()),
        "ped_judo": bool(text.str.contains("judo").any()),
        "ped_sambo": bool(text.str.contains("sambo").any()),
        "ped_achievements": int(len(pedigree)),
        "has_pedigree": True,
    }


def build_external_feature_snapshots(
    targets: pd.DataFrame,
    external_bouts: pd.DataFrame,
    *,
    pathway_stats: pd.DataFrame | None = None,
    pedigree: pd.DataFrame | None = None,
    exclude_organizations: Iterable[str] = ("ufc",),
) -> pd.DataFrame:
    """Build one external-evidence row per target using only dates before target.

    ``targets`` requires fighter_name/as_of_date and may carry fighter_id,
    fight_id, or other keys.  Exact normalized-name linkage is deliberate:
    ambiguous/fuzzy aliases must be resolved in a separate auditable alias table.
    """
    require_columns(targets, ["fighter_name", "as_of_date"], "cold-start targets")
    bouts = validate_external_bouts(external_bouts)
    excluded = {str(x).lower().strip() for x in exclude_organizations}
    bouts = bouts[~bouts["organization"].isin(excluded)].copy()
    histories = {key: g.sort_values("event_date") for key, g in bouts.groupby("fighter_key", sort=False)}

    pstats = None
    if pathway_stats is not None and not pathway_stats.empty:
        require_columns(pathway_stats, ["fighter_name", "event_date"], "pathway stats")
        pstats = pathway_stats.copy()
        pstats["event_date"] = pd.to_datetime(pstats["event_date"], errors="raise").dt.normalize()
        pstats["fighter_key"] = pstats["fighter_name"].map(normalize_name)

    ped = None
    if pedigree is not None and not pedigree.empty:
        require_columns(pedigree, ["fighter_name", "achievement_date", "discipline", "level"], "pedigree")
        ped = pedigree.copy()
        ped["achievement_date"] = pd.to_datetime(ped["achievement_date"], errors="raise").dt.normalize()
        ped["fighter_key"] = ped["fighter_name"].map(normalize_name)

    rows: list[dict[str, object]] = []
    for record in targets.to_dict("records"):
        as_of = pd.Timestamp(record["as_of_date"]).normalize()
        key = normalize_name(record["fighter_name"])
        h = histories.get(key)
        h = h[h["event_date"] < as_of].copy() if h is not None else bouts.iloc[0:0].copy()
        out = dict(record)
        out["as_of_date"] = as_of
        out["fighter_key"] = key
        out.update(_aggregate_bouts(h, as_of))

        if pstats is not None:
            ph = pstats[(pstats["fighter_key"] == key) & (pstats["event_date"] < as_of)]
            out.update(_aggregate_pathway(ph))
        if ped is not None:
            pdh = ped[(ped["fighter_key"] == key) & (ped["achievement_date"] < as_of)]
            out.update(_aggregate_pedigree(pdh))

        for column in BASE_FEATURE_COLUMNS:
            out.setdefault(column, np.nan)
        out["evidence_bucket"] = evidence_bucket(out.get("ext_bouts", 0))
        out["coverage_signature"] = "+".join(
            label for label, flag in (
                ("record", out.get("has_external_record", False)),
                ("quality", out.get("has_opponent_quality", False)),
                ("pathway", out.get("has_pathway_stats", False)),
                ("pedigree", out.get("has_pedigree", False)),
            ) if bool(flag)
        ) or "none"
        if not h.empty and not (h["event_date"] < as_of).all():
            raise AssertionError("cold-start feature leakage: external bout on/after target date")
        rows.append(out)
    return pd.DataFrame(rows)
