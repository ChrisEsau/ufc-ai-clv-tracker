"""Calculate immediate post-fight canonical FSR from an existing prefight FSR row.

This is the fast/incremental path.  It does NOT replay the fighter-rating database.
It reads the already-built FSR-32 prefight snapshot, derives the selected fight's
observations from the authoritative RFS / round / master parquets, reconstructs
only the leakage-safe population contexts needed by the observation functions,
and applies exactly one update for the selected fight.

Reservoir traits are state-derived rather than Elo updates, so their compact
historical state is reconstructed directly from raw RFS exposure rows.  Fresh
striking power is reconstructed directly from its evidence table through the
selected fight.  Neither operation rebuilds the canonical FSR database.

The module exposes PostfightFSRCalculator so batch diagnostics can reuse cached
population contexts across many fights/dates.

Research/shadow only.
"""
from __future__ import annotations

from bisect import insort
from collections import defaultdict
from copy import deepcopy
from pathlib import Path

import numpy as np
import pandas as pd

from scripts.experimental import build_fsr_32_database as power32
from scripts.experimental import build_fsr_canonical_database as canonical
from scripts.experimental import fsr_clinch_striking_v1 as clinch
from scripts.experimental import fsr_distance_striking_pressure_v1 as distance
from scripts.experimental import fsr_dynamic_families_v1 as dynamic
from scripts.experimental import fsr_finish_reservoir_traits_v1 as reservoir
from scripts.experimental import fsr_ground_striking_v1 as ground
from scripts.experimental import fsr_locked_families_v1 as locked_v1
from scripts.experimental import fsr_locked_families_v1_1 as locked_v11
from scripts.experimental import fsr_reversal_v1 as reversal


RFS_PATH = Path("data/features/round_fighter_state_history.parquet")
ROUND_PATH = Path("data/fight_details/ufc_round_stats.parquet")
MASTER_PATH = Path("data/master/ufc_master.parquet")
FSR32_PATH = Path(
    "data/simulation/rfs_mc_v2_shared_state/fsr_32_shadow/"
    "fsr_32_prefight_snapshots.parquet"
)

LOCKED_MAP = {
    "distance_precision": "distance_striking_precision",
    "distance_defense": "distance_striking_defense",
    "wrestling_entry": "wrestling_entry",
    "wrestling_conversion": "wrestling_conversion",
    "td_defense": "td_defense",
    "control_imposition": "control_imposition",
    "control_resistance": "control_resistance",
    "submission_pressure": "submission_pressure",
    "submission_conversion": "submission_conversion",
    "submission_resistance": "submission_resistance",
}
DYNAMIC_SKILLS = (
    "fatigue_accumulation_resistance",
    "fatigue_performance_resilience",
    "adversity_resistance",
    "adversity_recovery",
)


def _date_col(frame: pd.DataFrame) -> str:
    if "date" in frame.columns:
        return "date"
    if "event_date" in frame.columns:
        return "event_date"
    raise RuntimeError("frame has neither date nor event_date")


def _norm(frame: pd.DataFrame) -> pd.DataFrame:
    out = frame.copy()
    out["fight_id"] = out["fight_id"].astype(str)
    if "fighter_id" in out.columns:
        out["fighter_id"] = out["fighter_id"].astype(str)
    dc = _date_col(out)
    out["date"] = pd.to_datetime(out[dc], errors="raise")
    return out


def _clamp(module, value: float) -> float:
    return float(module.clamp(float(value), module.MIN_RATING, module.MAX_RATING))


class PostfightFSRCalculator:
    """Cached one-fight updater over the existing FSR-32 database."""

    def __init__(self) -> None:
        for path in (RFS_PATH, ROUND_PATH, MASTER_PATH, FSR32_PATH):
            if not path.exists():
                raise RuntimeError(f"required input not found: {path}")

        print("[postfight FSR] loading existing FSR-32 + source parquets...", flush=True)
        self.rfs = _norm(pd.read_parquet(RFS_PATH))
        self.rounds = _norm(pd.read_parquet(ROUND_PATH))
        self.master = _norm(pd.read_parquet(MASTER_PATH))
        self.fsr32 = _norm(pd.read_parquet(FSR32_PATH))
        self.fsr32["fighter_id"] = self.fsr32["fighter_id"].astype(str)

        self._locked_context: dict[pd.Timestamp, dict] = {}
        self._dynamic_context: dict[pd.Timestamp, dict] = {}
        self._ground_context: dict[pd.Timestamp, dict] = {}
        self._clinch_context: dict[pd.Timestamp, dict] = {}
        self._distance_context: dict[pd.Timestamp, dict] = {}
        self._reversal_context: dict[pd.Timestamp, dict] = {}
        self._reservoir_post_cache: dict[pd.Timestamp, dict[str, tuple[float, float]]] = {}

        print("[postfight FSR] preparing fresh-power evidence once...", flush=True)
        self.fight_order = power32._fight_order_table(self.master)
        self.power_evidence = power32._power_evidence_by_fighter_fight(
            self.master,
            self.rounds,
            self.fight_order,
        )
        self._fight_order_map = dict(
            zip(
                self.fight_order["fight_id"].astype(str),
                self.fight_order["fight_order"].astype(int),
            )
        )
        print("[postfight FSR] ready; no rating replay performed", flush=True)

    def _target(self, fight_id: str) -> tuple[pd.Timestamp, pd.DataFrame, pd.DataFrame]:
        fight_id = str(fight_id)
        target = self.rfs.loc[self.rfs["fight_id"].eq(fight_id)].copy()
        if len(target) != 2:
            raise RuntimeError(f"fight {fight_id}: expected 2 RFS rows, found {len(target)}")
        date = pd.Timestamp(target["date"].iloc[0])
        pre = self.fsr32.loc[self.fsr32["fight_id"].eq(fight_id)].copy()
        if len(pre) != 2:
            raise RuntimeError(f"fight {fight_id}: expected 2 FSR-32 rows, found {len(pre)}")
        return date, target, pre

    @staticmethod
    def _pair_rows(fight: pd.DataFrame):
        rows = [row for _, row in fight.iterrows()]
        if len(rows) != 2:
            raise RuntimeError("fight does not have exactly two fighter rows")
        return ((rows[0], rows[1]), (rows[1], rows[0]))

    # ------------------------------------------------------------------
    # Prior-date observation contexts. These scan raw observations only;
    # they never reconstruct/replay fighter ratings.
    # ------------------------------------------------------------------
    def _build_locked_context(self, date: pd.Timestamp) -> dict:
        if date in self._locked_context:
            return self._locked_context[date]
        pools = {key: [] for key in locked_v1.POOL_KEYS}
        weighted = defaultdict(float)
        quality_sum = defaultdict(float)
        updates = defaultdict(lambda: defaultdict(int))
        hist = self.rfs.loc[self.rfs["date"] < date].sort_values(
            ["date", "fight_id", "fighter_id"]
        )
        for _, day in hist.groupby("date", sort=True):
            day_weighted = defaultdict(float)
            day_quality = defaultdict(float)
            for _, fight in day.groupby("fight_id", sort=False):
                if len(fight) != 2:
                    continue
                for row, opp in self._pair_rows(fight):
                    fid = str(row["fighter_id"])
                    bundle = locked_v1.observation_bundle(row, opp, pools)
                    for skill in locked_v1.SKILLS:
                        obs, q = bundle[skill]
                        if obs is None or q <= 0.0:
                            continue
                        updates[fid][skill] += 1
                        day_weighted[skill] += q * float(obs)
                        day_quality[skill] += q
            for skill in locked_v1.SKILLS:
                weighted[skill] += day_weighted[skill]
                quality_sum[skill] += day_quality[skill]
            locked_v1.append_date_to_pools(day, pools)
        ctx = {"pools": pools, "weighted": weighted, "quality": quality_sum, "updates": updates}
        self._locked_context[date] = ctx
        return ctx

    def _build_dynamic_context(self, date: pd.Timestamp) -> dict:
        if date in self._dynamic_context:
            return self._dynamic_context[date]
        pools = {key: [] for key in dynamic.POOL_KEYS}
        weighted = defaultdict(float)
        quality_sum = defaultdict(float)
        updates = defaultdict(lambda: defaultdict(int))
        hist = self.rfs.loc[self.rfs["date"] < date].sort_values(
            ["date", "fight_id", "fighter_id"]
        )
        for _, day in hist.groupby("date", sort=True):
            day_weighted = defaultdict(float)
            day_quality = defaultdict(float)
            for _, row in day.iterrows():
                fid = str(row["fighter_id"])
                bundle = dynamic.observation_bundle(row, pools)
                for skill in DYNAMIC_SKILLS:
                    obs, q = bundle[skill]
                    if obs is None or q <= 0.0:
                        continue
                    updates[fid][skill] += 1
                    day_weighted[skill] += q * float(obs)
                    day_quality[skill] += q
            for skill in DYNAMIC_SKILLS:
                weighted[skill] += day_weighted[skill]
                quality_sum[skill] += day_quality[skill]
            dynamic.append_date_to_pools(day, pools)
        ctx = {"pools": pools, "weighted": weighted, "quality": quality_sum, "updates": updates}
        self._dynamic_context[date] = ctx
        return ctx

    def _build_three_skill_context(self, date: pd.Timestamp, module, cache: dict) -> dict:
        if date in cache:
            return cache[date]
        pools = {key: [] for key in module.POOL_KEYS}
        weighted = defaultdict(float)
        quality_sum = defaultdict(float)
        updates = defaultdict(lambda: defaultdict(int))
        hist = self.rfs.loc[self.rfs["date"] < date].sort_values(
            ["date", "fight_id", "fighter_id"]
        )
        for _, day in hist.groupby("date", sort=True):
            day_weighted = defaultdict(float)
            day_quality = defaultdict(float)
            for _, row in day.iterrows():
                fid = str(row["fighter_id"])
                bundle = module.observation_bundle(row, pools)
                for skill in module.SKILLS:
                    obs, q = bundle[skill]
                    if obs is None or q <= 0.0:
                        continue
                    updates[fid][skill] += 1
                    day_weighted[skill] += q * float(obs)
                    day_quality[skill] += q
            for skill in module.SKILLS:
                weighted[skill] += day_weighted[skill]
                quality_sum[skill] += day_quality[skill]
            module.append_date_to_pools(day, pools)
        ctx = {"pools": pools, "weighted": weighted, "quality": quality_sum, "updates": updates}
        cache[date] = ctx
        return ctx

    def _build_distance_context(self, date: pd.Timestamp) -> dict:
        if date in self._distance_context:
            return self._distance_context[date]
        pools = {key: [] for key in distance.POOL_KEYS}
        weighted = 0.0
        quality_sum = 0.0
        updates = defaultdict(int)
        hist = self.rfs.loc[self.rfs["date"] < date].sort_values(
            ["date", "fight_id", "fighter_id"]
        )
        for _, day in hist.groupby("date", sort=True):
            dw = 0.0
            dq = 0.0
            for _, row in day.iterrows():
                fid = str(row["fighter_id"])
                obs, q = distance.observation(row, pools)
                if obs is not None and q > 0.0:
                    updates[fid] += 1
                    dw += q * float(obs)
                    dq += q
            weighted += dw
            quality_sum += dq
            distance.append_date_to_pools(day, pools)
        ctx = {"pools": pools, "weighted": weighted, "quality": quality_sum, "updates": updates}
        self._distance_context[date] = ctx
        return ctx

    def _build_reversal_context(self, date: pd.Timestamp) -> dict:
        if date in self._reversal_context:
            return self._reversal_context[date]
        positive_pool: list[float] = []
        weighted = 0.0
        quality_sum = 0.0
        updates = defaultdict(int)
        hist = self.rfs.loc[self.rfs["date"] < date].sort_values(
            ["date", "fight_id", "fighter_id"]
        )
        for _, day in hist.groupby("date", sort=True):
            dw = 0.0
            dq = 0.0
            rates: list[float] = []
            for _, row in day.iterrows():
                fid = str(row["fighter_id"])
                revs = reversal.finite(row.get(reversal.C["reversals"])) or 0.0
                ctrl = reversal.finite(row.get(reversal.C["opp_control_seconds"])) or 0.0
                obs, q, rate = reversal.observation(revs, ctrl, positive_pool)
                if obs is not None and q > 0.0:
                    updates[fid] += 1
                    dw += q * float(obs)
                    dq += q
                    if rate is not None:
                        rates.append(float(rate))
            weighted += dw
            quality_sum += dq
            for rate in rates:
                insort(positive_pool, rate)
        ctx = {"pool": positive_pool, "weighted": weighted, "quality": quality_sum, "updates": updates}
        self._reversal_context[date] = ctx
        return ctx

    # ------------------------------------------------------------------
    # One target-fight Elo update per family.
    # ------------------------------------------------------------------
    def _update_locked(self, target: pd.DataFrame, pre_by_id: dict[str, pd.Series], date) -> dict[str, dict[str, float]]:
        ctx = self._build_locked_context(date)
        out = defaultdict(dict)
        for row, opp in self._pair_rows(target):
            fid = str(row["fighter_id"])
            oid = str(opp["fighter_id"])
            ratings = {
                fid: {skill: float(pre_by_id[fid][LOCKED_MAP.get(skill, skill)]) for skill in locked_v1.SKILLS},
                oid: {skill: float(pre_by_id[oid][LOCKED_MAP.get(skill, skill)]) for skill in locked_v1.SKILLS},
            }
            bundle = locked_v1.observation_bundle(row, opp, ctx["pools"])
            for skill, canonical_name in LOCKED_MAP.items():
                pre = ratings[fid][skill]
                obs, q = bundle[skill]
                if obs is None or q <= 0.0:
                    post = pre
                else:
                    baseline = locked_v11.population_baseline(ctx["weighted"], ctx["quality"], skill)
                    expected = locked_v11.expected_probability(ratings, fid, oid, skill, baseline)
                    delta = locked_v1.k_factor(ctx["updates"][fid][skill]) * q * (float(obs) - expected)
                    post = locked_v1.clamp(pre + delta, locked_v1.MIN_RATING, locked_v1.MAX_RATING)
                out[fid][canonical_name] = float(post)
        return out

    def _update_dynamic(self, target: pd.DataFrame, pre_by_id: dict[str, pd.Series], date) -> dict[str, dict[str, float]]:
        ctx = self._build_dynamic_context(date)
        out = defaultdict(dict)
        for _, row in target.iterrows():
            fid = str(row["fighter_id"])
            bundle = dynamic.observation_bundle(row, ctx["pools"])
            for skill in DYNAMIC_SKILLS:
                pre = float(pre_by_id[fid][skill])
                obs, q = bundle[skill]
                if obs is None or q <= 0.0:
                    post = pre
                else:
                    baseline = dynamic.population_baseline(ctx["weighted"], ctx["quality"], skill)
                    expected = dynamic.expected_probability(pre, baseline)
                    delta = dynamic.k_factor(ctx["updates"][fid][skill]) * q * (float(obs) - expected)
                    post = dynamic.clamp(pre + delta, dynamic.MIN_RATING, dynamic.MAX_RATING)
                out[fid][skill] = float(post)
        return out

    def _update_three_skill(self, target, pre_by_id, date, module, cache) -> dict[str, dict[str, float]]:
        ctx = self._build_three_skill_context(date, module, cache)
        out = defaultdict(dict)
        rows = [row for _, row in target.iterrows()]
        for idx, row in enumerate(rows):
            opp = rows[1 - idx]
            fid = str(row["fighter_id"])
            oid = str(opp["fighter_id"])
            bundle = module.observation_bundle(row, ctx["pools"])
            for skill in module.SKILLS:
                pre = float(pre_by_id[fid][skill])
                obs, q = bundle[skill]
                if obs is None or q <= 0.0:
                    post = pre
                else:
                    baseline = module.population_baseline(ctx["weighted"], ctx["quality"], skill)
                    if skill.endswith("_pressure"):
                        expected = module.expected_intrinsic(pre, baseline)
                    elif skill.endswith("_precision"):
                        defense_skill = skill.replace("_precision", "_defense")
                        expected = module.expected_matchup(pre, float(pre_by_id[oid][defense_skill]), baseline)
                    else:
                        precision_skill = skill.replace("_defense", "_precision")
                        expected = module.expected_matchup(pre, float(pre_by_id[oid][precision_skill]), baseline)
                    delta = module.k_factor(ctx["updates"][fid][skill]) * q * (float(obs) - expected)
                    post = module.clamp(pre + delta, module.MIN_RATING, module.MAX_RATING)
                out[fid][skill] = float(post)
        return out

    def _update_distance(self, target, pre_by_id, date) -> dict[str, dict[str, float]]:
        ctx = self._build_distance_context(date)
        out = defaultdict(dict)
        for _, row in target.iterrows():
            fid = str(row["fighter_id"])
            pre = float(pre_by_id[fid][distance.SKILL])
            obs, q = distance.observation(row, ctx["pools"])
            if obs is None or q <= 0.0:
                post = pre
            else:
                baseline = distance.population_baseline(ctx["weighted"], ctx["quality"])
                expected = distance.expected_intrinsic(pre, baseline)
                delta = distance.k_factor(ctx["updates"][fid]) * q * (float(obs) - expected)
                post = distance.clamp(pre + delta, distance.MIN_RATING, distance.MAX_RATING)
            out[fid][distance.SKILL] = float(post)
        return out

    def _update_reversal(self, target, pre_by_id, date) -> dict[str, dict[str, float]]:
        ctx = self._build_reversal_context(date)
        out = defaultdict(dict)
        rows = [row for _, row in target.iterrows()]
        for idx, row in enumerate(rows):
            opp = rows[1 - idx]
            fid = str(row["fighter_id"])
            oid = str(opp["fighter_id"])
            pre = float(pre_by_id[fid][reversal.SKILL])
            revs = reversal.finite(row.get(reversal.C["reversals"])) or 0.0
            ctrl = reversal.finite(row.get(reversal.C["opp_control_seconds"])) or 0.0
            obs, q, _ = reversal.observation(revs, ctrl, ctx["pool"])
            if obs is None or q <= 0.0:
                post = pre
            else:
                baseline = reversal.population_baseline(ctx["weighted"], ctx["quality"])
                expected = reversal.expected_matchup(
                    pre,
                    float(pre_by_id[oid]["control_imposition"]),
                    baseline,
                )
                delta = reversal.k_factor(ctx["updates"][fid]) * q * (float(obs) - expected)
                post = reversal.clamp(pre + delta, reversal.MIN_RATING, reversal.MAX_RATING)
            out[fid][reversal.SKILL] = float(post)
        return out

    # ------------------------------------------------------------------
    # Reservoir: reconstruct compact raw-exposure states, not rating replay.
    # Post value is the state that would be snapshotted on the next date.
    # ------------------------------------------------------------------
    def _reservoir_post_for_date(self, date: pd.Timestamp) -> dict[str, tuple[float, float]]:
        if date in self._reservoir_post_cache:
            return self._reservoir_post_cache[date]
        work = self.rfs.loc[self.rfs["date"] <= date].copy()
        for col in (
            reservoir.KD_COL,
            reservoir.SIG_ABS_COL,
            reservoir.HEAD_ABS_COL,
            reservoir.GROUND_ABS_COL,
            reservoir.OPP_CTRL_COL,
            reservoir.ROUNDS_COL,
            reservoir.KO_LOSS_COL,
        ):
            work[col] = pd.to_numeric(work[col], errors="coerce")
        work["_damage_exposure"] = work.apply(reservoir._damage_exposure, axis=1)

        states = defaultdict(reservoir._new_state)
        prior_sig: list[float] = []
        prior_damage: list[float] = []
        for fight_date, day in work.groupby("date", sort=True):
            kd_threshold = reservoir._quantile(prior_sig, reservoir.KD_HIGH_EXPOSURE_QUANTILE)
            dur_threshold = reservoir._quantile(prior_damage, reservoir.DURABILITY_HIGH_EXPOSURE_QUANTILE)
            for _, row in day.iterrows():
                fid = str(row["fighter_id"])
                state = states[fid]
                kd_abs = max(0.0, reservoir._finite(row.get(reservoir.KD_COL)))
                sig_abs = max(0.0, reservoir._finite(row.get(reservoir.SIG_ABS_COL)))
                exposure = max(0.0, reservoir._finite(row.get("_damage_exposure")))
                ko_loss = 1.0 if reservoir._finite(row.get(reservoir.KO_LOSS_COL)) >= 0.5 else 0.0
                state["fights"] += 1.0
                state["kd_absorbed"] += kd_abs
                state["sig_absorbed"] += sig_abs
                if kd_abs <= 0:
                    state["kd_free_fights"] += 1.0
                if kd_threshold is not None and sig_abs >= kd_threshold:
                    state["kd_high_exposure_fights"] += 1.0
                    if kd_abs <= 0:
                        state["kd_free_high_exposure"] += 1.0
                if dur_threshold is not None and exposure >= dur_threshold:
                    state["dur_high_exposure_fights"] += 1.0
                    state["dur_high_exposure_sum"] += exposure
                    if ko_loss < 0.5:
                        state["dur_high_survivals"] += 1.0
                        state["dur_high_survived_exposure_sum"] += exposure
                if ko_loss < 0.5:
                    state["survived_fights"] += 1.0
                    state["survived_exposure_sum"] += exposure
                else:
                    state["ko_losses"] += 1.0
                prior_sig.append(sig_abs)
                prior_damage.append(exposure)

        next_dur_threshold = reservoir._quantile(prior_damage, reservoir.DURABILITY_HIGH_EXPOSURE_QUANTILE)
        peer_avoidance: list[float] = []
        peer_free: list[float] = []
        peer_high: list[float] = []
        for state in states.values():
            if state["fights"] <= 0:
                continue
            a, f, h = reservoir._kd_raw_components(state)
            peer_avoidance.append(a)
            peer_free.append(f)
            if h is not None:
                peer_high.append(h)

        result: dict[str, tuple[float, float]] = {}
        for fid, state in states.items():
            fights = int(state["fights"])
            kd_score = reservoir._kd_score(state, peer_avoidance, peer_free, peer_high)
            dur_score = reservoir._durability_score(state, next_dur_threshold)
            result[str(fid)] = (
                reservoir._rating_from_score(kd_score, fights),
                reservoir._rating_from_score(dur_score, fights),
            )
        self._reservoir_post_cache[date] = result
        return result

    def _power_post(self, fighter_id: str, fight_id: str) -> float:
        if fight_id not in self._fight_order_map:
            raise RuntimeError(f"fight {fight_id} missing from master power chronology")
        target_order = int(self._fight_order_map[fight_id])
        hist = self.power_evidence.loc[
            self.power_evidence["fighter_id"].astype(str).eq(str(fighter_id))
            & self.power_evidence["fight_order"].astype(int).le(target_order)
        ].sort_values("fight_order")
        if hist.empty:
            return float(power32.POWER_NEUTRAL)

        cumulative_opportunity = 0.0
        cumulative_events = 0
        peak = 0.0
        running_positive = power32.POWER_NEUTRAL
        state = power32.POWER_NEUTRAL
        for i, row in enumerate(hist.itertuples(index=False), start=1):
            cumulative_opportunity += float(row.opportunity)
            if bool(row.power_event):
                cumulative_events += 1
                peak = max(peak, float(row.fight_power_evidence_v8))
            if cumulative_events == 0:
                state = power32._low_end_power_rating(cumulative_opportunity)
            else:
                candidate = power32._positive_power_rating(
                    prior_fights=i,
                    prior_power_events=cumulative_events,
                    peak_single_fight_evidence=peak,
                )
                running_positive = max(running_positive, candidate)
                state = running_positive
        return float(np.clip(state, power32.POWER_MIN, power32.POWER_MAX))

    def calculate(self, fight_id: str) -> pd.DataFrame:
        fight_id = str(fight_id)
        date, target, pre = self._target(fight_id)
        pre_by_id = {str(row["fighter_id"]): row for _, row in pre.iterrows()}
        fighter_ids = list(pre_by_id)

        print(
            f"[postfight FSR] {fight_id} | {date.date()} | direct one-fight update for "
            + ", ".join(str(pre_by_id[fid].get("fighter_name", fid)) for fid in fighter_ids),
            flush=True,
        )

        families = [
            self._update_locked(target, pre_by_id, date),
            self._update_dynamic(target, pre_by_id, date),
            self._update_three_skill(target, pre_by_id, date, ground, self._ground_context),
            self._update_reversal(target, pre_by_id, date),
            self._update_three_skill(target, pre_by_id, date, clinch, self._clinch_context),
            self._update_distance(target, pre_by_id, date),
        ]
        reservoir_post = self._reservoir_post_for_date(date)

        rows: list[dict[str, object]] = []
        for fid in fighter_ids:
            pre_row = pre_by_id[fid]
            post_values = {trait: float(pre_row[trait]) for trait in canonical.CANONICAL_RATINGS}
            for family in families:
                post_values.update(family[fid])
            if fid not in reservoir_post:
                raise RuntimeError(f"reservoir post state missing fighter {fid}")
            kd, dur = reservoir_post[fid]
            post_values["knockdown_resistance"] = float(kd)
            post_values["damage_durability"] = float(dur)
            post_values["striking_power"] = self._power_post(fid, fight_id)

            row: dict[str, object] = {
                "fight_id": fight_id,
                "date": date,
                "fighter_id": fid,
                "fighter_name": str(pre_row.get("fighter_name", fid)),
                "prior_ufc_fights_pre": int(pre_row.get("prior_ufc_fights", 0)),
            }
            for trait in canonical.CANONICAL_RATINGS:
                pv = float(pre_row[trait])
                qv = float(post_values[trait])
                row[f"{trait}_pre"] = pv
                row[f"{trait}_post"] = qv
                row[f"{trait}_delta"] = qv - pv
            rows.append(row)
        return pd.DataFrame(rows)


def validate_against_explicit(
    calculated: pd.DataFrame,
    explicit_traits_csv: Path,
    *,
    atol: float = 1e-6,
) -> pd.DataFrame:
    """Compare a direct one-fight update against a previously replayed sentinel result."""
    explicit = pd.read_csv(explicit_traits_csv)
    explicit["fighter_id"] = explicit["fighter_id"].astype(str)
    calc = calculated.copy()
    calc["fighter_id"] = calc["fighter_id"].astype(str)
    rows = []
    for row in explicit.itertuples(index=False):
        trait = str(row.trait)
        fid = str(row.fighter_id)
        if trait not in canonical.CANONICAL_RATINGS:
            continue
        match = calc.loc[calc["fighter_id"].eq(fid)]
        if len(match) != 1:
            raise RuntimeError(f"validation fighter {fid} missing from direct result")
        direct = float(match.iloc[0][f"{trait}_post"])
        expected = float(row.post_fsr)
        rows.append(
            {
                "fighter_id": fid,
                "fighter_name": str(row.fighter_name),
                "trait": trait,
                "direct_post": direct,
                "sentinel_post": expected,
                "abs_error": abs(direct - expected),
                "ok": abs(direct - expected) <= atol,
            }
        )
    return pd.DataFrame(rows)


if __name__ == "__main__":
    import argparse

    p = argparse.ArgumentParser()
    p.add_argument("--fight-id", required=True)
    p.add_argument("--output", type=Path)
    p.add_argument("--validate-explicit", type=Path)
    p.add_argument("--atol", type=float, default=1e-6)
    args = p.parse_args()

    calc = PostfightFSRCalculator()
    result = calc.calculate(args.fight_id)
    output = args.output or Path(
        "data/experimental/postfight_fsr_direct/"
        f"{args.fight_id}_pre_post.csv"
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    result.to_csv(output, index=False)
    print(f"[postfight FSR] wrote {output}", flush=True)

    if args.validate_explicit:
        check = validate_against_explicit(result, args.validate_explicit, atol=args.atol)
        bad = check.loc[~check["ok"]]
        print(
            f"[postfight FSR] validation max_abs_error={check['abs_error'].max():.9f} | "
            f"passed={len(check)-len(bad)}/{len(check)}",
            flush=True,
        )
        if not bad.empty:
            print(bad.sort_values("abs_error", ascending=False).head(20).to_string(index=False), flush=True)
            raise SystemExit(2)
        print("[postfight FSR] VALIDATED: direct one-fight update matches sentinel", flush=True)
