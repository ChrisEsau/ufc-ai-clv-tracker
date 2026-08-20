"""Empirical shadow KO/KD consequence model for EVENT CLOCK MC V1.

Architecture
------------
For every landed significant strike:

    1. Roll KO/TKO from the PRE-STRIKE state.
       If KO/TKO occurs, the fight ends.

    2. Otherwise roll KD.
       If KD occurs, record it and continue the fight.

A KD therefore cannot also be the KO strike.  A survived KD changes the
state seen by subsequent landed strikes.

This module intentionally does NOT use:
- cumulative trauma
- damage durability
- acute vulnerability
- same-strike KD -> KO conversion

The purpose of this shadow model is empirical calibration against historical
KD/strike, KO/strike, KO share, KD counts, and fighter allocation.
"""

from __future__ import annotations

from dataclasses import dataclass
from math import exp, log

import numpy as np

from pipeline.simulation.event_mc_v1.components.profiles import (
    MatchupProfiles,
    Side,
)


def _sigmoid(x: float) -> float:
    x = float(np.clip(x, -30.0, 30.0))
    return 1.0 / (1.0 + exp(-x))


def _logit(p: float) -> float:
    p = float(np.clip(p, 1e-12, 1.0 - 1e-12))
    return log(p / (1.0 - p))


@dataclass(frozen=True)
class ShadowKOKDCalibration:
    # ------------------------------------------------------------------
    # KD hazard
    #
    # Walk-forward round-state study, 2022-2026 training windows:
    #
    # attacker_power        +0.020741
    # attacker_age30        -0.030660
    # defender_kdres        -0.014421
    # defender_age30        +0.029369
    # prior_kd_scored       +0.502627
    #
    # Historical unconditional KD event rate / landed sig:
    # ~0.5594% in the aligned modeling frame.
    #
    # Intercept is expressed around centered ratings:
    # power=50, KDRES=50, age=30, no prior KD, time=0.
    # It will be calibrated in replay rather than assumed final.
    # ------------------------------------------------------------------

    kd_base_probability: float = 0.00640

    kd_power_beta: float = 0.020741
    kd_attacker_age_beta: float = -0.030660
    kd_kdres_beta: float = -0.014421
    kd_defender_age_beta: float = 0.029369
    kd_prior_kd_beta: float = 0.502627

    # Elapsed-time effect from the dynamic study.  This is retained in
    # shadow v0 because historical elapsed time added out-of-sample signal.
    kd_elapsed_minute_beta: float = -0.050585

    # Optional stamina term.  Zero initially because historical workload
    # proxies do not map directly to Event Clock's [0,1] stamina variable.
    # We calibrate this later rather than inventing a conversion.
    kd_stamina_beta: float = 0.0

    # ------------------------------------------------------------------
    # KO/TKO hazard
    #
    # Validated Event Clock shadow calibration.
    #
    # Historical replay calibration selected:
    #   KO base probability   = 0.00250
    #   prior-KD beta         = +1.00
    #   elapsed-time beta     = 0.00
    #
    # Power and fighter-age terms were retained through the orthogonal
    # ablation study.  Explicit elapsed time was removed because it reduced
    # KO-side discrimination; prior KD remains the dynamic vulnerability
    # state for subsequent landed strikes.
    #
    # On the 500-fight x 20-path fresh replay:
    #   realized KD / landed sig = 0.56445%  (hist 0.55945%)
    #   simulated KO share       = 33.60%    (hist 33.60%)
    #   fighter KO-winner AUC    = 0.6338
    # ------------------------------------------------------------------

    ko_base_probability: float = 0.00250

    ko_power_beta: float = 0.0200
    ko_attacker_age_beta: float = -0.0300
    ko_defender_age_beta: float = 0.0300

    ko_prior_kd_beta: float = 1.00

    # No independent elapsed-time KO penalty in the validated architecture.
    ko_elapsed_minute_beta: float = 0.0

    # Reserved for a future explicit Event Clock stamina study.
    ko_stamina_beta: float = 0.0


@dataclass(frozen=True)
class ShadowStrikeConsequence:
    attacker: Side
    defender: Side

    ko_probability: float
    ko_tko: bool

    kd_probability: float
    knockdown: bool

    prior_defender_kds: int


@dataclass(frozen=True)
class EventClockShadowKOKDModel:
    profiles: MatchupProfiles
    calibration: ShadowKOKDCalibration = ShadowKOKDCalibration()

    def _stamina(self, state, side: Side) -> float:
        value = (
            state.red_stamina
            if side is Side.RED
            else state.blue_stamina
        )
        return float(np.clip(value, 1e-6, 1.0))

    def kd_probability(
        self,
        *,
        state,
        attacker: Side,
        prior_defender_kds: int,
    ) -> float:
        defender = attacker.opponent

        a = self.profiles.fighter(attacker)
        d = self.profiles.fighter(defender)
        c = self.calibration

        elapsed_minutes = (
            float(state.fight_time_seconds) / 60.0
        )

        attacker_stamina = self._stamina(
            state,
            attacker,
        )

        z = _logit(c.kd_base_probability)

        z += (
            c.kd_power_beta
            * (float(a.striking_power) - 50.0)
        )

        z += (
            c.kd_attacker_age_beta
            * (float(a.age_years) - 30.0)
        )

        z += (
            c.kd_kdres_beta
            * (float(d.knockdown_resistance) - 50.0)
        )

        z += (
            c.kd_defender_age_beta
            * (float(d.age_years) - 30.0)
        )

        z += (
            c.kd_prior_kd_beta
            * float(prior_defender_kds)
        )

        z += (
            c.kd_elapsed_minute_beta
            * elapsed_minutes
        )

        # Zero by default in shadow v0.
        # log(stamina) behaves naturally on a multiplicative reservoir.
        z += (
            c.kd_stamina_beta
            * log(attacker_stamina)
        )

        return _sigmoid(z)

    def ko_probability(
        self,
        *,
        state,
        attacker: Side,
        prior_defender_kds: int,
    ) -> float:
        defender = attacker.opponent

        a = self.profiles.fighter(attacker)
        d = self.profiles.fighter(defender)
        c = self.calibration

        elapsed_minutes = (
            float(state.fight_time_seconds) / 60.0
        )

        attacker_stamina = self._stamina(
            state,
            attacker,
        )

        z = _logit(c.ko_base_probability)

        z += (
            c.ko_power_beta
            * (float(a.striking_power) - 50.0)
        )

        z += (
            c.ko_attacker_age_beta
            * (float(a.age_years) - 30.0)
        )

        z += (
            c.ko_defender_age_beta
            * (float(d.age_years) - 30.0)
        )

        z += (
            c.ko_prior_kd_beta
            * float(prior_defender_kds)
        )

        z += (
            c.ko_elapsed_minute_beta
            * elapsed_minutes
        )

        z += (
            c.ko_stamina_beta
            * log(attacker_stamina)
        )

        return _sigmoid(z)

    def resolve_landed_strike(
        self,
        *,
        state,
        attacker: Side,
        prior_defender_kds: int,
        rng,
    ) -> ShadowStrikeConsequence:
        """Resolve one landed strike.

        Ordering is intentionally:

            KO/TKO first
            then KD only if KO/TKO did not occur

        Therefore the same strike can never be both KD and KO/TKO.
        """

        defender = attacker.opponent

        p_ko = self.ko_probability(
            state=state,
            attacker=attacker,
            prior_defender_kds=prior_defender_kds,
        )

        ko_tko = bool(
            rng.random() < p_ko
        )

        if ko_tko:
            return ShadowStrikeConsequence(
                attacker=attacker,
                defender=defender,
                ko_probability=p_ko,
                ko_tko=True,
                kd_probability=0.0,
                knockdown=False,
                prior_defender_kds=int(
                    prior_defender_kds
                ),
            )

        p_kd = self.kd_probability(
            state=state,
            attacker=attacker,
            prior_defender_kds=prior_defender_kds,
        )

        knockdown = bool(
            rng.random() < p_kd
        )

        return ShadowStrikeConsequence(
            attacker=attacker,
            defender=defender,
            ko_probability=p_ko,
            ko_tko=False,
            kd_probability=p_kd,
            knockdown=knockdown,
            prior_defender_kds=int(
                prior_defender_kds
            ),
        )


