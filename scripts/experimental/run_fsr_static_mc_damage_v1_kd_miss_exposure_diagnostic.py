"""Compatibility runner for the KD-miss exposure diagnostic.

The frozen V0/Damage V1 stats contract names significant-strike attempts
``sig_att``. The diagnostic script originally referenced ``sig_attempted``.
This runner adds a read-only compatibility alias at runtime and then executes
the diagnostic without changing simulator mechanics or persisted artifacts.
"""

from __future__ import annotations

from scripts.experimental import fsr_static_mc_damage_v1 as damage


# Research-script compatibility only. Do not modify the frozen simulator schema.
if not hasattr(damage.DamageFighterStats, "sig_attempted"):
    damage.DamageFighterStats.sig_attempted = property(lambda self: self.sig_att)

from scripts.experimental import fsr_static_mc_damage_v1_kd_miss_exposure_diagnostic as audit


if __name__ == "__main__":
    audit.main()
