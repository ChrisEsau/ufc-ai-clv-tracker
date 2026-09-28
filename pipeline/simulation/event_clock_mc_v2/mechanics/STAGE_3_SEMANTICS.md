# Stage 3 MVP action semantics

Stage 3 resolves attempts into immutable outcomes and transition requests. It
does not advance time, charge stamina, mutate fight state, mutate the phase
timeline, or apply a finish. Time/stamina accounting belongs to the later causal
engine. These semantics must be revisited only through an approved later stage.

| Action family | Classification | Stage 3 physical meaning | Random? | Transition/consequence |
| --- | --- | --- | --- | --- |
| `STAND_ATTACK` | Direct | Standing strike attempt using matchup standing accuracy | Yes | Landing observation only |
| `STAND_COUNTER` | Direct/MVP | Counter-labeled standing strike; reactive eligibility is deferred | Yes | Landing observation only |
| `PRESSURE` | Tactical/MVP | Advances pressure without fabricating an attack | No | None |
| `RESET_RANGE` | Tactical/MVP | Range-reset movement without fabricating an attack | No | None |
| `CLINCH_ENTRY` | Direct/MVP | Contested entry under neutral clinch assumption | Yes | Successful entry requests RED/BLUE-controlled clinch |
| `TAKEDOWN_ENTRY` | Direct | Open takedown using matchup TD completion | Yes | Successful completion requests actor-controlled ground |
| `CLINCH_STRIKE` | Direct/MVP | Close-range strike under neutral clinch accuracy | Yes | Landing observation only |
| `CLINCH_CONTROL` | Maintenance/MVP | Maintains clinch without fabricated duration or damage | No | `CONTROLLED`; no transition |
| `CLINCH_TAKEDOWN` | Direct/MVP | Clinch takedown using matchup TD completion without a clinch modifier | Yes | Successful completion requests actor-controlled ground |
| `BREAK_CLINCH` | Direct/MVP | Contested separation under neutral assumption | Yes | Success requests standing |
| `GROUND_STRIKE` | Direct | Top strike using matchup ground accuracy | Yes | Landing observation only |
| `ADVANCE_POSITION` | Maintenance/MVP | Position-improvement intent; detailed position is not represented | No | `MAINTAINED`; no transition |
| `SUBMISSION_ATTACK` | Direct | Submission attempt using explicit matchup success input | Yes | Success requests typed `FinishMethod.SUBMISSION` termination; state is not mutated |
| `CONTROL` | Maintenance/MVP | Maintains top control without fabricated duration or damage | No | `CONTROLLED`; no transition |
| `DISENGAGE` | Tactical | Top fighter deliberately returns the fight to standing | No | Requests standing |
| `ESCAPE_STAND` | Direct | Bottom escape using explicit matchup escape probability | Yes | Success requests standing |
| `IMPROVE_POSITION` | Maintenance/MVP | Bottom position-improvement intent without detailed position | No | `MAINTAINED`; no transition |
| `REVERSAL` | Direct | Bottom reversal using explicit matchup reversal probability | Yes | Success requests ground with actor as controller |
| `BOTTOM_STRIKE` | Direct/MVP | Bottom strike using the same approved ground-accuracy input | Yes | Landing observation only |

## Preserved inputs and deferred mechanics

Standing accuracy, takedown completion, and ground accuracy consume the existing
directional FSR V3 runtime quantities directly; Stage 3 does not transform or
retune them. Submission, escape, and reversal probabilities are explicit inputs
because their legacy implementations cannot be reused without legacy profiles
or mutable legacy state.

Legacy damage, trauma, knockdown, KO/TKO, and submission engine objects require
physiology/profile state absent from the approved causal `FightState`. They are
therefore not wrapped or invoked here. A later approved mechanics integration
must compose those calculations with explicit causal physiology deltas rather
than introduce a second authoritative state.

All unsupported clinch assumptions are centralized in
`StructuralMVPPlaceholders`: entry `0.50`, strike landing `0.50`, and break
success `0.50`. These values are structural and uncalibrated.
