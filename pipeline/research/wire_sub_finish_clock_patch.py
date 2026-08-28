from pathlib import Path


def replace_once(text: str, old: str, new: str, label: str) -> str:
    if old not in text:
        raise RuntimeError(f"missing patch anchor: {label}")
    return text.replace(old, new, 1)


def patch_tick_clock() -> None:
    path = Path("pipeline/research/locked_brain_tick_clock.py")
    s = path.read_text()
    s = replace_once(s, "Every tick first evaluates the validated time-survival KO/TKO competing risk, then,\nif the fight survives, evaluates the currently admissible Brain actions.", "Every tick first evaluates the validated KO/TKO and submission time-survival competing risks, then,\nif the fight survives, evaluates the currently admissible Brain actions.", "docstring")
    s = replace_once(s, "KO_HAZARDS_BY_SIDE: dict[Side, np.ndarray] = {}\nKO_FIGHTER_NAMES_BY_SIDE: dict[Side, str] = {}", "KO_HAZARDS_BY_SIDE: dict[Side, np.ndarray] = {}\nSUB_HAZARDS_BY_SIDE: dict[Side, np.ndarray] = {}\nKO_FIGHTER_NAMES_BY_SIDE: dict[Side, str] = {}\nSUB_FIGHTER_NAMES_BY_SIDE: dict[Side, str] = {}", "finish globals")
    s = replace_once(s, "    ko_hazards_by_side=None,\n    ko_fighter_names_by_side=None,\n):\n    global STANDING_RATE_FN, GROUND_RATE_BY_SIDE, GROUND_BURST_BY_SIDE\n    global KO_HAZARDS_BY_SIDE, KO_FIGHTER_NAMES_BY_SIDE", "    ko_hazards_by_side=None,\n    ko_fighter_names_by_side=None,\n    sub_hazards_by_side=None,\n    sub_fighter_names_by_side=None,\n):\n    global STANDING_RATE_FN, GROUND_RATE_BY_SIDE, GROUND_BURST_BY_SIDE\n    global KO_HAZARDS_BY_SIDE, SUB_HAZARDS_BY_SIDE, KO_FIGHTER_NAMES_BY_SIDE, SUB_FIGHTER_NAMES_BY_SIDE", "configure signature")
    s = replace_once(s, "    KO_FIGHTER_NAMES_BY_SIDE = {\n        side: str(value) for side, value in (ko_fighter_names_by_side or {}).items()\n    }", "    KO_FIGHTER_NAMES_BY_SIDE = {\n        side: str(value) for side, value in (ko_fighter_names_by_side or {}).items()\n    }\n    SUB_HAZARDS_BY_SIDE = {\n        side: np.asarray(value, dtype=float) for side, value in (sub_hazards_by_side or {}).items()\n    }\n    SUB_FIGHTER_NAMES_BY_SIDE = {\n        side: str(value) for side, value in (sub_fighter_names_by_side or {}).items()\n    }", "configure body")

    start = s.index("def _ko_tick_probability(")
    end = s.index("\ndef _collision_policy_weights", start)
    finish_fn = '''def _finish_tick_probability(tick_end: float, exposure_seconds: float, rng):
    if not KO_HAZARDS_BY_SIDE or not SUB_HAZARDS_BY_SIDE:
        return None, None, None
    if set(KO_HAZARDS_BY_SIDE) != {Side.RED, Side.BLUE} or set(SUB_HAZARDS_BY_SIDE) != {Side.RED, Side.BLUE}:
        raise RuntimeError("embedded finish clock requires KO and SUB hazards for both sides")
    cause_hazards = {}
    for method, source in ((FinishMethod.KO_TKO, KO_HAZARDS_BY_SIDE), (FinishMethod.SUBMISSION, SUB_HAZARDS_BY_SIDE)):
        for side in Side:
            pieces = source[side]
            if pieces.size < 1:
                raise RuntimeError(f"empty {method.value} hazard vector for {side.value}")
            cause_hazards[(method, side)] = max(float(pieces[_ko_piece_index(tick_end, len(pieces))]), 0.0)
    total_hazard = float(sum(cause_hazards.values()))
    exposure = max(float(exposure_seconds), 0.0)
    any_probability = float(1.0 - math.exp(-total_hazard * exposure)) if total_hazard > 0 else 0.0
    any_draw = float(rng.random())
    winner = None
    finish_method = None
    cause_draw = None
    if any_draw < any_probability:
        cause_draw = float(rng.random())
        threshold = cause_draw * total_hazard
        cumulative = 0.0
        for method, side in ((FinishMethod.KO_TKO, Side.RED), (FinishMethod.KO_TKO, Side.BLUE), (FinishMethod.SUBMISSION, Side.RED), (FinishMethod.SUBMISSION, Side.BLUE)):
            cumulative += cause_hazards[(method, side)]
            if threshold <= cumulative:
                winner, finish_method = side, method
                break
    trace = {"ko": {}, "sub": {}, "any_finish_probability_in_interval": any_probability, "any_finish_draw": any_draw, "cause_draw_if_finish": cause_draw}
    for method, label, names in ((FinishMethod.KO_TKO, "ko", KO_FIGHTER_NAMES_BY_SIDE), (FinishMethod.SUBMISSION, "sub", SUB_FIGHTER_NAMES_BY_SIDE)):
        for side in Side:
            hazard = cause_hazards[(method, side)]
            trace[label][side.value] = {
                "fighter": names.get(side),
                "hazard_per_second": hazard,
                "exposure_seconds": exposure,
                "probability_in_interval": any_probability * hazard / total_hazard if total_hazard > 0 else 0.0,
                "probability_next_1s": (1.0 - math.exp(-total_hazard)) * hazard / total_hazard if total_hazard > 0 else 0.0,
                "any_finish_probability_in_interval": any_probability,
                "any_finish_draw": any_draw,
                "cause_draw_if_finish": cause_draw,
                "sampled_clock_time": float(tick_end) if winner is side and finish_method is method else None,
                "fires_in_this_tick_interval": winner is side and finish_method is method,
            }
    return winner, finish_method, trace

'''
    s = s[:start] + finish_fn + s[end + 1:]

    start = s.index("def _append_tick_trace(")
    end = s.index("\ndef _append_trace_decision", start)
    trace_fn = '''def _append_tick_trace(brain, state, diagnostics, candidates, selected, exposure_seconds, finish_trace=None, finish_winner=None, finish_method=None):
    if brain is None:
        return
    if not hasattr(brain, "tick_trace"):
        brain.tick_trace = []
    candidate_lookup = {(row["actor"].value, row["action"].value): row for row in candidates}
    traced_options = []
    for diagnostic in diagnostics:
        row = dict(diagnostic)
        candidate = candidate_lookup.get((row.get("actor"), row.get("action")))
        row["collision_weight"] = None if candidate is None else candidate.get("collision_weight")
        row["selection_probability_given_available"] = None if candidate is None else candidate.get("collision_probability")
        traced_options.append(row)
    finish_event = finish_winner is not None
    finish_action = None if not finish_event else ("ko_clock" if finish_method is FinishMethod.KO_TKO else "sub_clock")
    finish_trace = finish_trace or {}
    brain.tick_trace.append({
        "tick": len(brain.tick_trace) + 1,
        "timestamp": float(state.fight_time_seconds),
        "exposure_seconds": float(exposure_seconds),
        "round": int(state.round_number),
        "phase": state.phase.value,
        "ground_controller": None if state.ground_controller is None else state.ground_controller.value,
        "clinch_controller": None if state.clinch_controller is None else state.clinch_controller.value,
        "options": traced_options,
        "available_count": len(candidates),
        "collision": len(candidates) > 1,
        "collision_rule": "embedded_finish_competing_risk_first_then_brain_policy_weights_among_available",
        "selected_actor": finish_winner.value if finish_event else (None if selected is None else selected["actor"].value),
        "selected_action": finish_action if finish_event else (None if selected is None else selected["action"].value),
        "selected_probability_given_available": 1.0 if finish_event else (None if selected is None else float(selected["collision_probability"])),
        "finish_clock_event": finish_event,
        "finish_clock_method": None if finish_method is None else finish_method.value,
        "ko_clock_event": finish_event and finish_method is FinishMethod.KO_TKO,
        "sub_clock_event": finish_event and finish_method is FinishMethod.SUBMISSION,
        "any_finish_probability_in_interval": finish_trace.get("any_finish_probability_in_interval"),
        "any_finish_draw": finish_trace.get("any_finish_draw"),
        "cause_draw_if_finish": finish_trace.get("cause_draw_if_finish"),
        "ko": finish_trace.get("ko", {}),
        "sub": finish_trace.get("sub", {}),
    })

'''
    s = s[:start] + trace_fn + s[end + 1:]

    s = replace_once(s, "        return resolve_action(event, state, inputs, rng, placeholders, ko_kd_rng, submission_rng)", '''        resolved = resolve_action(event, state, inputs, rng, placeholders, ko_kd_rng, submission_rng)
        if event.action_family is ActionFamily.SUBMISSION_ATTACK and isinstance(resolved.consequence, SubmissionConsequence):
            return ActionResolution(event, ActionOutcome.FAILURE, consequence=SubmissionConsequence(attempted=True, conversion_probability=float(resolved.consequence.conversion_probability), success=False, termination=None))
        return resolved''', "nonterminal submission attempts")
    s = replace_once(s, "    ko_rng = np.random.default_rng((int(seed) ^ 0x4B4F425241494E) & ((1 << 63) - 1))", "    finish_rng = np.random.default_rng((int(seed) ^ 0x46494E495348) & ((1 << 63) - 1))", "finish rng")
    s = replace_once(s, "        ko_winner, ko_trace = _ko_tick_probability(next_tick, exposure_seconds, ko_rng)", "        finish_winner, finish_method, finish_trace = _finish_tick_probability(next_tick, exposure_seconds, finish_rng)", "finish draw")
    s = replace_once(s, '''        if ko_winner is not None:
            termination = FightTerminationRequest(ko_winner, FinishMethod.KO_TKO)
            state = replace(state, finished=True, winner=ko_winner, finish_method=FinishMethod.KO_TKO.value)
            _append_tick_trace(brain, state, [], [], None, exposure_seconds, ko_trace, ko_winner)
            break''', '''        if finish_winner is not None:
            termination = FightTerminationRequest(finish_winner, finish_method)
            state = replace(state, finished=True, winner=finish_winner, finish_method=finish_method.value)
            _append_tick_trace(brain, state, [], [], None, exposure_seconds, finish_trace, finish_winner, finish_method)
            break''', "finish termination")
    s = replace_once(s, "_append_tick_trace(brain, state, diagnostics, candidates, chosen, exposure_seconds, ko_trace, None)", "_append_tick_trace(brain, state, diagnostics, candidates, chosen, exposure_seconds, finish_trace, None, None)", "surviving tick trace")
    s = replace_once(s, "run_causal_path.embedded_ko_clock = True", "run_causal_path.embedded_ko_clock = True\nrun_causal_path.embedded_sub_clock = True\nrun_causal_path.embedded_finish_clock = True", "clock attrs")
    path.write_text(s)


def patch_harness() -> None:
    path = Path("pipeline/research/locked_brain_mc.py")
    s = path.read_text()
    s = replace_once(s, "KO/TKO time-survival hazard is evaluated inside that clock before ordinary Brain\nactions.", "KO/TKO and submission time-survival hazards are evaluated together inside that clock before ordinary Brain\nactions.", "harness docstring")
    s = replace_once(s, "from pipeline.research import locked_brain_tick_clock as tick_clock", "from pipeline.research import locked_brain_tick_clock as tick_clock\nfrom pipeline.research import sub_time_survival_locked_clock as sub_time_clock", "sub clock import")
    s = replace_once(s, '    "pipeline/research/allen_shahbazyan_decision_scored_outputs_2000.py": "6c23e1765b941c082c535588bf7a41b53fc6516d",', '    "pipeline/research/allen_shahbazyan_decision_scored_outputs_2000.py": "6c23e1765b941c082c535588bf7a41b53fc6516d",\n    "pipeline/research/sub_time_survival_oos.py": "0738ddb476266cc63cb5b1e17eb115d13ce84e6c",\n    "pipeline/research/sub_time_survival_locked_clock.py": "49daf21287988a28c1e1ca2e8ebe4d1e21f34ded",', "lock sub sources")
    s = replace_once(s, '''            _, _, _, ko_clock = time_ko._time_clock_inputs()
            ko_hazards = {side: np.asarray(ko_clock[names[side]]["hazards_per_second"], dtype=float) for side in Side}
            tick_clock.configure(''', '''            _, _, _, ko_clock = time_ko._time_clock_inputs()
            ko_hazards = {side: np.asarray(ko_clock[names[side]]["hazards_per_second"], dtype=float) for side in Side}
            sub_cutoff, sub_p0, sub_baseline_piece, sub_clock = sub_time_clock.time_clock_inputs(fight_id)
            sub_hazards = {side: np.asarray(sub_clock[names[side]]["hazards_per_second"], dtype=float) for side in Side}
            tick_clock.configure(''', "build sub hazards")
    s = replace_once(s, "                ko_fighter_names_by_side=names,\n            )", "                ko_fighter_names_by_side=names,\n                sub_hazards_by_side=sub_hazards,\n                sub_fighter_names_by_side=names,\n            )", "configure sub hazards")
    s = replace_once(s, '"clock_architecture": "one global 1-second probability clock with embedded KO competing risk plus state-legal action availability",', '"clock_architecture": "one global 1-second probability clock with embedded KO/TKO + submission four-cause competing risk plus state-legal action availability",', "manifest clock")
    s = replace_once(s, '                "ko_hazards_per_second": {names[side]: ko_hazards[side].tolist() for side in Side},\n                "kd": "OOS-selected static prefight KD hazard; no within-fight KD escalation",\n                "submission": "OOS-selected fighter-level submission attempt rate mapped to relevant ground opportunity; conversion unchanged",', '                "ko_hazards_per_second": {names[side]: ko_hazards[side].tolist() for side in Side},\n                "submission_finish_clock": "OOS-selected fighter piecewise-round time-survival hazard, prior_events=1.0, embedded as terminal competing risk independent of phase/ground time",\n                "submission_finish_cutoff": str(sub_cutoff.date()),\n                "submission_population_hazard_per_second": sub_p0,\n                "submission_baseline_hazard_per_second": sub_baseline_piece.tolist(),\n                "submission_hazards_per_second": {names[side]: sub_hazards[side].tolist() for side in Side},\n                "submission_matchup_inputs": {names[side]: sub_clock[names[side]] for side in Side},\n                "kd": "OOS-selected static prefight KD hazard; no within-fight KD escalation",\n                "submission": "ground submission attacks remain Brain/scoring attempts only; they cannot terminate; submission finish comes only from embedded survival clock",', "manifest sub")
    path.write_text(s)


if __name__ == "__main__":
    patch_tick_clock()
    patch_harness()
