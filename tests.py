"""
Tests for the production stage classifier.

Two suites:
  Unit tests  (offline) — mock _call(); test pipeline logic deterministically.
  Integration tests     — live API calls; validate prompt+model behaviour.

Run offline only:  python3 tests.py --offline
Run all:           python3 tests.py
"""

import sys
from dataclasses import dataclass
from typing import Optional
from unittest.mock import patch

from classifier import classify, _REFLECTION_THRESHOLD
from models import ClassificationResult, Stage


@dataclass
class Case:
    text: str
    expected: Stage
    label: str
    note: str = ""


# ---------------------------------------------------------------------------
# SUITE 1 — Offline unit tests (mocked _call)
# Tests pipeline logic: reflection triggering, fallback, invalid stage handling.
# These run without any API key and are fully deterministic.
# ---------------------------------------------------------------------------

def _mock_call(stage: Stage, confidence: float, reasoning: str = "mocked"):
    """Returns a mock that makes _call() return a fixed response."""
    return {"stage": stage, "confidence": confidence, "reasoning": reasoning}


def run_unit_tests(verbose: bool = True) -> int:
    failures = 0

    def check(label: str, got, expected, note: str = ""):
        nonlocal failures
        ok = got == expected
        if not ok:
            failures += 1
        if verbose:
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] unit::{label}")
            if not ok:
                print(f"       Expected: {expected!r}  Got: {got!r}")
                if note:
                    print(f"       Note: {note}")

    # 1. High-confidence path — critic should NOT run (1 call only)
    call_count = 0
    def single_call(client, system, user):
        nonlocal call_count
        call_count += 1
        return _mock_call("production", 0.95)

    with patch("classifier._call", side_effect=single_call):
        result = classify("some text")
    check("no_reflection_when_confident", result.method, "llm")
    check("single_call_when_confident", call_count, 1)

    # 2. Low-confidence path — critic SHOULD run (2 calls)
    call_count = 0
    responses = [
        _mock_call("production", 0.50),   # classifier: uncertain
        _mock_call("pre_production", 0.90),  # critic: overrides
    ]
    def two_calls(client, system, user):
        nonlocal call_count
        r = responses[call_count]
        call_count += 1
        return r

    with patch("classifier._call", side_effect=two_calls):
        result = classify("some ambiguous text")
    check("reflection_triggers_below_threshold", result.method, "llm+reflection")
    check("critic_call_count", call_count, 2)
    check("critic_can_override_stage", result.stage, "pre_production")

    # 3. Empty input — returns without any API call
    call_count = 0
    with patch("classifier._call", side_effect=two_calls):
        result = classify("")
    check("empty_input_no_api_call", call_count, 0)
    check("empty_input_confidence_zero", result.confidence, 0.0)
    check("empty_input_method_none", result.method, "none")

    # 4. API failure — fallback returned, no crash
    def always_fails(client, system, user):
        raise RuntimeError("API down")

    with patch("classifier._call", side_effect=always_fails):
        result = classify("some text")
    check("api_failure_no_crash", result.confidence, 0.0)
    check("api_failure_returns_result", isinstance(result, ClassificationResult), True)

    # 5. Invalid stage in response — coerced to fallback
    with patch("classifier._call", return_value={"stage": "post_production", "confidence": 0.9, "reasoning": "x"}):
        result = classify("some text")
    check("invalid_stage_coerced", result.stage, "development")
    check("invalid_stage_low_confidence", result.confidence, 0.1)

    # 6. Literal type check — method values are valid
    for method_val in ("none", "llm", "llm+reflection"):
        r = ClassificationResult(stage="development", confidence=0.9, reasoning="x", method=method_val)
        check(f"literal_type_{method_val}", r.method, method_val)


    print()
    return failures


# ---------------------------------------------------------------------------
# SUITE 2 — Integration test cases (live API)
# These call the real API. Results can vary with model updates but serve as
# a behavioural regression baseline. Run with: python3 tests.py
# ---------------------------------------------------------------------------

CASES: list[Case] = [
    # --- DEVELOPMENT ---
    Case(
        label="dev_option",
        expected="development",
        text="Sony Pictures has optioned the rights to the bestselling novel 'The Hollow Hours'. "
             "A writer is attached to adapt the screenplay.",
    ),
    Case(
        label="dev_greenlight",
        expected="development",
        text="Netflix greenlighted the 8-episode series after a competitive bidding war. "
             "No casting has begun.",
    ),
    Case(
        label="dev_pitch",
        expected="development",
        text="The showrunner pitched the concept to HBO last Tuesday. Still in early talks "
             "about the budget and episode count. Financing discussions ongoing.",
    ),
    Case(
        label="dev_messy",
        expected="development",
        text="Heard thru the grapevine - big director (won't say who) is in talks 2 attach "
             "himself to the sequel. Studio still figuring $ out. Script deal announced "
             "last wk on deadline",
    ),
    Case(
        label="dev_set_up",
        expected="development",
        text="The project has been set up at Blumhouse. The spec script sold in a high six "
             "figure deal. Packaging underway.",
    ),

    # --- PRE-PRODUCTION ---
    Case(
        label="preprod_casting",
        expected="pre_production",
        text="Casting call going out this week for the lead role. They're looking for a woman "
             "25-35. Open casting in NYC and LA.",
    ),
    Case(
        label="preprod_location",
        expected="pre_production",
        text="The location scout wrapped up yesterday. Production designer confirmed. "
             "They've locked in Budapest for the exterior shoots.",
    ),
    Case(
        label="preprod_table_read",
        expected="pre_production",
        text="First table read happened Monday with the full cast. Rehearsals start next week "
             "before the crew moves to set.",
    ),
    Case(
        label="preprod_messy",
        expected="pre_production",
        text="pre prod is FINALLY underway after all those delays lol. "
             "crew hiring started, still scouting secondary locations in albuquerque. "
             "shoot dates TBD but looking like late Q3",
    ),
    Case(
        label="preprod_office",
        expected="pre_production",
        text="Production office is open in Burbank. Art department and storyboard artists "
             "reporting in. Production schedule being finalised.",
    ),

    # --- PRODUCTION ---
    Case(
        label="prod_principal",
        expected="production",
        text="Principal photography began Monday at Pinewood Studios. "
             "The unit publicist confirmed the shoot is on schedule.",
    ),
    Case(
        label="prod_wrap",
        expected="production",
        text="🎬 AND THAT'S A WRAP on Season 3!! After 87 days of filming across 6 countries "
             "we finally did it. Incredible cast and crew.",
    ),
    Case(
        label="prod_onset",
        expected="production",
        text="Sources say filming is underway in New Orleans. Spotted on set: the lead actress "
             "in full costume. Shoot day 14.",
    ),
    Case(
        label="prod_messy",
        expected="production",
        text="day 3 of shooting - already behind schedule bc of rain. dailies look gorgeous tho. "
             "second unit doing pickups downtown while we set up the big scene",
    ),
    Case(
        label="prod_shooting",
        expected="production",
        text="The cameras are rolling in Prague this week. Shot on location in the old city. "
             "Production wrapping end of month.",
    ),

    # --- AMBIGUOUS ---
    Case(
        label="ambiguous_cast_announced",
        expected="pre_production",
        text="Cast announced today: [big star] will lead the series. "
             "The show begins filming in Toronto this fall.",
        note="Cast announced could be dev or pre-prod; 'begins filming' tips it.",
    ),
    Case(
        label="ambiguous_mixed",
        expected="development",
        text="The studio acquired the book rights last year. Now they're close to a greenlight "
             "but no crew has been assembled yet and casting hasn't started.",
        note="Acquisition + greenlight pending = still development.",
    ),
    Case(
        label="ambiguous_vague",
        expected="production",
        text="Long day on set. Can't say much but it's going well :)",
        note="Minimal text — 'on set' is the only signal.",
    ),

    # --- HARD EDGE CASES ---
    Case(
        label="hard_negation",
        expected="pre_production",
        text="The studio denied rumors that filming has started. "
             "A spokesperson confirmed they are still finalizing locations and assembling the crew.",
        note="'filming' is explicitly negated; current state is pre-prod.",
    ),
    Case(
        label="hard_past_project",
        expected="development",
        text="After their troubled shoot wrapped two years ago, the same team is now "
             "pitching a new thriller to streamers. No deals signed yet.",
        note="'shoot/wrapped' are historical; current action is pitching = development.",
    ),
    Case(
        label="hard_future_principal",
        expected="pre_production",
        text="Amazon Studios announced it will begin principal photography "
             "on the spy thriller in Vancouver this October.",
        note="'principal photography' is future tense — cameras haven't rolled yet.",
    ),
    Case(
        label="hard_metaphor",
        expected="development",
        text="The script is really shooting for the stars — best draft I've read all year. "
             "Still needs studio approval and a greenlight before anything moves forward.",
        note="'shooting' is metaphorical; 'needs greenlight' confirms development.",
    ),
    Case(
        label="hard_production_as_noun",
        expected="pre_production",
        text="Sources close to the production confirm the director had creative differences "
             "with the studio, but cameras are not rolling yet and casting wraps next week.",
        note="'production' = the project noun; 'cameras not rolling' + 'casting' = pre-prod.",
    ),
    Case(
        label="hard_mixed_timeline",
        expected="development",
        text="Season 2 just wrapped last Friday. The writers' room for Season 3 "
             "opens Monday — no director attached yet.",
        note="S2 wrapped is past; current action is S3 writers room = development.",
    ),
    Case(
        label="hard_implicit_onset",
        expected="production",
        text="Just spotted [redacted] in full period costume and heavy makeup "
             "outside the old mill district with a massive crew and lighting rigs. "
             "Something is definitely happening over there 👀",
        note="No filming keywords; must infer from costume+crew+lighting rigs = production.",
    ),
    Case(
        label="hard_executive_vague",
        expected="development",
        text="The CEO noted in the earnings call: 'We remain deeply committed to this "
             "franchise and expect meaningful progress in the coming quarters.'",
        note="Corporate non-answer; no stage signals — best guess is development.",
    ),
    Case(
        label="hard_writers_room",
        expected="development",
        text="The writing staff is pulling 14-hour days breaking every episode of the season. "
             "No director attached yet and casting hasn't begun.",
        note="Active writers room = development, not pre-prod; no crew or casting.",
    ),
    Case(
        label="hard_greenlight_with_date",
        expected="pre_production",
        text="The project was officially greenlit this morning. "
             "Shoot date is set for March and the production designer starts Monday.",
        note="Greenlit = was development; shoot date + production designer = now pre-prod.",
    ),
]


def run_integration_tests(verbose: bool = True) -> int:
    failures = 0

    for case in CASES:
        result: ClassificationResult = classify(case.text)
        ok = result.stage == case.expected

        if not ok:
            failures += 1

        if verbose:
            status = "PASS" if ok else "FAIL"
            print(f"[{status}] {case.label}")
            if not ok:
                print(f"       Expected:  {case.expected}")
                print(f"       Got:       {result.stage}")
            print(f"       Confidence: {result.confidence:.0%}  Method: {result.method}")
            print(f"       Reasoning:  {result.reasoning}")
            if case.note:
                print(f"       Note:       {case.note}")
            print()

    return failures


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    offline_only = "--offline" in sys.argv

    print("=" * 60)
    print("UNIT TESTS (offline, mocked)")
    print("=" * 60)
    unit_failures = run_unit_tests()
    print(f"Unit: {'PASSED' if unit_failures == 0 else f'FAILED ({unit_failures})'}")

    if offline_only:
        sys.exit(1 if unit_failures else 0)

    print()
    print("=" * 60)
    print(f"INTEGRATION TESTS (live API, {len(CASES)} cases)")
    print("=" * 60)
    int_failures = run_integration_tests()
    total = len(CASES)
    passed = total - int_failures
    print(f"Integration: {passed}/{total} passed")

    if unit_failures or int_failures:
        sys.exit(1)
    else:
        print("\nAll tests passed.")
