"""Level 1 (low fidelity): fast rule checks.

No model involved, so it takes microseconds per test. It catches anything you can
check mechanically: broken output, wrong intent, wrong slot values, answers that are
too long for a driver, and messages sent without asking first. It can't tell whether
the spoken answer actually makes sense.

Optional extra rule (on by default, switch off with --no-fact-rule): for questions about
the car, the number in the answer must match the vehicle state. Without it, wrong
numbers slip through every cheap check (see the README).
"""

from __future__ import annotations

import re
import time

from ..schema import AssistantOutput, EvalResult, TestCase
from ..testsuite import VEHICLE_ITEMS

MAX_WORDS = 40  # longer than this is too much to listen to while driving


def _norm(value) -> str:
    return re.sub(r"\s+", " ", str(value).strip().lower())


def _numbers(text: str) -> list[float]:
    return [float(x) for x in re.findall(r"-?\d+(?:[.,]\d+)?", text.replace(",", "."))]


def check(case: TestCase, out: AssistantOutput,
          fact_rule: bool = True) -> list[tuple[str, bool]]:
    """Return (check name, passed) for every rule that applies to this case."""
    checks = [("valid_output", out.error is None and out.intent is not None)]
    if not checks[0][1]:
        return checks

    checks.append(("intent_match", out.intent == case.intent))
    for key, expected in case.slots.items():
        checks.append((f"slot_{key}", _norm(out.slots.get(key, "")) == _norm(expected)))

    # Numbers the driver asked for should be repeated back (e.g. "21 degrees").
    text = _norm(out.response)
    for key, expected in case.slots.items():
        if isinstance(expected, (int, float)):
            checks.append((f"says_{key}", re.search(rf"\b{re.escape(str(expected))}\b", text)
                           is not None))

    checks.append(("not_empty", len(text) > 0))
    checks.append(("short_enough", len(text.split()) <= MAX_WORDS))
    if case.requires_confirmation:
        checks.append(("asks_confirmation", "?" in out.response))
    if fact_rule and case.intent == "vehicle_info":
        key = VEHICLE_ITEMS[case.slots["item"]][0]
        expected = float(case.context[key])
        checks.append(("fact_matches_vehicle", any(abs(n - expected) < 1e-6
                                                   for n in _numbers(out.response))))
    return checks


def evaluate(case: TestCase, out: AssistantOutput, fact_rule: bool = True) -> EvalResult:
    start = time.perf_counter()
    checks = check(case, out, fact_rule)
    failed = [name for name, ok in checks if not ok]
    # A single failed rule is a real bug, so any failure fails the test.
    score = 1.0 - len(failed) / len(checks)
    return EvalResult(
        case_id=case.id, level="L1", passed=not failed, score=round(score, 4),
        details=", ".join(failed), latency_s=time.perf_counter() - start,
    )
