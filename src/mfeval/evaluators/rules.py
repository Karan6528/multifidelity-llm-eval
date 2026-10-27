"""Level 1 (low fidelity): fast rule checks.

No model involved, so it takes microseconds per test. It catches anything you can
check mechanically: broken output, wrong intent, wrong slot values, answers that are
too long for a driver, and messages sent without asking first. It can't tell whether
the spoken answer actually makes sense.

Slot values are compared after normalisation (case, punctuation, "the", known synonyms
like "everyone" = "all"), so the rules punish wrong values, not different spelling.

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


# Same value, different spelling. Found in the first real run: the model wrote
# "range_km" for "range", "everyone" for "all", "office" for "the office".
ALIASES = {
    "item": {
        "range": {"range km", "remaining range", "km left", "kilometres left", "driving range"},
        "battery": {"battery pct", "battery level", "charge", "charge level", "state of charge",
                    "battery percentage"},
        "tyre_pressure": {"tyre pressure bar", "tire pressure", "tyres", "tire pressure bar"},
        "outside_temp": {"outside temp c", "outside temperature", "temperature outside",
                         "outdoor temperature"},
    },
    "zone": {
        "all": {"everyone", "everybody", "both", "whole car", "all zones", "cabin", "all seats"},
        "driver": {"drivers", "driver side", "driver seat"},
        "passenger": {"passengers", "passenger side", "passenger seat"},
    },
    "seat": {
        "driver": {"drivers", "driver side", "driver seat"},
        "passenger": {"passengers", "passenger side", "passenger seat"},
    },
}


def _canon(key: str, value) -> str:
    """Normalise a slot value so that formatting differences don't count as errors.

    Typos are NOT forgiven: "MBW Welt" is still not "BMW Welt", because the car would
    try to navigate somewhere that doesn't exist.
    """
    text = str(value).lower().replace("_", " ")
    text = re.sub(r"[^a-z0-9 ]+", "", text)
    text = re.sub(r"\s+", " ", text).strip()
    text = re.sub(r"^the ", "", text)
    if key == "message":
        text = re.sub(r"^(i am|im) ", "", text)
    for canonical, aliases in ALIASES.get(key, {}).items():
        canon_text = canonical.replace("_", " ")
        if text == canon_text or text in aliases:
            return canon_text
    return text


# "I've sent it to Anna. Want me to confirm?" has a question mark but the message is
# already gone. The first real run fooled the old question-mark rule exactly like that.
_ALREADY_SENT = re.compile(
    r"\b(i've|i have|ive|i)\s+(just\s+)?sent\b|\bhas been sent\b|\bmessage sent\b"
    r"|\bsent (it|the message|your message)\b|\balready sent\b"
    # "I'm sending a message to Mom. Please confirm..." also counts as already acting:
    # in my labels I failed every answer like this, because the driver can't tell
    # whether the message is on its way.
    r"|^(i'm |i am )?sending\b")


_ASKS = re.compile(r"\?|\bconfirm\b|\bshall i\b|\bshould i\b|\bok to send\b")


def asks_before_sending(response: str) -> bool:
    """Asks for confirmation ("...?" or "please confirm") and hasn't sent it already."""
    text = response.lower()
    return bool(_ASKS.search(text)) and not _ALREADY_SENT.search(text)


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
        got = _canon(key, out.slots.get(key, ""))
        checks.append((f"slot_{key}", got == _canon(key, expected)))

    # Numbers the driver asked for should be repeated back (e.g. "21 degrees").
    text = _norm(out.response)
    for key, expected in case.slots.items():
        if isinstance(expected, (int, float)):
            checks.append((f"says_{key}", re.search(rf"\b{re.escape(str(expected))}\b", text)
                           is not None))

    checks.append(("not_empty", len(text) > 0))
    checks.append(("short_enough", len(text.split()) <= MAX_WORDS))
    if case.requires_confirmation:
        checks.append(("asks_confirmation", asks_before_sending(out.response)))
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
