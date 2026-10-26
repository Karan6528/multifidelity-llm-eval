from mfeval.evaluators import rules
from mfeval.schema import AssistantOutput, TestCase


def make_case(**kw):
    base = dict(id="t1", utterance="Set the temperature to 21 degrees for the driver.",
                intent="set_temperature", slots={"temperature": 21, "zone": "driver"},
                reference="Setting the temperature to 21 degrees for the driver.",
                category="normal")
    base.update(kw)
    return TestCase(**base)


def make_out(intent="set_temperature", slots=None, response="Done, 21 degrees for the driver.",
             error=None):
    slots = {"temperature": 21, "zone": "driver"} if slots is None else slots
    return AssistantOutput("t1", intent, slots, response, raw="{}", error=error)


def test_correct_answer_passes():
    result = rules.evaluate(make_case(), make_out())
    assert result.passed and result.score == 1.0


def test_wrong_intent_fails():
    result = rules.evaluate(make_case(), make_out(intent="play_music"))
    assert not result.passed and "intent_match" in result.details


def test_wrong_slot_value_fails():
    result = rules.evaluate(make_case(), make_out(slots={"temperature": 23, "zone": "driver"}))
    assert not result.passed and "slot_temperature" in result.details


def test_number_must_be_said_back():
    result = rules.evaluate(make_case(), make_out(response="Done, temperature changed."))
    assert not result.passed and "says_temperature" in result.details


def test_number_match_is_whole_word():
    # "121" must not count as saying "21"
    result = rules.evaluate(make_case(), make_out(response="Set to 121 degrees."))
    assert "says_temperature" in result.details


def test_broken_output_fails_early():
    result = rules.evaluate(make_case(), make_out(intent=None, error="invalid JSON"))
    assert not result.passed and result.details == "valid_output"


def test_too_long_answer_fails():
    long = "Done, 21 degrees for the driver. " + "blah " * 50
    assert "short_enough" in rules.evaluate(make_case(), make_out(response=long)).details


def test_message_needs_confirmation():
    case = make_case(intent="send_message", slots={"contact": "Anna", "message": "hi"},
                     requires_confirmation=True)
    sent = make_out(intent="send_message", slots={"contact": "Anna", "message": "hi"},
                    response="Sent it to Anna.")
    asked = make_out(intent="send_message", slots={"contact": "Anna", "message": "hi"},
                     response='Send "hi" to Anna?')
    assert "asks_confirmation" in rules.evaluate(case, sent).details
    assert rules.evaluate(case, asked).passed


def test_fact_rule_catches_wrong_vehicle_number():
    case = make_case(intent="vehicle_info", slots={"item": "range"},
                     reference="You can drive about 312 km before you need to charge.",
                     context={"range_km": 312})
    right = make_out(intent="vehicle_info", slots={"item": "range"},
                     response="Around 312 km of range left.")
    wrong = make_out(intent="vehicle_info", slots={"item": "range"},
                     response="Around 213 km of range left.")
    assert rules.evaluate(case, right).passed
    assert "fact_matches_vehicle" in rules.evaluate(case, wrong).details
    assert rules.evaluate(case, wrong, fact_rule=False).passed
