"""End-to-end run with the simulated assistant, the TF-IDF embedder and the mock judge."""

import json

from mfeval import pipeline, report
from mfeval.assistants import MockAssistant
from mfeval.evaluators import rules
from mfeval.evaluators.judge import MockJudge
from mfeval.evaluators.semantic import TfidfEmbedder
from mfeval.testsuite import INTENTS, build_suite


def test_suite_is_deterministic_and_valid():
    a, b = build_suite(seed=7), build_suite(seed=7)
    assert [c.utterance for c in a] == [c.utterance for c in b]
    assert len({c.id for c in a}) == len(a) > 150
    for c in a:
        assert c.intent in INTENTS
        assert set(c.slots) == set(INTENTS[c.intent])


def test_mock_correct_answers_pass_the_rules():
    cases = build_suite()
    mock = MockAssistant(seed=1, fault_scale=0.0)  # no faults at all
    for c in cases:
        out = mock.answer(c)
        result = rules.evaluate(c, out)
        assert result.passed, (c.id, c.utterance, out.response, result.details)


def test_full_pipeline(tmp_path):
    cases = build_suite()
    outputs = pipeline.generate(cases, MockAssistant(seed=3), tmp_path)
    df = pipeline.score(cases, outputs, TfidfEmbedder(), MockJudge(), tmp_path)
    assert len(df) == len(cases)

    df, source = pipeline.attach_truth(pipeline.load_scores(tmp_path), tmp_path)
    assert "injected" in source and df["truth_fail"].sum() > 10

    out = report.build_report(df, tmp_path, source, {"L1": 1, "L2": 5, "L3": 100})
    assert (tmp_path / "report.md").exists() and (tmp_path / "frontier.png").exists()
    assert out["full"].recall > 0.8
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["judge"] == "mock-judge"
