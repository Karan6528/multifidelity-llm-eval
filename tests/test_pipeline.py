"""End-to-end run with the simulated assistant, the TF-IDF embedder and the mock judge."""

import json

import pandas as pd

from mfeval import pipeline, report
from mfeval.assistants import MockAssistant
from mfeval.evaluators import rules
from mfeval.evaluators.judge import MockJudge
from mfeval.evaluators.semantic import TfidfEmbedder
from mfeval.schema import write_jsonl
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

    df, source, mode = pipeline.attach_truth(pipeline.load_scores(tmp_path), tmp_path)
    assert mode == "exact" and "injected" in source and df["truth_fail"].sum() > 10

    out = report.build_report(df, tmp_path, source, {"L1": 1, "L2": 5, "L3": 100}, mode=mode)
    assert (tmp_path / "report.md").exists() and (tmp_path / "frontier.png").exists()
    assert out["full"].recall > 0.8
    meta = json.loads((tmp_path / "meta.json").read_text())
    assert meta["judge"] == "mock-judge"


def _fake_real_run(tmp_path):
    """A finished 'real' run: mock answers, but meta says it came from a real model."""
    cases = build_suite()
    outputs = pipeline.generate(cases, MockAssistant(seed=5), tmp_path)
    pipeline.score(cases, outputs, TfidfEmbedder(), MockJudge(), tmp_path)
    for o in outputs:
        o.injected_fault = None  # a real model doesn't tell us its mistakes
    write_jsonl(tmp_path / "outputs.jsonl", outputs)
    scores = pd.read_csv(tmp_path / "scores.csv", keep_default_na=False)
    scores["injected_fault"] = ""
    scores.to_csv(tmp_path / "scores.csv", index=False)
    meta = json.loads((tmp_path / "meta.json").read_text())
    meta["assistant"] = "ollama:test"
    (tmp_path / "meta.json").write_text(json.dumps(meta))
    return cases, outputs


def test_real_run_without_labels_makes_no_recall_claims(tmp_path):
    _fake_real_run(tmp_path)
    df, _, mode = pipeline.attach_truth(pipeline.load_scores(tmp_path), tmp_path, None)
    assert mode == "unlabelled"
    out = report.build_report(df, tmp_path, "", {"L1": 1, "L2": 5, "L3": 100}, mode=mode)
    text = (tmp_path / "report.md").read_text()
    assert out["mode"] == "unlabelled" and "No ground truth yet" in text
    assert "failures caught" not in text and not (tmp_path / "frontier.png").exists()


def test_labels_override_and_disagreements_are_dropped(tmp_path):
    _fake_real_run(tmp_path)
    df = pipeline.load_scores(tmp_path)
    disagree = list(df[df.l1_pass != df.l3_pass].case_id)
    agree_pass = df[df.l1_pass & df.l3_pass].case_id.iloc[0]
    labels = pd.DataFrame({"case_id": [disagree[0], agree_pass],
                           "human_verdict": ["fail", "fail"]})
    labels["human_fail"] = True
    kept, desc, mode = pipeline.attach_truth(df, tmp_path, labels)
    assert mode == "adjudicated" and "2 answers" in desc
    truth = dict(zip(kept.case_id, kept.truth_fail))
    assert truth[disagree[0]] and truth[agree_pass]  # my label wins over both levels
    assert all(cid not in truth for cid in disagree[1:])  # unlabelled disagreements left out


def test_labelling_starts_with_disagreements(tmp_path):
    _fake_real_run(tmp_path)
    df = pipeline.load_scores(tmp_path)
    n_dis = int((df.l1_pass != df.l3_pass).sum())
    picked = pipeline.pick_for_labelling(df, n=40, seed=1)
    assert len(picked) == 40 == len(set(picked))
    first = df.set_index("case_id").loc[picked[: min(n_dis, 32)]]
    assert (first.l1_pass != first.l3_pass).all()
    tail = df.set_index("case_id").loc[picked[-8:]]
    assert ((tail.l1_pass == tail.l3_pass) & (tail.l2_pass == tail.l1_pass)).any()


def test_rescore_keeps_judge_and_backs_up(tmp_path):
    cases, outputs = _fake_real_run(tmp_path)
    before = pipeline.load_scores(tmp_path)
    after = pipeline.rescore(cases, outputs, tmp_path, fact_rule=False)
    assert (tmp_path / "scores_v1.csv").exists()
    assert list(after.l3_details) == list(before.l3_details)
    assert (~after.l1_pass).sum() <= (~before.l1_pass).sum()
