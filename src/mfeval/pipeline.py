"""Run the assistant over the suite and score every answer at every level."""

from __future__ import annotations

import json
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

from .assistants import Assistant
from .evaluators import rules, semantic
from .schema import AssistantOutput, TestCase, read_jsonl, write_jsonl


def _progress(i: int, n: int, what: str) -> None:
    if i == n or i % 20 == 0:
        print(f"  {what}: {i}/{n}", file=sys.stderr, flush=True)


def generate(cases: list[TestCase], assistant: Assistant, run_dir: Path) -> list[AssistantOutput]:
    """Ask the assistant every question and save the raw answers."""
    run_dir.mkdir(parents=True, exist_ok=True)
    outputs = []
    for i, case in enumerate(cases, 1):
        outputs.append(assistant.answer(case))
        _progress(i, len(cases), "answers")
    write_jsonl(run_dir / "outputs.jsonl", outputs)
    meta = {"assistant": assistant.name, "n_cases": len(cases),
            "created": datetime.now(timezone.utc).isoformat(timespec="seconds")}
    (run_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return outputs


def load_outputs(run_dir: Path) -> list[AssistantOutput]:
    return read_jsonl(run_dir / "outputs.jsonl", AssistantOutput)


def score(cases: list[TestCase], outputs: list[AssistantOutput], embedder, judge,
          run_dir: Path, l2_threshold: float | None = None,
          fact_rule: bool = True) -> pd.DataFrame:
    """Score every answer at L1, L2 and L3 and save scores.csv."""
    by_id = {c.id: c for c in cases}
    cases = [by_id[o.case_id] for o in outputs]

    l1 = [rules.evaluate(c, o, fact_rule) for c, o in zip(cases, outputs)]
    l2 = semantic.evaluate_batch(cases, outputs, embedder, l2_threshold)
    l3 = []
    start = time.perf_counter()
    for i, (c, o) in enumerate(zip(cases, outputs), 1):
        l3.append(judge.evaluate(c, o))
        _progress(i, len(cases), "judge")
    judge_seconds = time.perf_counter() - start

    df = pd.DataFrame({
        "case_id": [c.id for c in cases],
        "category": [c.category for c in cases],
        "intent": [c.intent for c in cases],
        "utterance": [c.utterance for c in cases],
        "response": [o.response for o in outputs],
        "injected_fault": [o.injected_fault or "" for o in outputs],
        "l1_pass": [r.passed for r in l1], "l1_score": [r.score for r in l1],
        "l1_details": [r.details for r in l1], "l1_seconds": [r.latency_s for r in l1],
        "l2_pass": [r.passed for r in l2], "l2_score": [r.score for r in l2],
        "l2_seconds": [r.latency_s for r in l2],
        "l3_pass": [r.passed for r in l3], "l3_score": [r.score for r in l3],
        "l3_details": [r.details for r in l3], "l3_seconds": [r.latency_s for r in l3],
    })
    df.to_csv(run_dir / "scores.csv", index=False)

    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"embedder": embedder.name, "judge": judge.name, "fact_rule": fact_rule,
                 "judge_seconds_total": round(judge_seconds, 2)})
    meta_path.write_text(json.dumps(meta, indent=2))
    return df


def load_scores(run_dir: Path) -> pd.DataFrame:
    df = pd.read_csv(run_dir / "scores.csv", keep_default_na=False)
    for col in ("l1_pass", "l2_pass", "l3_pass"):
        df[col] = df[col].astype(str).str.lower().eq("true")
    return df


def attach_truth(df: pd.DataFrame, run_dir: Path) -> tuple[pd.DataFrame, str]:
    """Decide what counts as 'really failed'.

    * Mock runs: the faults we injected (exact ground truth).
    * Real runs: an answer counts as broken if it breaks a hard rule (a wrong slot value
      is a bug whatever the judge thinks) or the judge fails it. Your own labels
      (labels.csv) are then used to check how far the judge can be trusted.
    """
    df = df.copy()
    if (df["injected_fault"] != "").any() or json.loads(
            (run_dir / "meta.json").read_text()).get("assistant") == "mock":
        df["truth_fail"] = df["injected_fault"] != ""
        return df, "injected faults (exact)"
    df["truth_fail"] = ~df["l1_pass"] | ~df["l3_pass"]
    return df, "failed a hard rule or the LLM judge (judge checked against your labels)"


def load_labels(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "labels.csv"
    if not path.exists():
        return None
    labels = pd.read_csv(path)
    labels["human_fail"] = labels["human_verdict"].str.lower().eq("fail")
    return labels
