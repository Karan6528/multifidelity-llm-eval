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


def rescore(cases: list[TestCase], outputs: list[AssistantOutput], run_dir: Path,
            fact_rule: bool = True, embedder=None,
            l2_threshold: float | None = None) -> pd.DataFrame:
    """Re-run the cheap levels on a finished run and keep the judge's verdicts.

    The judge is the slow part (35 minutes in my first real run), so after changing a
    rule there is no reason to run it again. The first version of scores.csv is kept as
    scores_v1.csv so the two can be compared.
    """
    path = run_dir / "scores.csv"
    v1 = run_dir / "scores_v1.csv"
    if not v1.exists():
        v1.write_bytes(path.read_bytes())
    df = load_scores(run_dir)
    by_id = {c.id: c for c in cases}
    out_by_id = {o.case_id: o for o in outputs}
    ordered_cases = [by_id[cid] for cid in df["case_id"]]
    ordered_outs = [out_by_id[cid] for cid in df["case_id"]]

    l1 = [rules.evaluate(c, o, fact_rule) for c, o in zip(ordered_cases, ordered_outs)]
    df["l1_pass"] = [r.passed for r in l1]
    df["l1_score"] = [r.score for r in l1]
    df["l1_details"] = [r.details for r in l1]
    df["l1_seconds"] = [r.latency_s for r in l1]
    if embedder is not None:
        l2 = semantic.evaluate_batch(ordered_cases, ordered_outs, embedder, l2_threshold)
        df["l2_pass"] = [r.passed for r in l2]
        df["l2_score"] = [r.score for r in l2]
        df["l2_seconds"] = [r.latency_s for r in l2]
    df.to_csv(path, index=False)

    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    meta.update({"fact_rule": fact_rule, "rescored": True})
    if embedder is not None:
        meta["embedder"] = embedder.name
    meta_path.write_text(json.dumps(meta, indent=2))
    return df


def is_simulated(df: pd.DataFrame, run_dir: Path) -> bool:
    meta_path = run_dir / "meta.json"
    meta = json.loads(meta_path.read_text()) if meta_path.exists() else {}
    return bool((df["injected_fault"] != "").any()) or meta.get("assistant") == "mock"


def attach_truth(df: pd.DataFrame, run_dir: Path,
                 labels: pd.DataFrame | None = None) -> tuple[pd.DataFrame, str, str]:
    """Decide what counts as 'really failed'. Returns (df, description, mode).

    * Simulated runs ("exact"): the faults that were injected.
    * Real runs with my labels ("adjudicated"): my label where I gave one. For the rest,
      the verdict of rules and judge when they agree. Answers where they disagree and I
      haven't labelled them are left out, because there is no honest answer for them yet.
    * Real runs without labels ("unlabelled"): there is no ground truth. The report then
      only shows how the levels agree with each other, never a recall number. (The first
      version counted "fails a rule OR the judge" as truth, which made the rules look
      97% perfect by definition.)
    """
    df = df.copy()
    if is_simulated(df, run_dir):
        df["truth_fail"] = df["injected_fault"] != ""
        return df, "injected faults (exact)", "exact"
    if labels is None or not len(labels):
        return df, "none yet: label answers with `mfeval label`", "unlabelled"

    human = dict(zip(labels["case_id"], labels["human_fail"]))
    agree = df["l1_pass"] == df["l3_pass"]
    truth = []
    for cid, ok, l1 in zip(df["case_id"], agree, df["l1_pass"]):
        if cid in human:
            truth.append(bool(human[cid]))
        elif ok:
            truth.append(not l1)
        else:
            truth.append(None)
    df["truth_fail"] = truth
    n_human = int(df["case_id"].isin(set(human)).sum())
    kept = df[df["truth_fail"].notna()].copy()
    kept["truth_fail"] = kept["truth_fail"].astype(bool)
    n_dropped = len(df) - len(kept)
    desc = (f"my labels for {n_human} answers, rules + judge agreement for "
            f"{len(kept) - n_human}; {n_dropped} unlabelled disagreements left out")
    return kept, desc, "adjudicated"


def pick_for_labelling(df: pd.DataFrame, n: int, seed: int = 42,
                       done: set[str] | None = None) -> list[str]:
    """Choose which answers to label, most informative first.

    1. rules and judge disagree (someone is wrong, my label decides)
    2. rules and judge agree but similarity disagrees with both (the case that caught
       "I can turn off the warnings for you" in the first run)
    3. a random share of answers where everything agrees, to check that agreement
       doesn't hide mistakes. At least a fifth of the budget goes here.
    """
    import random

    done = done or set()
    rng = random.Random(seed)
    rest = df[~df["case_id"].isin(done)]
    l1, l2, l3 = rest["l1_pass"], rest["l2_pass"], rest["l3_pass"]
    groups = [
        list(rest[l1 != l3]["case_id"]),
        list(rest[(l1 == l3) & (l2 != l1)]["case_id"]),
        list(rest[(l1 == l3) & (l2 == l1)]["case_id"]),
    ]
    for g in groups:
        rng.shuffle(g)
    n_check = min(len(groups[2]), max(1, n // 5))
    picked = (groups[0] + groups[1])[: n - n_check] + groups[2][:n_check]
    leftovers = [c for g in groups for c in g if c not in set(picked)]
    picked += leftovers[: n - len(picked)]
    return picked


def load_labels(run_dir: Path) -> pd.DataFrame | None:
    path = run_dir / "labels.csv"
    if not path.exists():
        return None
    labels = pd.read_csv(path)
    if not len(labels):
        return None
    labels["human_fail"] = labels["human_verdict"].str.lower().eq("fail")
    return labels
