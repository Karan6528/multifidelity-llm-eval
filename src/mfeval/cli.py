"""Command line interface. Run `mfeval --help` (or `python -m mfeval --help`)."""

from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import pandas as pd
import yaml

from . import pipeline, report
from .assistants import make_assistant
from .evaluators.judge import make_judge
from .evaluators.semantic import make_embedder
from .testsuite import load_suite, save_suite

ROOT = Path.cwd()
DEFAULT_SUITE = ROOT / "data" / "testsuite.jsonl"


def load_config(path: Path | None) -> dict:
    path = path or ROOT / "configs" / "default.yaml"
    return yaml.safe_load(path.read_text()) if path.exists() else {}


def _suite(path: Path):
    if not path.exists():
        print(f"No test suite at {path}, building it.", file=sys.stderr)
        save_suite(path)
    return load_suite(path)


def cmd_suite(args, cfg) -> None:
    cases = save_suite(Path(args.out), seed=args.seed)
    counts = pd.Series([c.category for c in cases]).value_counts().to_dict()
    print(f"Wrote {len(cases)} test cases to {args.out}: {counts}")


def cmd_run(args, cfg) -> None:
    cases = _suite(Path(args.suite))
    assistant = make_assistant(args.backend, args.model, seed=args.seed)
    run_dir = Path(args.runs) / args.name
    pipeline.generate(cases, assistant, run_dir)
    print(f"Saved {len(cases)} answers from {assistant.name} to {run_dir}/outputs.jsonl")


def cmd_score(args, cfg) -> None:
    run_dir = Path(args.runs) / args.name
    cases = _suite(Path(args.suite))
    outputs = pipeline.load_outputs(run_dir)
    judge = make_judge(args.judge, args.judge_model)
    embedder = make_embedder(args.embedder)
    threshold = (cfg.get("l2_threshold") or {}).get(args.embedder)
    df = pipeline.score(cases, outputs, embedder, judge, run_dir, threshold,
                        fact_rule=not args.no_fact_rule)
    print(f"Scored {len(df)} answers: L1 failed {(~df.l1_pass).sum()}, "
          f"L2 failed {(~df.l2_pass).sum()}, L3 failed {(~df.l3_pass).sum()}")


def cmd_report(args, cfg) -> None:
    run_dir = Path(args.runs) / args.name
    df, truth_source = pipeline.attach_truth(pipeline.load_scores(run_dir), run_dir)
    costs = {**{"L1": 1, "L2": 5, "L3": 100}, **(cfg.get("costs") or {})}
    out = report.build_report(df, run_dir, truth_source, costs, pipeline.load_labels(run_dir))
    full, best = out["full"], out["match_full"]
    print(f"Rules + full judge: {100 * full.recall:.1f}% of {out['n_fail']} failures "
          f"at {full.cost_pct:.0f}% cost.")
    if best:
        print(f"Best near-equal option: {best.family} ({best.label}) -> "
              f"{100 * best.recall:.1f}% at {best.cost_pct:.1f}% cost.")
    print(f"Report: {run_dir / 'report.md'}")


def cmd_all(args, cfg) -> None:
    cmd_run(args, cfg)
    cmd_score(args, cfg)
    cmd_report(args, cfg)


def cmd_label(args, cfg) -> None:
    """Label a random sample of answers yourself, to check how much the judge can be trusted."""
    run_dir = Path(args.runs) / args.name
    df = pipeline.load_scores(run_dir)
    cases = {c.id: c for c in _suite(Path(args.suite))}
    path = run_dir / "labels.csv"
    empty = pd.DataFrame(columns=["case_id", "human_verdict"])
    done = pd.read_csv(path) if path.exists() else empty
    todo = [cid for cid in df["case_id"] if cid not in set(done["case_id"])]
    random.Random(args.seed).shuffle(todo)
    todo = todo[: max(0, args.n - len(done))]
    print(f"Labelling {len(todo)} answers. Type p = pass, f = fail, s = skip, q = quit.\n")
    rows = done.to_dict("records")
    for i, cid in enumerate(todo, 1):
        c, r = cases[cid], df.set_index("case_id").loc[cid]
        print(f"[{i}/{len(todo)}] {cid}  ({c.category})")
        print(f"  Driver:    {c.utterance}")
        print(f"  Vehicle:   {c.context}")
        print(f"  Expected:  {c.reference}")
        print(f"  Assistant: {r['response'] or '(no valid answer)'}")
        while True:
            ans = input("  pass/fail? [p/f/s/q] ").strip().lower()
            if ans in {"p", "f", "s", "q"}:
                break
        if ans == "q":
            break
        if ans != "s":
            rows.append({"case_id": cid, "human_verdict": "pass" if ans == "p" else "fail"})
            pd.DataFrame(rows).to_csv(path, index=False)
        print()
    print(f"Saved {len(rows)} labels to {path}. Run `mfeval report --name {args.name}` to see "
          "how well each level agrees with you.")


def build_parser(cfg: dict) -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="mfeval", description=__doc__)
    p.add_argument("--config", type=Path, default=None,
                   help="YAML config (default: configs/default.yaml)")
    sub = p.add_subparsers(dest="command", required=True)

    def common(sp, need_run=True):
        sp.add_argument("--suite", default=str(DEFAULT_SUITE))
        sp.add_argument("--runs", default=cfg.get("runs_dir", "runs"))
        if need_run:
            sp.add_argument("--name", default="mock", help="run name, i.e. folder under runs/")
        sp.add_argument("--seed", type=int, default=42)

    sp = sub.add_parser("suite", help="build the test suite")
    sp.add_argument("--out", default=str(DEFAULT_SUITE))
    sp.add_argument("--seed", type=int, default=7)
    sp.set_defaults(func=cmd_suite)

    def run_args(sp):
        sp.add_argument("--backend", choices=["mock", "ollama"], default="mock")
        sp.add_argument("--model", default=cfg.get("assistant_model", "llama3.2:3b"))

    def score_args(sp):
        sp.add_argument("--embedder", choices=["st", "tfidf"], default="tfidf")
        sp.add_argument("--judge", choices=["mock", "ollama"], default="mock")
        sp.add_argument("--judge-model", default=cfg.get("judge_model", "qwen2.5:7b"))
        sp.add_argument("--no-fact-rule", action="store_true",
                        help="switch off the L1 rule that checks numbers against the vehicle state")

    sp = sub.add_parser("run", help="ask the assistant every test question")
    common(sp)
    run_args(sp)
    sp.set_defaults(func=cmd_run)

    sp = sub.add_parser("score", help="score every answer at L1, L2 and L3")
    common(sp)
    score_args(sp)
    sp.set_defaults(func=cmd_score)

    sp = sub.add_parser("report", help="compare selection strategies and write report.md")
    common(sp)
    sp.set_defaults(func=cmd_report)

    sp = sub.add_parser("all", help="run + score + report in one go")
    common(sp)
    run_args(sp)
    score_args(sp)
    sp.set_defaults(func=cmd_all)

    sp = sub.add_parser("label", help="label answers yourself to validate the judge")
    common(sp)
    sp.add_argument("--n", type=int, default=50)
    sp.set_defaults(func=cmd_label)
    return p


def main(argv: list[str] | None = None) -> None:
    pre = argparse.ArgumentParser(add_help=False)
    pre.add_argument("--config", type=Path, default=None)
    known, _ = pre.parse_known_args(argv)
    cfg = load_config(known.config)
    args = build_parser(cfg).parse_args(argv)
    args.func(args, cfg)


if __name__ == "__main__":
    main()
