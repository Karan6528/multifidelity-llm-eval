"""Turn the scores into a markdown report and a cost-vs-recall chart."""

from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import pandas as pd  # noqa: E402

from .metrics import agreement_table, cohen_kappa, fault_catch_table  # noqa: E402
from .selection import (  # noqa: E402
    MIN_PRECISION,
    StrategyResult,
    cheapest_reaching,
    pareto_front,
    run_strategies,
)

COLORS = {
    "Rules only (L1)": "#6b7280",
    "Rules + similarity (L1+L2)": "#0ea5e9",
    "Judge on everything (L3)": "#111827",
    "Rules + judge on everything": "#dc2626",
    "Rules + judge on random sample": "#f59e0b",
    "Cascade (L1 → L2 → L3 on grey zone)": "#8b5cf6",
    "Budget-aware (judge the least similar first)": "#16a34a",
}


def md_table(df: pd.DataFrame, floatfmt: str = "{:.2f}") -> str:
    def fmt(v):
        return floatfmt.format(v) if isinstance(v, float) else str(v)

    lines = ["| " + " | ".join(df.columns) + " |", "|" + "---|" * len(df.columns)]
    lines += ["| " + " | ".join(fmt(v) for v in row) + " |" for row in df.itertuples(index=False)]
    return "\n".join(lines)


def plot_frontier(results: list[StrategyResult], path: Path, title: str) -> None:
    fig, ax = plt.subplots(figsize=(8, 5), dpi=150)
    for family, color in COLORS.items():
        pts = [r for r in results if r.family == family]
        if not pts:
            continue
        pts.sort(key=lambda r: r.cost_pct)
        size = 90 if len(pts) == 1 else 28
        marker = "D" if len(pts) == 1 else "o"
        good = [r for r in pts if r.precision >= MIN_PRECISION]
        bad = [r for r in pts if r.precision < MIN_PRECISION]
        ax.scatter([r.cost_pct for r in good], [100 * r.recall for r in good], s=size, c=color,
                   marker=marker, label=family, alpha=0.9, edgecolors="white", linewidths=0.6,
                   zorder=3)
        if bad:  # too many false alarms: drawn hollow
            ax.scatter([r.cost_pct for r in bad], [100 * r.recall for r in bad], s=size,
                       facecolors="none", edgecolors=color, marker=marker, alpha=0.6,
                       linewidths=1, zorder=3, label=None if good else family)
    front = pareto_front(results)
    ax.step([r.cost_pct for r in front], [100 * r.recall for r in front], where="post",
            color="#16a34a", alpha=0.35, linewidth=2, zorder=2,
            label="Best trade-off (Pareto front)")
    ax.set_xscale("log")
    ax.set_xlabel("Compute cost, % of running the judge on every test (log scale)")
    ax.set_ylabel("Failures caught (recall, %)")
    ax.set_title(title, fontsize=11, loc="left")
    ax.grid(True, which="both", alpha=0.25)
    low = min(100 * r.recall for r in results)
    ax.set_ylim(max(0, (low // 10) * 10 - 5), 102)
    ax.legend(fontsize=7.5, loc="lower right", frameon=False)
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    fig.tight_layout()
    fig.savefig(path)
    plt.close(fig)


def _strategy_rows(results: list[StrategyResult]) -> pd.DataFrame:
    return pd.DataFrame([{
        "strategy": r.family, "setting": r.label, "cost (% of full judge)": round(r.cost_pct, 1),
        "failures caught": f"{100 * r.recall:.1f}%", "precision": f"{100 * r.precision:.1f}%",
        "missed": r.missed,
    } for r in results])


def build_report(df: pd.DataFrame, run_dir: Path, truth_source: str, costs: dict,
                 labels: pd.DataFrame | None = None) -> dict:
    meta = json.loads((run_dir / "meta.json").read_text())
    results = run_strategies(df, costs)
    pd.DataFrame([r.__dict__ for r in results]).to_csv(run_dir / "strategies.csv", index=False)

    full = next(r for r in results if r.family == "Rules + judge on everything")
    judge_only = next(r for r in results if r.family == "Judge on everything (L3)")
    l1_only = next(r for r in results if r.family == "Rules only (L1)")
    match_full = cheapest_reaching(results, full.recall - 0.02)
    at_90 = cheapest_reaching(results, 0.90)

    plot_frontier(results, run_dir / "frontier.png",
                  f"Cost vs. failures caught  ·  run: {run_dir.name}  ·  {len(df)} tests")

    n_fail = int(df["truth_fail"].sum())
    lines = [
        f"# Evaluation report: `{run_dir.name}`",
        "",
        f"- Assistant: `{meta.get('assistant')}`  ·  L2 embedder: `{meta.get('embedder')}`  ·  "
        f"L3 judge: `{meta.get('judge')}`  ·  vehicle fact rule: "
        f"{'on' if meta.get('fact_rule', True) else 'off'}",
        f"- Tests: {len(df)}  ·  failed answers: {n_fail} ({100 * n_fail / len(df):.1f}%)  ·  "
        f"ground truth: {truth_source}",
        f"- Cost units per test: L1 = {costs['L1']:g}, L2 = {costs['L2']:g}, L3 = {costs['L3']:g}",
        "",
        "## Headline",
        "",
        f"- Rules + the judge on every test catch **{100 * full.recall:.1f}%** of failures "
        f"at {full.cost_pct:.0f}% cost (the judge alone: {100 * judge_only.recall:.1f}%).",
        f"- Rules alone catch {100 * l1_only.recall:.1f}% at {l1_only.cost_pct:.1f}% of the cost.",
    ]
    if match_full:
        lines.append(
            f"- **{match_full.family} ({match_full.label})** catches "
            f"**{100 * match_full.recall:.1f}%** of failures at **{match_full.cost_pct:.1f}%** "
            "of the cost, within 2 points of the full check.")
    if at_90 and at_90 is not match_full:
        lines.append(f"- Cheapest way to catch at least 90%: {at_90.family} ({at_90.label}), "
                     f"{100 * at_90.recall:.1f}% at {at_90.cost_pct:.1f}% of the cost.")
    lines += [
        "",
        "![Cost vs recall](frontier.png)",
        "",
        f"Hollow points flag too many correct answers (precision below {MIN_PRECISION:.0%}), "
        "so they don't count as options.",
        "",
        "## Best trade-offs (Pareto front)",
        "",
        f"Among strategies with at least {MIN_PRECISION:.0%} precision, nothing is both cheaper "
        "and better than these.",
        "",
        md_table(_strategy_rows(pareto_front(results))),
        "",
        "## What each level catches on its own",
        "",
        md_table(agreement_table(df)),
        "",
    ]
    if (df["injected_fault"] != "").any():
        catch = fault_catch_table(df)
        for col in ("L1 rules", "L2 similarity", "L3 judge"):
            catch[col] = catch[col].map(lambda v: f"{100 * v:.0f}%")
        lines += ["### By fault type (simulated assistant)", "", md_table(catch), ""]

    if labels is not None and len(labels):
        merged = df.merge(labels[["case_id", "human_fail"]], on="case_id")
        rows = []
        for level, col in (("L1 rules", "l1_pass"), ("L2 similarity", "l2_pass"),
                           ("L3 judge", "l3_pass")):
            flagged = ~merged[col]
            rows.append({"level": level,
                         "agreement": float((flagged == merged["human_fail"]).mean()),
                         "kappa": cohen_kappa(flagged, merged["human_fail"])})
        lines += [f"## Agreement with my own labels ({len(merged)} answers)", "",
                  md_table(pd.DataFrame(rows)), ""]

    by_cat = (df.groupby("category")["truth_fail"].agg(["count", "mean"]).reset_index()
              .rename(columns={"count": "tests", "mean": "failure rate"}))
    by_cat["failure rate"] = by_cat["failure rate"].map(lambda v: f"{100 * v:.0f}%")
    lines += ["## Where the assistant fails", "", md_table(by_cat), ""]
    if "l3_seconds" in df and df["l3_seconds"].sum() > 0:
        lines += [f"Measured time per test: L1 {1000 * df['l1_seconds'].mean():.2f} ms, "
                  f"L2 {1000 * df['l2_seconds'].mean():.1f} ms, "
                  f"L3 {df['l3_seconds'].mean():.2f} s.",
                  ""]

    (run_dir / "report.md").write_text("\n".join(lines), encoding="utf-8")
    return {"full": full, "match_full": match_full, "at_90": at_90, "l1_only": l1_only,
            "n_fail": n_fail}
