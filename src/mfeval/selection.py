"""Test selection across fidelity levels: which tests are worth sending to the judge?

Every test is scored at all three levels once ("offline replay"). Then each strategy
below is simulated on those scores: what it would have flagged, and what it would have
cost. That is much cheaper than re-running the pipeline per strategy, and it makes the
comparison fair, since all strategies see the same answers.

Costs are in abstract units per test (see configs/default.yaml). The reference cost is
running the judge on every test.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

DEFAULT_COSTS = {"L1": 1.0, "L2": 5.0, "L3": 100.0}


@dataclass
class StrategyResult:
    family: str
    label: str
    cost: float
    cost_pct: float
    recall: float
    precision: float
    flagged: int
    missed: int


def _score(df: pd.DataFrame, flagged: np.ndarray, cost: float, costs: dict, family: str,
           label: str) -> StrategyResult:
    truth = df["truth_fail"].to_numpy(bool)
    tp = int((flagged & truth).sum())
    n_fail = int(truth.sum())
    full_cost = costs["L3"] * len(df)
    return StrategyResult(
        family=family, label=label, cost=cost, cost_pct=100 * cost / full_cost,
        recall=tp / n_fail if n_fail else 1.0,
        precision=tp / int(flagged.sum()) if flagged.sum() else 1.0,
        flagged=int(flagged.sum()), missed=n_fail - tp,
    )


def run_strategies(df: pd.DataFrame, costs: dict | None = None,
                   n_random_seeds: int = 50) -> list[StrategyResult]:
    costs = {**DEFAULT_COSTS, **(costs or {})}
    n = len(df)
    l1 = df["l1_pass"].to_numpy(bool)
    l2 = df["l2_score"].to_numpy(float)
    l3 = df["l3_pass"].to_numpy(bool)
    n_l1_pass = int(l1.sum())
    base_cost = costs["L1"] * n  # rules always run on everything: they are practically free
    l2_cost = costs["L2"] * n_l1_pass  # L2 only for tests that passed the rules
    results: list[StrategyResult] = []

    # 1) Rules only.
    results.append(_score(df, ~l1, base_cost, costs, "Rules only (L1)", "all tests"))

    # 2) Rules + similarity threshold, no judge. Sweep the threshold.
    for t in np.round(np.arange(0.2, 0.91, 0.05), 2):
        flagged = ~l1 | (l1 & (l2 < t))
        results.append(_score(df, flagged, base_cost + l2_cost, costs, "Rules + similarity (L1+L2)",
                              f"threshold {t}"))

    # 3) Judge on everything: the expensive reference.
    results.append(_score(df, ~l3, costs["L3"] * n, costs, "Judge on everything (L3)", "all tests"))
    results.append(_score(df, ~l1 | ~l3, base_cost + costs["L3"] * n, costs,
                          "Rules + judge on everything", "all tests"))

    # 4) Rules, then the judge on a random share of what passed. Averaged over seeds.
    pass_idx = np.flatnonzero(l1)
    for share in (0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
        k = int(round(share * len(pass_idx)))
        runs = []
        for seed in range(n_random_seeds):
            chosen = np.zeros(n, bool)
            chosen[np.random.default_rng(seed).choice(pass_idx, size=k, replace=False)] = True
            runs.append(_score(df, ~l1 | (chosen & ~l3), base_cost + costs["L3"] * k, costs,
                               "", ""))
        results.append(StrategyResult(
            "Rules + judge on random sample", f"{int(share * 100)}% sampled",
            runs[0].cost, runs[0].cost_pct,
            float(np.mean([r.recall for r in runs])), float(np.mean([r.precision for r in runs])),
            int(np.mean([r.flagged for r in runs])), int(round(np.mean([r.missed for r in runs]))),
        ))

    # 5) Cascade: similarity decides the clear cases, the judge only sees the grey zone.
    for lo in (0.0, 0.1, 0.2, 0.3):
        for hi in (0.5, 0.6, 0.7, 0.8, 0.9):
            if hi <= lo:
                continue
            unsure = l1 & (l2 >= lo) & (l2 < hi)
            flagged = ~l1 | (l1 & (l2 < lo)) | (unsure & ~l3)
            cost = base_cost + l2_cost + costs["L3"] * int(unsure.sum())
            results.append(_score(df, flagged, cost, costs, "Cascade (L1 → L2 → L3 on grey zone)",
                                  f"grey zone {lo}–{hi}"))

    # 6) Budget-aware: spend a fixed judge budget on the lowest-similarity tests first.
    order = pass_idx[np.argsort(l2[pass_idx], kind="stable")]
    for share in (0.05, 0.1, 0.15, 0.2, 0.3, 0.4, 0.5, 0.6, 0.8):
        k = int(round(share * len(pass_idx)))
        judged = np.zeros(n, bool)
        judged[order[:k]] = True
        cost = base_cost + l2_cost + costs["L3"] * k
        results.append(_score(df, ~l1 | (judged & ~l3), cost, costs,
                              "Budget-aware (judge the least similar first)",
                              f"judge {int(share * 100)}% of L1 passes"))

    # 7) Rules as triage: on real data the rules raise false alarms (a "MBW Welt" the
    # driver is fine with, a polite refusal with the wrong intent label). So instead of
    # failing what the rules flag, send it to the judge to confirm. Optionally also
    # judge some rule passes, least similar first.
    n_l1_fail = int((~l1).sum())
    for share in (0.0, 0.1, 0.2, 0.3, 0.5):
        k = int(round(share * len(pass_idx)))
        judged = np.zeros(n, bool)
        judged[order[:k]] = True
        cost = base_cost + costs["L3"] * n_l1_fail + (l2_cost + costs["L3"] * k if k else 0)
        flagged = (~l1 & ~l3) | (judged & ~l3)
        label = "judge confirms rule fails" + (f" + {int(share * 100)}% of passes" if k else "")
        results.append(_score(df, flagged, cost, costs, "Rules triage, judge confirms", label))
    return results


# A strategy that flags half of the correct answers is useless: engineers would drown in
# false alarms. So only strategies whose flags are mostly real failures count as options.
MIN_PRECISION = 0.8


def usable(results: list[StrategyResult], min_precision: float = MIN_PRECISION):
    return [r for r in results if r.precision >= min_precision]


def pareto_front(results: list[StrategyResult],
                 min_precision: float = MIN_PRECISION) -> list[StrategyResult]:
    """Usable strategies that no other usable strategy beats on both cost and recall."""
    front, best = [], -1.0
    for r in sorted(usable(results, min_precision), key=lambda r: (r.cost, -r.recall)):
        if r.recall > best + 1e-9:
            front.append(r)
            best = r.recall
    return front


def cheapest_reaching(results: list[StrategyResult], recall: float,
                      min_precision: float = MIN_PRECISION) -> StrategyResult | None:
    ok = [r for r in usable(results, min_precision) if r.recall >= recall - 1e-9]
    return min(ok, key=lambda r: (r.cost, -r.recall)) if ok else None
