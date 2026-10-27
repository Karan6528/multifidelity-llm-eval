import numpy as np
import pandas as pd

from mfeval.metrics import cohen_kappa
from mfeval.selection import cheapest_reaching, pareto_front, run_strategies


def toy_scores(n=100, seed=0):
    """Failures have low similarity; the judge is right; rules catch half of them."""
    rng = np.random.default_rng(seed)
    truth = rng.random(n) < 0.3
    l1_pass = ~(truth & (rng.random(n) < 0.5))
    l2 = np.where(truth, rng.uniform(0.0, 0.6, n), rng.uniform(0.4, 1.0, n))
    return pd.DataFrame({"truth_fail": truth, "l1_pass": l1_pass, "l2_score": l2,
                         "l2_pass": l2 >= 0.5, "l3_pass": ~truth})


def by_family(results, family):
    return [r for r in results if r.family == family]


def test_full_judge_is_reference_cost():
    results = run_strategies(toy_scores())
    full = by_family(results, "Judge on everything (L3)")[0]
    assert full.cost_pct == 100.0 and full.recall == 1.0


def test_more_budget_never_catches_less():
    results = by_family(run_strategies(toy_scores()),
                        "Budget-aware (judge the least similar first)")
    recalls = [r.recall for r in sorted(results, key=lambda r: r.cost)]
    assert all(b >= a for a, b in zip(recalls, recalls[1:]))


def test_budget_aware_beats_random_when_similarity_is_informative():
    results = run_strategies(toy_scores(n=400))
    budget = {r.label.split()[1]: r for r in by_family(
        results, "Budget-aware (judge the least similar first)")}
    random_ = {r.label.split()[0]: r for r in by_family(results, "Rules + judge on random sample")}
    assert budget["30%"].recall > random_["30%"].recall


def test_pareto_front_is_sorted_and_improving():
    front = pareto_front(run_strategies(toy_scores()))
    costs = [r.cost for r in front]
    recalls = [r.recall for r in front]
    assert costs == sorted(costs)
    assert all(b > a for a, b in zip(recalls, recalls[1:]))


def test_cheapest_reaching_respects_target():
    results = run_strategies(toy_scores())
    best = cheapest_reaching(results, 0.9)
    assert best is not None and best.recall >= 0.9 and best.precision >= 0.8


def test_kappa():
    assert cohen_kappa([1, 0, 1, 0], [1, 0, 1, 0]) == 1.0
    assert abs(cohen_kappa([1, 1, 0, 0], [1, 0, 1, 0])) < 1e-9


def test_triage_only_pays_the_judge_for_rule_failures():
    df = toy_scores()
    results = by_family(run_strategies(df), "Rules triage, judge confirms")
    base = next(r for r in results if r.label == "judge confirms rule fails")
    n_fail_l1 = int((~df["l1_pass"]).sum())
    assert base.cost == len(df) * 1 + 100 * n_fail_l1
    assert base.precision == 1.0  # the toy judge is always right, so no false alarms
