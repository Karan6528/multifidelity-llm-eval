"""Agreement metrics: how far can each level be trusted?"""

from __future__ import annotations

import numpy as np
import pandas as pd


def cohen_kappa(a, b) -> float:
    """Cohen's kappa for two binary raters (1 = perfect agreement, 0 = chance level)."""
    a, b = np.asarray(a, bool), np.asarray(b, bool)
    if len(a) == 0:
        return float("nan")
    observed = float((a == b).mean())
    p_a, p_b = a.mean(), b.mean()
    expected = p_a * p_b + (1 - p_a) * (1 - p_b)
    if expected == 1.0:
        return 1.0
    return (observed - expected) / (1 - expected)


def agreement_table(df: pd.DataFrame, truth_col: str = "truth_fail") -> pd.DataFrame:
    """Compare each level's fail/pass verdict with the reference labels."""
    truth = df[truth_col].to_numpy(bool)
    rows = []
    levels = (("L1 rules", "l1_pass"), ("L2 similarity", "l2_pass"), ("L3 judge", "l3_pass"))
    for level, col in levels:
        flagged = ~df[col].to_numpy(bool)
        tp = int((flagged & truth).sum())
        rows.append({
            "level": level,
            "accuracy": float((flagged == truth).mean()),
            "recall": tp / truth.sum() if truth.sum() else 1.0,
            "precision": tp / flagged.sum() if flagged.sum() else 1.0,
            "kappa": cohen_kappa(flagged, truth),
        })
    return pd.DataFrame(rows)


def fault_catch_table(df: pd.DataFrame) -> pd.DataFrame:
    """For each injected fault type: share caught by each level on its own (mock runs only)."""
    faults = df[df["injected_fault"].notna() & (df["injected_fault"] != "")]
    rows = []
    for fault, g in faults.groupby("injected_fault"):
        rows.append({
            "fault": fault, "count": len(g),
            "L1 rules": float((~g["l1_pass"]).mean()),
            "L2 similarity": float((~g["l2_pass"]).mean()),
            "L3 judge": float((~g["l3_pass"]).mean()),
        })
    return pd.DataFrame(rows).sort_values("count", ascending=False)
