#!/usr/bin/env python3
"""Descriptive sensitivity of the four-edit aggregate to attack weights.

The script reweights existing per-attack means; it does not rerun watermark
embedding/detection and does not perform inferential testing. Erase40 receives
weight p, while Randomcrop50, Cornercrop50 and Retain20 share (1-p)/3.
Run from the repository root:
    python scripts/analyze_attack_weight_sensitivity.py
"""
from pathlib import Path
import csv
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
SOURCE = ROOT / "results_summaries" / "final_descriptive_by_attack.csv"
OUTPUT = ROOT / "results_summaries" / "attack_weight_sensitivity.csv"
ATTACKS = ["erase40", "randomcrop50", "cornercrop50", "splicing20"]
METHODS = ["single", "single_random", "random", "texture", "wam_random_mask", "hybrid"]
METRICS = {
    "BitAcc": "mean_vote_bitacc",
    "Recovery31": "mean_vote_recovery31",
}


def top_interval(values):
    """Return p interval where hybrid >= every comparator for linear scores."""
    h = values["hybrid"]
    h_other = sum(h[1:]) / 3.0
    lo, hi = 0.0, 1.0
    for method in METHODS:
        if method == "hybrid":
            continue
        b = values[method]
        d_erase = h[0] - b[0]
        d_other = h_other - sum(b[1:]) / 3.0
        slope = d_erase - d_other
        if abs(slope) < 1e-12:
            if d_other < 0:
                return None
            continue
        crossing = -d_other / slope
        if slope > 0:
            lo = max(lo, crossing)
        else:
            hi = min(hi, crossing)
    if lo > hi:
        return None
    return max(0.0, lo), min(1.0, hi)


def main():
    df = pd.read_csv(SOURCE)
    required = {"dataset", "method", "attack", *METRICS.values()}
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns in source summary: {sorted(missing)}")
    rows = []
    for dataset in ["COCO500", "DIV2K100"]:
        ds = df[df.dataset == dataset]
        if len(ds[ds.method.isin(METHODS) & ds.attack.isin(ATTACKS)]) != 24:
            raise ValueError(f"Expected 24 localized-policy attack rows for {dataset}")
        for metric_name, column in METRICS.items():
            values = {}
            for method in METHODS:
                sub = ds[ds.method == method].set_index("attack")
                if not set(ATTACKS).issubset(sub.index):
                    raise ValueError(f"Missing attack values for {dataset}/{method}")
                values[method] = [float(sub.loc[a, column]) for a in ATTACKS]
            interval = top_interval(values)
            if interval is None:
                raise RuntimeError(f"Hybrid has no top-ranked interval for {dataset}/{metric_name}")
            lo, hi = interval
            rows.append({
                "dataset": dataset,
                "metric": metric_name,
                "erase40_weight_lower": f"{lo:.6f}",
                "erase40_weight_upper": f"{hi:.6f}",
                "interpretation": "exploratory descriptive ranking; no inferential test",
            })
    with OUTPUT.open("w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)
    for row in rows:
        print(row)


if __name__ == "__main__":
    main()
