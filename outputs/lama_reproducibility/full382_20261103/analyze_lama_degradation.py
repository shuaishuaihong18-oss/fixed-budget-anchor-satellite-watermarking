from pathlib import Path
from itertools import combinations

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

D = Path(
    "outputs/revision_202610/"
    "lama_object_inpainting_full382_20261103"
)

OUT = D / "statistical_analysis"
OUT.mkdir(exist_ok=True)

df = pd.read_csv(D / "clean_vs_attacked_382.csv")

STRATEGIES = [
    "single", "single_random", "random",
    "texture", "wam_random_mask", "hybrid"
]

METRICS = {
    "Original BitAcc": "delta_orig_bitacc",
    "Voting BitAcc": "delta_vote_bitacc",
    "Original Recovery": "delta_orig_recovery31",
    "Voting Recovery": "delta_vote_recovery31",
}

assert len(df) == 2292
assert df["image"].nunique() == 382

rng = np.random.default_rng(20261103)
N_BOOT = 5000


def holm(p):
    p = np.asarray(p, dtype=float)
    order = np.argsort(p)
    out = np.empty(len(p))
    running = 0.0

    for rank, idx in enumerate(order):
        running = max(
            running,
            (len(p) - rank) * p[idx]
        )
        out[idx] = min(running, 1.0)

    return out


def test_wilcoxon(x):
    x = np.asarray(x, dtype=float)

    if np.allclose(x, 0):
        return 0.0, 1.0

    r = wilcoxon(
        x,
        zero_method="pratt",
        alternative="two-sided",
        method="auto"
    )

    return float(r.statistic), float(r.pvalue)


def ci95(x):
    x = np.asarray(x, dtype=float)
    n = len(x)

    idx = rng.integers(
        0, n, size=(N_BOOT, n)
    )

    means = x[idx].mean(axis=1)

    return np.quantile(means, [0.025, 0.975])


summary = []
friedman = []
pairs = []

for metric, col in METRICS.items():

    wide = df.pivot(
        index="image",
        columns="strategy",
        values=col
    ).loc[:, STRATEGIES]

    assert wide.shape == (382, 6)
    assert not wide.isna().any().any()

    arrays = [
        wide[s].to_numpy(dtype=float)
        for s in STRATEGIES
    ]

    # Strategy-specific degradation
    for s in STRATEGIES:
        x = wide[s].to_numpy(dtype=float)
        lo, hi = ci95(x)

        summary.append({
            "metric": metric,
            "strategy": s,
            "mean_degradation": x.mean(),
            "ci95_low": lo,
            "ci95_high": hi,
        })

    # Friedman test on matched image-level degradation
    if all(
        np.array_equal(arrays[0], a)
        for a in arrays[1:]
    ):
        stat, p = 0.0, 1.0
    else:
        r = friedmanchisquare(*arrays)
        stat, p = float(r.statistic), float(r.pvalue)

    friedman.append({
        "metric": metric,
        "friedman_statistic": stat,
        "p_raw": p,
    })

    # Pairwise comparisons of degradation
    for a, b in combinations(STRATEGIES, 2):

        diff = (
            wide[a].to_numpy(dtype=float)
            - wide[b].to_numpy(dtype=float)
        )

        stat, p = test_wilcoxon(diff)
        lo, hi = ci95(diff)

        pairs.append({
            "metric": metric,
            "strategy_A": a,
            "strategy_B": b,
            "mean_delta_difference_A_minus_B": diff.mean(),
            "ci95_low": lo,
            "ci95_high": hi,
            "wilcoxon_stat": stat,
            "p_raw": p,
        })

summary_df = pd.DataFrame(summary)
friedman_df = pd.DataFrame(friedman)
pairs_df = pd.DataFrame(pairs)

friedman_df["p_holm"] = holm(
    friedman_df["p_raw"]
)

# Holm correction across all 60 pairwise tests
pairs_df["p_holm"] = holm(
    pairs_df["p_raw"]
)

summary_df.to_csv(
    OUT / "strategy_degradation_summary.csv",
    index=False
)

friedman_df.to_csv(
    OUT / "strategy_degradation_friedman.csv",
    index=False
)

pairs_df.to_csv(
    OUT / "strategy_degradation_posthoc.csv",
    index=False
)

print("\n========== DEGRADATION SUMMARY ==========")
print(summary_df.round(5).to_string(index=False))

print("\n========== FRIEDMAN ON DEGRADATION ==========")
print(friedman_df.round(7).to_string(index=False))

print("\n========== PAIRED POST-HOC ==========")
print(
    pairs_df.sort_values(
        ["metric", "p_holm"]
    ).round(6).to_string(index=False)
)

print("\n========== OUTPUT ==========")
print("Images:", df.image.nunique())
print("Rows:", len(df))
print("Pairwise comparisons:", len(pairs_df))
print("PASS")
