import itertools
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon, friedmanchisquare

D = Path(
    "outputs/revision_202610/"
    "lama_object_inpainting_full382_20261103"
)

P = D / "clean_vs_attacked_382.csv"
OUT = D / "statistical_analysis"
OUT.mkdir(exist_ok=True)

STRATEGIES = [
    "single",
    "single_random",
    "random",
    "texture",
    "wam_random_mask",
    "hybrid",
]

METRICS = [
    ("Original BitAcc", "clean_orig_bitacc", "orig_bitacc"),
    ("Voting BitAcc", "clean_vote_bitacc", "vote_bitacc"),
    ("Original Recovery", "clean_orig_recovery31", "orig_recovery31"),
    ("Voting Recovery", "clean_vote_recovery31", "vote_recovery31"),
]

N_BOOT = 5000
rng = np.random.default_rng(20261103)


def holm_adjust(p_values):
    p = np.asarray(p_values, dtype=float)
    n = len(p)
    order = np.argsort(p)
    adjusted = np.empty(n)
    running = 0.0

    for rank, idx in enumerate(order):
        running = max(
            running,
            (n - rank) * p[idx]
        )
        adjusted[idx] = min(running, 1.0)

    return adjusted


def paired_wilcoxon(x):
    x = np.asarray(x, dtype=float)

    if np.allclose(x, 0):
        return 0.0, 1.0

    result = wilcoxon(
        x,
        zero_method="pratt",
        alternative="two-sided",
        method="auto"
    )

    return float(result.statistic), float(result.pvalue)


def bootstrap_ci(x):
    x = np.asarray(x, dtype=float)
    n = len(x)

    indices = rng.integers(
        0, n, size=(N_BOOT, n)
    )

    boot_means = x[indices].mean(axis=1)

    low, high = np.quantile(
        boot_means, [0.025, 0.975]
    )

    return float(low), float(high)


print("=" * 75)
print("1. DATA INTEGRITY")
print("=" * 75)

df = pd.read_csv(P)

assert len(df) == 2292
assert df["image"].nunique() == 382
assert df["strategy"].nunique() == 6

assert not df.duplicated(
    ["image", "strategy"]
).any()

assert df.groupby("image")["strategy"].nunique().eq(6).all()

assert df.groupby("strategy")["image"].nunique().eq(382).all()

assert not df.isna().any().any()

for name, clean_col, attack_col in METRICS:
    assert clean_col in df.columns
    assert attack_col in df.columns

    delta_col = (
        "delta_"
        + ("orig" if "Original" in name else "vote")
        + "_"
        + ("bitacc" if "BitAcc" in name else "recovery31")
    )

    expected_delta = (
        df[attack_col] - df[clean_col]
    )

    assert np.allclose(
        df[delta_col],
        expected_delta,
        atol=1e-10
    )

print("Rows:", len(df))
print("Images:", df.image.nunique())
print("Strategies:", df.strategy.nunique())
print("Missing:", int(df.isna().sum().sum()))
print("Duplicate keys:", df.duplicated(
    ["image", "strategy"]
).sum())

# ---------------------------------------------------------
# 2. PAIRED CLEAN VS ATTACKED
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("2. PAIRED CLEAN VS ATTACKED")
print("=" * 75)

attack_results = []

for strategy in STRATEGIES:

    sub = df[
        df.strategy == strategy
    ].sort_values("image")

    for name, clean_col, attack_col in METRICS:

        clean = sub[clean_col].to_numpy(dtype=float)
        attacked = sub[attack_col].to_numpy(dtype=float)

        # Positive delta = improvement after attack.
        # Negative delta = degradation after attack.
        delta = attacked - clean

        mean_delta = float(delta.mean())

        ci_low, ci_high = bootstrap_ci(delta)

        stat, p = paired_wilcoxon(delta)

        attack_results.append({
            "strategy": strategy,
            "metric": name,
            "n_images": len(delta),
            "clean_mean": float(clean.mean()),
            "attacked_mean": float(attacked.mean()),
            "delta_attack_minus_clean": mean_delta,
            "delta_ci95_low": ci_low,
            "delta_ci95_high": ci_high,
            "wilcoxon_stat": stat,
            "p_raw": p,
        })

attack_df = pd.DataFrame(attack_results)

# One Holm family: 6 strategies x 4 endpoints.
attack_df["p_holm"] = holm_adjust(
    attack_df["p_raw"].values
)

attack_df.to_csv(
    OUT / "paired_attack_effect.csv",
    index=False
)

print(
    attack_df[
        [
            "strategy",
            "metric",
            "clean_mean",
            "attacked_mean",
            "delta_attack_minus_clean",
            "delta_ci95_low",
            "delta_ci95_high",
            "p_holm",
        ]
    ].round(5).to_string(index=False)
)

# ---------------------------------------------------------
# 3. FRIEDMAN OMNIBUS TEST
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("3. FRIEDMAN TEST: SIX STRATEGIES")
print("=" * 75)

friedman_results = []

for name, clean_col, attack_col in METRICS:

    wide = df.pivot(
        index="image",
        columns="strategy",
        values=attack_col
    ).loc[:, STRATEGIES]

    arrays = [
        wide[s].to_numpy(dtype=float)
        for s in STRATEGIES
    ]

    if all(
        np.array_equal(arrays[0], a)
        for a in arrays[1:]
    ):
        statistic, p = 0.0, 1.0
    else:
        result = friedmanchisquare(*arrays)
        statistic = float(result.statistic)
        p = float(result.pvalue)

    friedman_results.append({
        "metric": name,
        "statistic": statistic,
        "p_raw": p,
    })

friedman_df = pd.DataFrame(friedman_results)

friedman_df["p_holm"] = holm_adjust(
    friedman_df["p_raw"].values
)

friedman_df.to_csv(
    OUT / "friedman_strategy_test.csv",
    index=False
)

print(friedman_df.round(6).to_string(index=False))

# ---------------------------------------------------------
# 4. PAIRED POST-HOC STRATEGY COMPARISONS
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("4. PAIRED POST-HOC COMPARISONS")
print("=" * 75)

pair_results = []

for name, clean_col, attack_col in METRICS:

    wide = df.pivot(
        index="image",
        columns="strategy",
        values=attack_col
    ).loc[:, STRATEGIES]

    for a, b in itertools.combinations(STRATEGIES, 2):

        xa = wide[a].to_numpy(dtype=float)
        xb = wide[b].to_numpy(dtype=float)

        diff = xa - xb

        stat, p = paired_wilcoxon(diff)

        pair_results.append({
            "metric": name,
            "strategy_A": a,
            "strategy_B": b,
            "mean_difference_A_minus_B": float(diff.mean()),
            "wilcoxon_stat": stat,
            "p_raw": p,
        })

pair_df = pd.DataFrame(pair_results)

# Conservative Holm correction across all 60 tests.
pair_df["p_holm"] = holm_adjust(
    pair_df["p_raw"].values
)

pair_df.to_csv(
    OUT / "strategy_pairwise_posthoc.csv",
    index=False
)

print(
    pair_df.sort_values(
        ["metric", "p_holm"]
    ).round(6).to_string(index=False)
)

# ---------------------------------------------------------
# 5. DESCRIPTIVE STRATEGY SUMMARY
# ---------------------------------------------------------

print("\n" + "=" * 75)
print("5. FINAL DESCRIPTIVE SUMMARY")
print("=" * 75)

summary = df.groupby("strategy").agg(
    images=("image", "nunique"),

    clean_orig_bitacc=("clean_orig_bitacc", "mean"),
    attacked_orig_bitacc=("orig_bitacc", "mean"),

    clean_vote_bitacc=("clean_vote_bitacc", "mean"),
    attacked_vote_bitacc=("vote_bitacc", "mean"),

    clean_orig_recovery=("clean_orig_recovery31", "mean"),
    attacked_orig_recovery=("orig_recovery31", "mean"),

    clean_vote_recovery=("clean_vote_recovery31", "mean"),
    attacked_vote_recovery=("vote_recovery31", "mean"),
).reindex(STRATEGIES)

summary.to_csv(
    OUT / "strategy_descriptive_summary.csv"
)

print(summary.round(5).to_string())

print("\n" + "=" * 75)
print("STATISTICAL ANALYSIS COMPLETE")
print("=" * 75)

print("Output directory:", OUT)
print("1. paired_attack_effect.csv")
print("2. friedman_strategy_test.csv")
print("3. strategy_pairwise_posthoc.csv")
print("4. strategy_descriptive_summary.csv")

print("PASS")
