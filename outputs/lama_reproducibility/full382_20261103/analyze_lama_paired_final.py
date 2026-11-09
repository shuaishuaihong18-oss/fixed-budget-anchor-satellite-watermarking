import json
import itertools
import os
from pathlib import Path

import numpy as np
import pandas as pd

from scipy.stats import wilcoxon, binomtest

ROOT = Path(os.environ.get("WAM_ROOT", Path(__file__).resolve().parents[3])).resolve()
D = ROOT / "outputs/revision_202610/lama_object_inpainting_full382_20261103"

ATTACK_FILE = D / "lama_object_inpainting_382.csv"
CLEAN_FILE = D / "clean_matched_382.csv"

OUT = D / "final_statistics"
OUT.mkdir(exist_ok=True)

SEED = 20261104
N_BOOT = 10000

rng = np.random.default_rng(SEED)

# --------------------------------------------------
# 1. Load and validate
# --------------------------------------------------

att = pd.read_csv(ATTACK_FILE)
clean = pd.read_csv(CLEAN_FILE)

keys = [
    "image",
    "source_id",
    "eligible_index",
    "geometry_seed",
    "strategy",
]

required_att = keys + [
    "attack",
    "actual_area",
    "mask_frac",
    "overlap_of_watermark",
    "orig_bitacc",
    "vote_bitacc",
    "orig_recovery31",
    "vote_recovery31",
]

required_clean = keys + [
    "clean_orig_bitacc",
    "clean_vote_bitacc",
    "clean_orig_recovery31",
    "clean_vote_recovery31",
]

for c in required_att:
    assert c in att.columns, f"Missing attacked column: {c}"

for c in required_clean:
    assert c in clean.columns, f"Missing clean column: {c}"

assert not att.duplicated(keys).any()
assert not clean.duplicated(keys).any()

df = att.merge(
    clean[required_clean],
    on=keys,
    how="outer",
    validate="one_to_one",
    indicator=True,
)

assert (df["_merge"] == "both").all(), "Unmatched rows detected"

df.drop(columns="_merge", inplace=True)

assert len(df) == 2292
assert df["image"].nunique() == 382

STRATEGIES = [
    "hybrid",
    "single",
    "single_random",
    "random",
    "texture",
    "wam_random_mask",
]

assert set(df["strategy"].unique()) == set(STRATEGIES)

counts = df.groupby("strategy")["image"].nunique()
assert (counts == 382).all()

assert df[required_att + required_clean].notna().all().all()

# --------------------------------------------------
# 2. Calculate paired degradation
# --------------------------------------------------

df["vote_drop"] = (
    df["clean_vote_bitacc"] - df["vote_bitacc"]
)

df["orig_drop"] = (
    df["clean_orig_bitacc"] - df["orig_bitacc"]
)

df["vote_recovery_drop"] = (
    df["clean_vote_recovery31"] - df["vote_recovery31"]
)

df["orig_recovery_drop"] = (
    df["clean_orig_recovery31"] - df["orig_recovery31"]
)

# Positive drop = performance deterioration.
# Convert BitAcc differences into percentage points.

df["vote_drop_pp"] = df["vote_drop"] * 100
df["orig_drop_pp"] = df["orig_drop"] * 100

df["recovery_drop_pp"] = (
    df["vote_recovery_drop"] * 100
)

# A recovery loss means clean recovery succeeded,
# but recovery failed after LaMa attack.

df["recovery_lost"] = (
    (df["clean_vote_recovery31"] >= 0.5) &
    (df["vote_recovery31"] < 0.5)
).astype(int)

# --------------------------------------------------
# 3. Statistical functions
# --------------------------------------------------

def bootstrap_ci(x):
    x = np.asarray(x, dtype=float)
    n = len(x)

    if n == 0:
        return np.nan, np.nan

    samples = rng.integers(
        0, n, size=(N_BOOT, n)
    )

    means = x[samples].mean(axis=1)

    return tuple(
        np.quantile(means, [0.025, 0.975])
    )


def wilcoxon_p(x):
    x = np.asarray(x, dtype=float)

    if np.allclose(x, 0):
        return 1.0

    return float(
        wilcoxon(x, alternative="two-sided").pvalue
    )


def exact_mcnemar_p(a, b):
    a = np.asarray(a).astype(bool)
    b = np.asarray(b).astype(bool)

    n01 = int(np.sum(a & ~b))
    n10 = int(np.sum(~a & b))

    n = n01 + n10

    if n == 0:
        return 1.0, n01, n10

    p = binomtest(
        min(n01, n10),
        n,
        0.5,
        alternative="two-sided",
    ).pvalue

    return float(p), n01, n10


def holm_adjust(pvalues):
    pvalues = np.asarray(pvalues, dtype=float)

    m = len(pvalues)
    order = np.argsort(pvalues)

    adjusted = np.zeros(m)
    running = 0.0

    for rank, idx in enumerate(order):
        value = (m - rank) * pvalues[idx]
        running = max(running, value)
        adjusted[idx] = min(1.0, running)

    return adjusted


# --------------------------------------------------
# 4. Per-strategy results
# --------------------------------------------------

summary_rows = []

for strategy in STRATEGIES:

    g = df[df["strategy"] == strategy].sort_values("image")

    vote_drop = g["vote_drop_pp"].to_numpy()

    lo, hi = bootstrap_ci(vote_drop)

    recovery_drop = g["recovery_drop_pp"].to_numpy()

    rlo, rhi = bootstrap_ci(recovery_drop)

    rec_p, lost_n, gained_n = exact_mcnemar_p(
        g["clean_vote_recovery31"],
        g["vote_recovery31"],
    )

    summary_rows.append({
        "strategy": strategy,
        "n": len(g),

        "clean_vote_bitacc_pct":
            g["clean_vote_bitacc"].mean() * 100,

        "attacked_vote_bitacc_pct":
            g["vote_bitacc"].mean() * 100,

        "vote_drop_mean_pp":
            vote_drop.mean(),

        "vote_drop_median_pp":
            np.median(vote_drop),

        "vote_drop_ci95_low":
            lo,

        "vote_drop_ci95_high":
            hi,

        "vote_drop_wilcoxon_p":
            wilcoxon_p(vote_drop),

        "clean_recovery_rate_pct":
            g["clean_vote_recovery31"].mean() * 100,

        "attacked_recovery_rate_pct":
            g["vote_recovery31"].mean() * 100,

        "recovery_drop_pp":
            recovery_drop.mean(),

        "recovery_drop_ci95_low":
            rlo,

        "recovery_drop_ci95_high":
            rhi,

        "recovery_loss_count":
            int(g["recovery_lost"].sum()),

        "recovery_loss_rate_pct":
            g["recovery_lost"].mean() * 100,

        "recovery_mcnemar_p":
            rec_p,

        "recovery_gained_count":
            gained_n,

        "orig_clean_bitacc_pct":
            g["clean_orig_bitacc"].mean() * 100,

        "orig_attacked_bitacc_pct":
            g["orig_bitacc"].mean() * 100,

        "orig_drop_mean_pp":
            g["orig_drop_pp"].mean(),

        "mean_watermark_overlap_pct":
            g["overlap_of_watermark"].mean() * 100,

        "mean_object_area_pct":
            g["mask_frac"].mean() * 100,

        "mean_watermark_area_pct":
            g["actual_area"].mean() * 100,
    })


summary = pd.DataFrame(summary_rows)

summary["vote_drop_wilcoxon_holm_p"] = holm_adjust(
    summary["vote_drop_wilcoxon_p"]
)

summary["recovery_mcnemar_holm_p"] = holm_adjust(
    summary["recovery_mcnemar_p"]
)

summary.to_csv(
    OUT / "strategy_summary.csv",
    index=False,
)

# --------------------------------------------------
# 5. All paired strategy comparisons
# --------------------------------------------------

drop_pivot = df.pivot(
    index="image",
    columns="strategy",
    values="vote_drop_pp",
).loc[:, STRATEGIES]

loss_pivot = df.pivot(
    index="image",
    columns="strategy",
    values="recovery_lost",
).loc[:, STRATEGIES]

pair_rows = []

for a, b in itertools.combinations(STRATEGIES, 2):

    da = drop_pivot[a].to_numpy()
    db = drop_pivot[b].to_numpy()

    diff = da - db

    lo, hi = bootstrap_ci(diff)

    pa = wilcoxon_p(diff)

    la = loss_pivot[a].to_numpy()
    lb = loss_pivot[b].to_numpy()

    recovery_diff = (la - lb) * 100

    rlo, rhi = bootstrap_ci(recovery_diff)

    rp, discord_a, discord_b = exact_mcnemar_p(
        la, lb
    )

    pair_rows.append({
        "strategy_a": a,
        "strategy_b": b,

        "mean_drop_a_pp": da.mean(),
        "mean_drop_b_pp": db.mean(),

        "a_minus_b_drop_pp": diff.mean(),

        "drop_difference_ci95_low": lo,
        "drop_difference_ci95_high": hi,

        "bitacc_wilcoxon_p": pa,

        "loss_rate_a_pct": la.mean() * 100,
        "loss_rate_b_pct": lb.mean() * 100,

        "a_minus_b_loss_rate_pp":
            recovery_diff.mean(),

        "loss_difference_ci95_low": rlo,
        "loss_difference_ci95_high": rhi,

        "recovery_mcnemar_p": rp,

        "discordant_a_only": discord_a,
        "discordant_b_only": discord_b,
    })


pairs = pd.DataFrame(pair_rows)

pairs["bitacc_holm_p"] = holm_adjust(
    pairs["bitacc_wilcoxon_p"]
)

pairs["recovery_holm_p"] = holm_adjust(
    pairs["recovery_mcnemar_p"]
)

pairs.to_csv(
    OUT / "all_pairwise_comparisons.csv",
    index=False,
)

# --------------------------------------------------
# 6. Hybrid-focused comparison
# --------------------------------------------------

hybrid = pairs[
    pairs["strategy_a"] == "hybrid"
].copy()

hybrid.to_csv(
    OUT / "hybrid_vs_all_comparisons.csv",
    index=False,
)

# --------------------------------------------------
# 7. Save complete paired dataset
# --------------------------------------------------

df.to_csv(
    OUT / "paired_analysis_all_images.csv",
    index=False,
)

metadata = {
    "experiment": "LaMa object inpainting paired analysis",
    "images": int(df["image"].nunique()),
    "strategies": STRATEGIES,
    "rows": int(len(df)),
    "primary_metric": "vote_bitacc",
    "paired_degradation": "clean minus attacked",
    "positive_degradation": "performance deterioration",
    "bootstrap_repetitions": N_BOOT,
    "bootstrap_seed": SEED,
    "confidence_interval": "95% percentile paired bootstrap",
    "bitacc_test": "Wilcoxon signed-rank",
    "recovery_test": "Exact McNemar",
    "multiple_comparison": "Holm correction",
    "pairwise_tests": len(pairs),
}

(OUT / "metadata.json").write_text(
    json.dumps(metadata, indent=2),
    encoding="utf-8",
)

# --------------------------------------------------
# 8. Print results
# --------------------------------------------------

print("\n" + "=" * 80)
print("LA MA PAIRED ANALYSIS COMPLETE")
print("=" * 80)

print("\n1. STRATEGY SUMMARY")

cols = [
    "strategy",
    "clean_vote_bitacc_pct",
    "attacked_vote_bitacc_pct",
    "vote_drop_mean_pp",
    "vote_drop_ci95_low",
    "vote_drop_ci95_high",
    "vote_drop_wilcoxon_holm_p",
    "recovery_loss_rate_pct",
]

print(
    summary[cols].round(4).to_string(index=False)
)

print("\n2. HYBRID VS OTHER STRATEGIES")

cols2 = [
    "strategy_a",
    "strategy_b",
    "mean_drop_a_pp",
    "mean_drop_b_pp",
    "a_minus_b_drop_pp",
    "drop_difference_ci95_low",
    "drop_difference_ci95_high",
    "bitacc_holm_p",
    "a_minus_b_loss_rate_pp",
    "recovery_holm_p",
]

print(
    hybrid[cols2].round(4).to_string(index=False)
)

print("\n3. INTERPRETATION")

print(
    "For strategy_a minus strategy_b degradation:"
)

print(
    "Negative = strategy_a suffers LESS degradation."
)

print(
    "Positive = strategy_a suffers MORE degradation."
)

print(
    "A Holm-adjusted p < 0.05 indicates a statistically "
    "significant paired difference."
)

print("\n4. OUTPUT FILES")

for p in sorted(OUT.iterdir()):
    if p.is_file():
        print(p.name, "|", p.stat().st_size, "bytes")

print("\nFINAL ANALYSIS: PASS")
