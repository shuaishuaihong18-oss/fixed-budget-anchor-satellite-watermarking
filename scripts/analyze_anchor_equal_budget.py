import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

OUT = Path(
    "outputs/revision_202610/anchor_equal_budget_20261028"
)
SOURCE = OUT / "anchor_free_equal_budget_results.csv"

HYBRID = "hybrid_control"
FREE = "anchor_free_equal_budget_3sat"

ATTACKS = [
    "none", "erase40", "randomcrop50",
    "cornercrop50", "splicing20",
]

ROBUST_METRICS = [
    "orig_bitacc", "vote_bitacc",
    "orig_recovery31", "vote_recovery31",
]
QUALITY_METRICS = ["psnr", "ssim"]

N_BOOT = 10000
BOOT_SEED = 20261028


def bootstrap_ci(diff, seed):
    """Image-level paired percentile bootstrap CI for mean difference."""
    d = np.asarray(diff, dtype=float)
    if len(d) == 0 or not np.isfinite(d).all():
        raise ValueError("Invalid paired differences")

    rng = np.random.default_rng(seed)
    n = len(d)
    boot_means = np.empty(N_BOOT)

    # Batch to keep memory use modest.
    batch = 500
    for start in range(0, N_BOOT, batch):
        size = min(batch, N_BOOT - start)
        idx = rng.integers(0, n, size=(size, n))
        boot_means[start:start + size] = d[idx].mean(axis=1)

    lo, hi = np.quantile(boot_means, [0.025, 0.975])
    return float(lo), float(hi)


def paired_test(x, y, seed):
    """All differences are defined as Hybrid minus Anchor-free."""
    x = np.asarray(x, dtype=float)
    y = np.asarray(y, dtype=float)

    if len(x) != len(y) or not np.isfinite(x).all() \
            or not np.isfinite(y).all():
        raise ValueError("Invalid or unpaired input")

    d = x - y
    lo, hi = bootstrap_ci(d, seed)

    if np.allclose(d, 0):
        p = 1.0
    else:
        p = float(wilcoxon(
            d,
            zero_method="wilcox",
            alternative="two-sided",
            method="auto",
        ).pvalue)

    return {
        "n_images": len(d),
        "hybrid_mean": float(x.mean()),
        "anchor_free_mean": float(y.mean()),
        "mean_diff_hybrid_minus_free": float(d.mean()),
        "ci95_low": lo,
        "ci95_high": hi,
        "p_raw": p,
    }


def holm_adjust(pvalues):
    """Holm-Bonferroni adjustment without extra dependencies."""
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    adjusted = np.empty(m, dtype=float)
    running_max = 0.0

    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * p[idx])
        running_max = max(running_max, value)
        adjusted[idx] = running_max

    return adjusted


if not SOURCE.exists():
    raise FileNotFoundError(SOURCE)

df = pd.read_csv(SOURCE)

required = {
    "image", "method", "attack", "actual_area",
    "psnr", "ssim", *ROBUST_METRICS,
}
missing = required - set(df.columns)
if missing:
    raise ValueError(f"Missing columns: {sorted(missing)}")

if set(df["method"].unique()) != {HYBRID, FREE}:
    raise ValueError("Unexpected method labels")

if set(df["attack"].unique()) != set(ATTACKS):
    raise ValueError("Unexpected attack labels")

if df.duplicated(["method", "image", "attack"]).any():
    raise ValueError("Duplicate method-image-attack keys")

hy = (
    df[df["method"] == HYBRID]
    .set_index(["image", "attack"])
    .sort_index()
)
fr = (
    df[df["method"] == FREE]
    .set_index(["image", "attack"])
    .sort_index()
)

if not hy.index.equals(fr.index):
    raise ValueError("Methods do not have identical paired keys")

if len(hy) != 2500 or hy.index.get_level_values("image").nunique() != 500:
    raise ValueError("Unexpected sample size")

# Verify all 500 images have all five attack conditions.
for method in [HYBRID, FREE]:
    sub = df[df["method"] == method]
    counts = sub.groupby("image")["attack"].nunique()
    if len(counts) != 500 or not (counts == 5).all():
        raise ValueError(f"Incomplete image/attack coverage: {method}")

print("PASS: 500 paired images and five conditions per method.")
print("Direction: positive difference means Hybrid is better.")

# 1. Descriptive results by attack.
desc_rows = []
for attack in ATTACKS:
    h = hy.xs(attack, level="attack").sort_index()
    f = fr.xs(attack, level="attack").sort_index()

    if not h.index.equals(f.index):
        raise ValueError(f"Unmatched images for attack {attack}")

    for metric in [
        "actual_area", "psnr", "ssim", *ROBUST_METRICS
    ]:
        desc_rows.append({
            "attack": attack,
            "metric": metric,
            "n_images": len(h),
            "hybrid_mean": float(h[metric].mean()),
            "anchor_free_mean": float(f[metric].mean()),
            "mean_diff_hybrid_minus_free": float(
                (h[metric] - f[metric]).mean()
            ),
        })

desc = pd.DataFrame(desc_rows)
desc.to_csv(
    OUT / "anchor_equal_budget_descriptive_by_attack.csv",
    index=False,
)

# 2. Robustness paired tests: 5 attacks x 4 metrics = 20 tests.
robust_rows = []
counter = 0

for attack in ATTACKS:
    h = hy.xs(attack, level="attack").sort_index()
    f = fr.xs(attack, level="attack").sort_index()

    for metric in ROBUST_METRICS:
        result = paired_test(
            h[metric].to_numpy(),
            f[metric].to_numpy(),
            BOOT_SEED + counter,
        )
        robust_rows.append({
            "attack": attack,
            "metric": metric,
            **result,
        })
        counter += 1

robust = pd.DataFrame(robust_rows)
robust["p_holm"] = holm_adjust(robust["p_raw"].to_numpy())

robust.to_csv(
    OUT / "anchor_equal_budget_paired_tests_holm.csv",
    index=False,
)

# 3. Quality: use attack='none' once per image.
# Do NOT count the repeated quality values from all five attack rows.
h0 = hy.xs("none", level="attack").sort_index()
f0 = fr.xs("none", level="attack").sort_index()

quality_rows = []
for i, metric in enumerate(QUALITY_METRICS):
    result = paired_test(
        h0[metric].to_numpy(),
        f0[metric].to_numpy(),
        BOOT_SEED + 100 + i,
    )
    quality_rows.append({"metric": metric, **result})

quality = pd.DataFrame(quality_rows)
quality["p_holm"] = holm_adjust(quality["p_raw"].to_numpy())

quality.to_csv(
    OUT / "anchor_equal_budget_quality_tests_holm.csv",
    index=False,
)

# 4. Area and quality summary, using one clean row per image.
summary_rows = []
for method, sub in [
    (HYBRID, h0),
    (FREE, f0),
]:
    summary_rows.append({
        "method": method,
        "n_images": len(sub),
        "mean_actual_area": float(sub["actual_area"].mean()),
        "mean_psnr": float(sub["psnr"].mean()),
        "mean_ssim": float(sub["ssim"].mean()),
    })

summary = pd.DataFrame(summary_rows)
summary.to_csv(
    OUT / "anchor_equal_budget_area_quality_summary.csv",
    index=False,
)

area_result = paired_test(
    h0["actual_area"].to_numpy(),
    f0["actual_area"].to_numpy(),
    BOOT_SEED + 200,
)
pd.DataFrame([area_result]).to_csv(
    OUT / "anchor_equal_budget_area_paired_summary.csv",
    index=False,
)

print("\n========== AREA AND QUALITY SUMMARY ==========")
print(summary.to_string(index=False))

print("\n========== PAIRED AREA DIFFERENCE ==========")
print(pd.DataFrame([area_result]).to_string(index=False))

print("\n========== ROBUSTNESS PAIRED TESTS ==========")
print(
    robust[
        ["attack", "metric", "mean_diff_hybrid_minus_free",
         "ci95_low", "ci95_high", "p_raw", "p_holm"]
    ].round(6).to_string(index=False)
)

print("\n========== QUALITY PAIRED TESTS ==========")
print(
    quality[
        ["metric", "mean_diff_hybrid_minus_free",
         "ci95_low", "ci95_high", "p_raw", "p_holm"]
    ].round(6).to_string(index=False)
)

print("\n========== OUTPUT FILES ==========")
for name in [
    "anchor_equal_budget_descriptive_by_attack.csv",
    "anchor_equal_budget_paired_tests_holm.csv",
    "anchor_equal_budget_quality_tests_holm.csv",
    "anchor_equal_budget_area_quality_summary.csv",
    "anchor_equal_budget_area_paired_summary.csv",
]:
    print(OUT / name)

print("\nPASS: all paired analyses completed.")
print("Bootstrap: image-level paired percentile CI, 95%.")
print("Holm correction: 20 robustness tests; 2 quality tests.")
