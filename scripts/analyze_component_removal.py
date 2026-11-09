from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

D = Path(__file__).resolve().parent

SRC = D / "component_removal_attacks_500.csv"
OUT = D / "equal_area_paired_results.csv"

if OUT.exists():
    raise FileExistsError(OUT)

df = pd.read_csv(SRC)

conditions = [
    "anchor_removed",
    "remove_S1_S2",
    "remove_S1_S3",
    "remove_S2_S3",
]

pairs = [
    "remove_S1_S2",
    "remove_S1_S3",
    "remove_S2_S3",
]

attacks = [
    "erase40",
    "randomcrop50",
    "cornercrop50",
    "splicing20",
]

metrics = [
    "orig_bitacc",
    "vote_bitacc",
    "orig_recovery31",
    "vote_recovery31",
]

assert len(df) == 20000
assert df.image.nunique() == 500
assert not df.duplicated(
    ["image", "condition", "attack"]
).any()

assert set(conditions).issubset(set(df.condition))

# Verify identical attack seeds across conditions.
assert (
    df.groupby(["image", "attack"])["attack_seed"]
      .nunique()
      .max() == 1
)

for c in ["actual_area"] + metrics:
    df[c] = pd.to_numeric(df[c])

area = df.pivot(
    index=["image", "attack"],
    columns="condition",
    values="actual_area"
)

results = []
rng = np.random.default_rng(20261030)

print("=" * 70)
print("NEAR-EQUAL-AREA PAIRED COMPARISON")
print("=" * 70)

for metric in metrics:

    pivot = df.pivot(
        index=["image", "attack"],
        columns="condition",
        values=metric
    )

    for candidate in pairs:

        for attack in attacks:

            x = pivot.xs(
                attack, level="attack"
            )

            a = x[candidate].to_numpy()
            b = x["anchor_removed"].to_numpy()

            diff = a - b
            n = len(diff)

            assert n == 500

            # Paired bootstrap CI.
            indices = rng.integers(
                0, n, size=(10000, n)
            )

            boot = diff[indices].mean(axis=1)

            ci_low, ci_high = np.quantile(
                boot, [0.025, 0.975]
            )

            if np.allclose(diff, 0):
                p = 1.0
            else:
                p = float(wilcoxon(diff).pvalue)

            ar = area.xs(
                attack, level="attack"
            )

            area_diff = (
                ar[candidate].to_numpy()
                - ar["anchor_removed"].to_numpy()
            ) * 100

            results.append({
                "candidate": candidate,
                "reference": "anchor_removed",
                "attack": attack,
                "metric": metric,
                "n_images": n,
                "mean_diff_candidate_minus_reference":
                    float(diff.mean()),
                "ci95_low": float(ci_low),
                "ci95_high": float(ci_high),
                "p_raw": p,
                "mean_area_diff_pp": float(
                    area_diff.mean()
                ),
                "mean_abs_area_diff_pp": float(
                    np.abs(area_diff).mean()
                ),
                "p95_abs_area_diff_pp": float(
                    np.quantile(np.abs(area_diff), 0.95)
                ),
                "max_abs_area_diff_pp": float(
                    np.abs(area_diff).max()
                ),
            })

out = pd.DataFrame(results)

assert len(out) == 48

# Global Holm correction.
p = out.p_raw.to_numpy()
order = np.argsort(p, kind="mergesort")

adjusted = np.empty(len(p))
running = 0.0
m = len(p)

for rank, idx in enumerate(order):
    running = max(
        running,
        (m - rank) * p[idx]
    )
    adjusted[idx] = min(1.0, running)

out["p_holm_global48"] = adjusted
out["significant"] = adjusted < 0.05

out.to_csv(OUT, index=False)

print("\n========== RESULTS ==========")
print(
    out[[
        "candidate",
        "attack",
        "metric",
        "mean_diff_candidate_minus_reference",
        "ci95_low",
        "ci95_high",
        "p_holm_global48",
        "mean_area_diff_pp",
        "p95_abs_area_diff_pp",
    ]].round(6).to_string(index=False)
)

print("\n========== FINAL SUMMARY ==========")
print("Comparisons:", len(out))
print(
    "Significant after global Holm:",
    int(out.significant.sum())
)
print(
    "Non-significant:",
    int((~out.significant).sum())
)

print("\nSaved:", OUT)
print("CPU analysis only. No GPU experiment performed.")
