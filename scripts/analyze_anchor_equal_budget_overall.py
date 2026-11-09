import numpy as np
import pandas as pd
from pathlib import Path
from scipy.stats import wilcoxon

out = Path(
    "outputs/revision_202610/anchor_equal_budget_20261028"
)
src = out / "anchor_free_equal_budget_results.csv"

HYBRID = "hybrid_control"
FREE = "anchor_free_equal_budget_3sat"

attacks = [
    "erase40", "randomcrop50",
    "cornercrop50", "splicing20",
]
metrics = [
    "orig_bitacc", "vote_bitacc",
    "orig_recovery31", "vote_recovery31",
]

df = pd.read_csv(src)

df = df[df["attack"].isin(attacks)].copy()

# 每张图像先对四种攻击取平均，避免把 2,000 条攻击记录
# 错当成 2,000 个相互独立的样本。
per_image = (
    df.groupby(["method", "image"], as_index=False)[metrics]
      .mean()
)

per_image.to_csv(
    out / "anchor_equal_budget_four_attack_per_image.csv",
    index=False,
)

def holm_adjust(pvalues):
    p = np.asarray(pvalues, dtype=float)
    m = len(p)
    order = np.argsort(p)
    result = np.empty(m)
    running = 0.0

    for rank, idx in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[idx]))
        result[idx] = running

    return result

rng = np.random.default_rng(20261029)
N_BOOT = 10000
rows = []

for j, metric in enumerate(metrics):
    wide = per_image.pivot(
        index="image", columns="method", values=metric
    )

    if set(wide.columns) != {HYBRID, FREE}:
        raise ValueError(f"Unexpected methods for {metric}")

    wide = wide.dropna().sort_index()

    if len(wide) != 500:
        raise ValueError(
            f"{metric}: expected 500 paired images, got {len(wide)}"
        )

    diff = (
        wide[HYBRID].to_numpy()
        - wide[FREE].to_numpy()
    )

    boot = np.empty(N_BOOT)
    for start in range(0, N_BOOT, 500):
        n = min(500, N_BOOT - start)
        idx = rng.integers(0, len(diff), size=(n, len(diff)))
        boot[start:start+n] = diff[idx].mean(axis=1)

    lo, hi = np.quantile(boot, [0.025, 0.975])

    p = (
        1.0 if np.allclose(diff, 0)
        else float(wilcoxon(
            diff,
            alternative="two-sided",
            method="auto",
        ).pvalue)
    )

    rows.append({
        "metric": metric,
        "n_images": len(diff),
        "hybrid_mean": float(wide[HYBRID].mean()),
        "anchor_free_mean": float(wide[FREE].mean()),
        "mean_diff_hybrid_minus_free": float(diff.mean()),
        "ci95_low": float(lo),
        "ci95_high": float(hi),
        "p_raw": p,
    })

result = pd.DataFrame(rows)
result["p_holm"] = holm_adjust(result["p_raw"].to_numpy())

path = out / "anchor_equal_budget_overall_four_attack_tests.csv"
result.to_csv(path, index=False)

print("========== FOUR-ATTACK OVERALL PAIRED TESTS ==========")
print(result.to_string(index=False))
print("\nSaved:", path)
print("Bootstrap: paired image-level percentile CI95.")
print("Holm correction: four overall metric comparisons.")
print("Positive differences favor Hybrid.")
