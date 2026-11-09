import csv
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path("outputs/revision_202610/quality_only_20261027")
SOURCE = ROOT / "quality_per_image.csv"
OUT = ROOT / "quality_paired_tests_holm.csv"

DATASETS = ["COCO500", "DIV2K100"]
BASELINES = [
    "single", "single_random", "random",
    "texture", "wam_random_mask",
]
METRICS = ["psnr", "ssim", "lpips"]
N_BOOT = 5000

rng = np.random.default_rng(20261028)

with SOURCE.open(newline="") as f:
    rows = list(csv.DictReader(f))

# Index by dataset, image, and method.
index = {
    (r["dataset"], r["image"], r["method"]): r
    for r in rows
}

results = []

for dataset in DATASETS:
    images = sorted({
        r["image"] for r in rows if r["dataset"] == dataset
    })

    for metric in METRICS:
        for baseline in BASELINES:
            hybrid_values = []
            baseline_values = []

            for image in images:
                a = index[(dataset, image, "hybrid")][metric]
                b = index[(dataset, image, baseline)][metric]

                if not a.strip() or not b.strip():
                    continue

                a, b = float(a), float(b)
                if np.isfinite(a) and np.isfinite(b):
                    hybrid_values.append(a)
                    baseline_values.append(b)

            a = np.asarray(hybrid_values)
            b = np.asarray(baseline_values)
            diff = a - b

            if len(diff) < 2:
                raise RuntimeError(
                    f"Insufficient paired observations: "
                    f"{dataset}, {metric}, {baseline}"
                )

            test = wilcoxon(
                diff,
                alternative="two-sided",
                zero_method="pratt",
                method="auto",
            )

            boot = rng.choice(
                diff,
                size=(N_BOOT, len(diff)),
                replace=True,
            ).mean(axis=1)

            lo, hi = np.quantile(boot, [0.025, 0.975])

            results.append({
                "dataset": dataset,
                "metric": metric,
                "comparison": f"hybrid_vs_{baseline}",
                "n": len(diff),
                "mean_diff": float(diff.mean()),
                "ci95_low": float(lo),
                "ci95_high": float(hi),
                "p_raw": float(test.pvalue),
            })

# Holm correction across all 30 comparisons.
m = len(results)
order = sorted(range(m), key=lambda i: results[i]["p_raw"])
running_max = 0.0

for rank, idx in enumerate(order):
    adjusted = min(
        1.0,
        (m - rank) * results[idx]["p_raw"],
    )
    running_max = max(running_max, adjusted)
    results[idx]["p_holm"] = running_max

fields = [
    "dataset", "metric", "comparison", "n",
    "mean_diff", "ci95_low", "ci95_high",
    "p_raw", "p_holm",
]

with OUT.open("w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=fields)
    writer.writeheader()

    for r in results:
        writer.writerow({
            k: (
                f"{r[k]:.8g}"
                if isinstance(r[k], float)
                else r[k]
            )
            for k in fields
        })

print("Comparisons:", len(results))
print("Holm correction family: all 30 comparisons")
print("Saved:", OUT)

for r in results:
    print(
        f"{r['dataset']:10s} {r['metric']:6s} "
        f"{r['comparison']:28s} "
        f"diff={r['mean_diff']:+.6f} "
        f"CI95=[{r['ci95_low']:+.6f}, "
        f"{r['ci95_high']:+.6f}] "
        f"p={r['p_raw']:.4g} "
        f"Holm={r['p_holm']:.4g}"
    )
