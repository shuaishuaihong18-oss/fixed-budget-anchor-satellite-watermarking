import csv
import math
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import wilcoxon

ROOT = Path("outputs/revision_202610")
SUMMARY = ROOT / "summary"
SUMMARY.mkdir(parents=True, exist_ok=True)

DATASETS = {
    "COCO500": (
        ROOT / "raw/coco500_clean_allattacks/results.csv",
        500,
    ),
    "DIV2K100": (
        ROOT / "raw/paired100_allattacks/results.csv",
        100,
    ),
}

METHODS = [
    "full_wam_reference",
    "single",
    "single_random",
    "random",
    "texture",
    "wam_random_mask",
    "hybrid",
]

BASELINES = [
    "single",
    "single_random",
    "random",
    "texture",
    "wam_random_mask",
]

ATTACKS = [
    "none",
    "erase40",
    "randomcrop50",
    "cornercrop50",
    "splicing20",
]

ROBUST_ATTACKS = [
    "erase40",
    "randomcrop50",
    "cornercrop50",
    "splicing20",
]

METRICS = [
    "vote_bitacc",
    "vote_recovery31",
]

SEED = 20261027
N_BOOT = 10000
rng = np.random.default_rng(SEED)


def write_csv(path, rows):
    if not rows:
        print("WARNING: no rows to write:", path)
        return
    with path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


def holm_adjust(pvalues):
    """Holm step-down family-wise error correction."""
    pvalues = np.asarray(pvalues, dtype=float)
    m = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(m, dtype=float)
    running = 0.0

    for rank, idx in enumerate(order):
        value = min(1.0, (m - rank) * pvalues[idx])
        running = max(running, value)
        adjusted[idx] = running

    return adjusted.tolist()


def paired_stats(differences):
    """Image-level paired differences: Hybrid minus baseline."""
    d = np.asarray(differences, dtype=float)
    d = d[np.isfinite(d)]
    n = len(d)

    if n < 2:
        raise ValueError(f"Insufficient paired observations: n={n}")

    mean_diff = float(d.mean())

    # Resample image-level paired differences, not individual bits
    # or attack rows. The same image remains the statistical unit.
    boot_means = np.empty(N_BOOT, dtype=float)
    for start in range(0, N_BOOT, 500):
        end = min(start + 500, N_BOOT)
        indices = rng.integers(0, n, size=(end - start, n))
        boot_means[start:end] = d[indices].mean(axis=1)

    ci_low, ci_high = np.quantile(
        boot_means, [0.025, 0.975]
    )

    if np.allclose(d, 0):
        pvalue = 1.0
    else:
        try:
            pvalue = float(
                wilcoxon(
                    d,
                    zero_method="wilcox",
                    alternative="two-sided",
                    method="auto",
                ).pvalue
            )
        except ValueError:
            pvalue = 1.0

    return {
        "n_images": n,
        "mean_diff": mean_diff,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_raw": pvalue,
    }


all_descriptive = []
all_area = []
all_overall = []
all_by_attack = []

for dataset, (csv_path, expected_images) in DATASETS.items():
    print(f"\n========== AUDIT: {dataset} ==========")

    if not csv_path.exists():
        raise FileNotFoundError(csv_path)

    with csv_path.open(newline="") as f:
        rows = list(csv.DictReader(f))

    print("Rows:", len(rows))
    print("Unique images:", len(set(r["image"] for r in rows)))

    image_names = sorted(set(r["image"] for r in rows))
    methods = sorted(set(r["method"] for r in rows))
    attacks = sorted(set(r["attack"] for r in rows))

    expected_rows = expected_images * len(METHODS) * len(ATTACKS)

    if len(rows) != expected_rows:
        raise ValueError(
            f"{dataset}: expected {expected_rows} rows, got {len(rows)}"
        )

    if len(image_names) != expected_images:
        raise ValueError(
            f"{dataset}: expected {expected_images} images, "
            f"got {len(image_names)}"
        )

    if set(methods) != set(METHODS):
        raise ValueError(
            f"{dataset}: unexpected methods: {methods}"
        )

    if set(attacks) != set(ATTACKS):
        raise ValueError(
            f"{dataset}: unexpected attacks: {attacks}"
        )

    records = {}
    counts = Counter()
    messages = defaultdict(set)
    area_values = {}

    for r in rows:
        key = (r["image"], r["method"], r["attack"])
        counts[key] += 1

        if key in records:
            raise ValueError(f"Duplicate result row: {key}")

        record = dict(r)
        for metric in [
            "actual_area",
            "orig_bitacc",
            "vote_bitacc",
            "orig_recovery31",
            "vote_recovery31",
        ]:
            record[metric] = float(r[metric])

        records[key] = record
        messages[r["image"]].add(r["message"])

        area_key = (r["image"], r["method"])
        area = float(r["actual_area"])

        if area_key in area_values and not math.isclose(
            area_values[area_key], area, abs_tol=1e-6
        ):
            raise ValueError(
                f"Area varies across attacks: {area_key}"
            )

        area_values[area_key] = area

    if any(n != 1 for n in counts.values()):
        raise ValueError(f"{dataset}: duplicate combination detected")

    if len(counts) != expected_rows:
        raise ValueError(f"{dataset}: incomplete image-method-attack grid")

    if any(len(v) != 1 for v in messages.values()):
        raise ValueError(
            f"{dataset}: message differs across methods/attacks"
        )

    if any(len(message) != 32 for per_image in messages.values() for message in per_image):
        raise ValueError(f"{dataset}: a message is not 32 bits")

    print("Methods:", methods)
    print("Attacks:", attacks)
    print("Image-method-attack grid: complete")
    print("Shared 32-bit message per image: verified")
    print("Area consistent across attacks: verified")

    # Area summary: count each image-method combination only once.
    for method in METHODS:
        vals = [
            area_values[(img, method)]
            for img in image_names
        ]
        all_area.append({
            "dataset": dataset,
            "method": method,
            "n_images": len(vals),
            "mean_actual_area": float(np.mean(vals)),
            "min_actual_area": float(np.min(vals)),
            "max_actual_area": float(np.max(vals)),
        })

    # Descriptive summary by method and attack.
    for method in METHODS:
        for attack in ATTACKS:
            subset = [
                records[(img, method, attack)]
                for img in image_names
            ]

            item = {
                "dataset": dataset,
                "method": method,
                "attack": attack,
                "n_images": len(subset),
                "mean_actual_area": float(np.mean([
                    x["actual_area"] for x in subset
                ])),
            }

            for metric in [
                "orig_bitacc",
                "vote_bitacc",
                "orig_recovery31",
                "vote_recovery31",
            ]:
                item["mean_" + metric] = float(np.mean([
                    x[metric] for x in subset
                ]))

            all_descriptive.append(item)

    # Overall robustness:
    # first average the four attack scores within each image,
    # then compare methods using paired image-level differences.
    for metric in METRICS:
        per_method = {}

        for method in METHODS:
            per_method[method] = np.array([
                np.mean([
                    records[(img, method, attack)][metric]
                    for attack in ROBUST_ATTACKS
                ])
                for img in image_names
            ])

        hybrid = per_method["hybrid"]

        for baseline in BASELINES:
            base = per_method[baseline]
            result = paired_stats(hybrid - base)

            all_overall.append({
                "dataset": dataset,
                "metric": metric,
                "comparison": f"hybrid_vs_{baseline}",
                "hybrid_mean": float(hybrid.mean()),
                "baseline_mean": float(base.mean()),
                **result,
            })

    # Attack-specific paired tests.
    for attack in ROBUST_ATTACKS:
        for metric in METRICS:
            hybrid = np.array([
                records[(img, "hybrid", attack)][metric]
                for img in image_names
            ])

            for baseline in BASELINES:
                base = np.array([
                    records[(img, baseline, attack)][metric]
                    for img in image_names
                ])

                result = paired_stats(hybrid - base)

                all_by_attack.append({
                    "dataset": dataset,
                    "attack": attack,
                    "metric": metric,
                    "comparison": f"hybrid_vs_{baseline}",
                    "hybrid_mean": float(hybrid.mean()),
                    "baseline_mean": float(base.mean()),
                    **result,
                })


# Holm families:
# 1) Overall robustness: each dataset x metric, five baselines.
# 2) Attack-specific: each dataset x attack x metric, five baselines.
for rows, group_keys in [
    (all_overall, ["dataset", "metric"]),
    (all_by_attack, ["dataset", "attack", "metric"]),
]:
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[tuple(row[k] for k in group_keys)].append(i)

    for _, indices in groups.items():
        adjusted = holm_adjust([
            rows[i]["p_raw"] for i in indices
        ])
        for i, p_adj in zip(indices, adjusted):
            rows[i]["p_holm"] = p_adj


write_csv(
    SUMMARY / "final_descriptive_by_attack.csv",
    all_descriptive,
)
write_csv(
    SUMMARY / "final_actual_area.csv",
    all_area,
)
write_csv(
    SUMMARY / "final_paired_overall.csv",
    all_overall,
)
write_csv(
    SUMMARY / "final_paired_by_attack.csv",
    all_by_attack,
)

print("\n========== OVERALL ROBUSTNESS ==========")
for r in all_overall:
    print(
        f"{r['dataset']:10s} {r['metric']:18s} "
        f"{r['comparison']:28s} "
        f"Hybrid={r['hybrid_mean']:.4f} "
        f"Baseline={r['baseline_mean']:.4f} "
        f"Diff={r['mean_diff']:+.4f} "
        f"CI95=[{r['ci95_low']:+.4f}, {r['ci95_high']:+.4f}] "
        f"p={r['p_raw']:.4g} Holm={r['p_holm']:.4g}"
    )

print("\n========== OUTPUT FILES ==========")
for name in [
    "final_descriptive_by_attack.csv",
    "final_actual_area.csv",
    "final_paired_overall.csv",
    "final_paired_by_attack.csv",
]:
    print(SUMMARY / name)

print("\nSTATISTICAL ANALYSIS COMPLETE")
