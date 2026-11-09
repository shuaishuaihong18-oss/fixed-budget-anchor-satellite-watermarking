from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path.cwd()
OUT = ROOT / "outputs/revision_202610/iou_sensitivity_full_20261028"
INPUT = OUT / "iou_sensitivity_results.csv"
MAIN = ROOT / "outputs/revision_202610/raw/coco500_clean_allattacks/results.csv"

THRESHOLDS = [0.00, 0.01, 0.03, 0.05, 0.10]
CANDIDATES = [0.00, 0.01, 0.05, 0.10]
BASELINE = 0.03
ATTACKS = ["none", "erase40", "randomcrop50", "cornercrop50", "splicing20"]
ROBUST_ATTACKS = ["erase40", "randomcrop50", "cornercrop50", "splicing20"]

ROBUST_METRICS = [
    "orig_bitacc", "vote_bitacc",
    "orig_recovery31", "vote_recovery31",
]
QUALITY_METRICS = ["actual_area", "psnr", "ssim"]

N_BOOT = 10000
rng = np.random.default_rng(20261028)


def holm_adjust(pvalues):
    """Holm step-down adjusted p-values."""
    pvalues = np.asarray(pvalues, dtype=float)
    n = len(pvalues)
    order = np.argsort(pvalues)
    adjusted = np.empty(n)
    running = 0.0

    for rank, idx in enumerate(order):
        running = max(running, (n - rank) * pvalues[idx])
        adjusted[idx] = min(1.0, running)

    return adjusted


def paired_comparison(frame, metric, candidate, scope, attack=""):
    wide = frame.pivot(
        index="image", columns="max_iou", values=metric
    )

    pair = wide[[BASELINE, candidate]].dropna()
    a = pair[BASELINE].to_numpy(dtype=float)
    b = pair[candidate].to_numpy(dtype=float)
    diff = b - a

    if len(diff) < 2:
        raise ValueError(f"Insufficient pairs: {scope}, {metric}")

    if np.allclose(diff, 0):
        p_raw = 1.0
    else:
        p_raw = float(
            wilcoxon(
                diff,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            ).pvalue
        )

    # Paired image-level percentile bootstrap CI for the mean difference.
    samples = rng.choice(
        diff, size=(N_BOOT, len(diff)), replace=True
    ).mean(axis=1)
    ci_low, ci_high = np.percentile(samples, [2.5, 97.5])

    return {
        "scope": scope,
        "attack": attack,
        "metric": metric,
        "baseline_iou": BASELINE,
        "candidate_iou": candidate,
        "n_images": len(diff),
        "mean_baseline": float(a.mean()),
        "mean_candidate": float(b.mean()),
        "mean_diff_candidate_minus_baseline": float(diff.mean()),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_raw": p_raw,
    }


def add_holm(frame, family_columns):
    frame = frame.copy()
    frame["p_holm"] = np.nan

    for _, indices in frame.groupby(
        family_columns, dropna=False
    ).groups.items():
        indices = list(indices)
        frame.loc[indices, "p_holm"] = holm_adjust(
            frame.loc[indices, "p_raw"].to_numpy()
        )

    return frame


def main():
    if not INPUT.exists():
        raise FileNotFoundError(INPUT)

    df = pd.read_csv(INPUT)
    df["max_iou"] = df["max_iou"].round(2)

    required = {
        "image", "message", "method", "attack", "attack_seed",
        "max_iou", "actual_area", "psnr", "ssim",
        *ROBUST_METRICS,
    }
    missing = required - set(df.columns)
    if missing:
        raise ValueError(f"Missing columns: {sorted(missing)}")

    print("=" * 72)
    print("1. DATA INTEGRITY AUDIT")
    print("=" * 72)

    expected_rows = 500 * len(THRESHOLDS) * len(ATTACKS)
    if len(df) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, got {len(df)}"
        )

    if df["image"].nunique() != 500:
        raise ValueError("Expected 500 unique images")

    if set(df["max_iou"].unique()) != set(THRESHOLDS):
        raise ValueError("Unexpected IoU thresholds")

    if set(df["attack"].unique()) != set(ATTACKS):
        raise ValueError("Unexpected attack categories")

    keys = ["image", "max_iou", "attack"]
    if df.duplicated(keys).any():
        raise ValueError("Duplicate image/threshold/attack rows")

    counts = df.groupby(["max_iou", "attack"]).size()
    if not (counts == 500).all():
        raise ValueError("Incomplete threshold/attack combinations")

    if (df.groupby("image")["message"].nunique() > 1).any():
        raise ValueError("Payload differs across conditions")

    seed_counts = df.groupby(["image", "attack"])[
        "attack_seed"
    ].nunique()
    if (seed_counts > 1).any():
        raise ValueError("Attack seeds differ across IoU conditions")

    area_spread = df.groupby(["image", "max_iou"])[
        "actual_area"
    ].agg(lambda s: s.max() - s.min()).max()

    psnr_spread = df.groupby(["image", "max_iou"])[
        "psnr"
    ].agg(lambda s: s.max() - s.min()).max()

    ssim_spread = df.groupby(["image", "max_iou"])[
        "ssim"
    ].agg(lambda s: s.max() - s.min()).max()

    if max(area_spread, psnr_spread, ssim_spread) > 1e-7:
        raise ValueError(
            "Area/quality metrics unexpectedly vary across attacks"
        )

    print("Rows:", len(df))
    print("Images:", df["image"].nunique())
    print("Unique keys: PASS")
    print("Payload consistency: PASS")
    print("Attack seed consistency: PASS")
    print("Area/quality invariance across attacks: PASS")

    # Independently compare the default IoU condition with the main Hybrid.
    print("\n" + "=" * 72)
    print("2. DEFAULT CONDITION REPRODUCTION AUDIT")
    print("=" * 72)

    if not MAIN.exists():
        raise FileNotFoundError(MAIN)

    ref = pd.read_csv(MAIN)
    ref = ref[ref["method"] == "hybrid"].copy()

    current = df[df["max_iou"] == BASELINE].copy()
    merged = current.merge(
        ref,
        on=["image", "attack"],
        how="inner",
        suffixes=("_new", "_ref"),
        validate="one_to_one",
    )

    if len(merged) != 2500:
        raise ValueError(
            f"Expected 2500 matched reference rows, got {len(merged)}"
        )

    reference_diffs = {}
    for metric in [
        "actual_area", "orig_bitacc", "vote_bitacc",
        "orig_recovery31", "vote_recovery31",
    ]:
        col_new, col_ref = f"{metric}_new", f"{metric}_ref"
        if col_new not in merged or col_ref not in merged:
            print(f"Reference comparison skipped: {metric} unavailable")
            continue

        delta = (
            pd.to_numeric(merged[col_new])
            - pd.to_numeric(merged[col_ref])
        ).abs()
        reference_diffs[metric] = float(delta.max())
        print(
            f"{metric}: max_abs_diff="
            f"{reference_diffs[metric]:.10f}"
        )

    bit_metrics = [
        "orig_bitacc", "vote_bitacc",
        "orig_recovery31", "vote_recovery31",
    ]
    if any(reference_diffs.get(m, float("inf")) > 1e-7
           for m in bit_metrics):
        raise ValueError("Default condition failed reference reproduction")

    if reference_diffs.get("actual_area", float("inf")) > 1e-5:
        raise ValueError("Default area differs from Hybrid reference")

    print("PASS: Default condition reproduces the main Hybrid results.")

    # Descriptive results by threshold and attack.
    descriptive = (
        df.groupby(["max_iou", "attack"])
        .agg(
            n_images=("image", "nunique"),
            mean_actual_area=("actual_area", "mean"),
            mean_psnr=("psnr", "mean"),
            mean_ssim=("ssim", "mean"),
            **{f"mean_{m}": (m, "mean") for m in ROBUST_METRICS},
        )
        .reset_index()
    )
    descriptive.to_csv(
        OUT / "iou_sensitivity_descriptive_by_attack.csv",
        index=False,
    )

    # Average the four attacked outcomes within each image first.
    attacked = df[df["attack"].isin(ROBUST_ATTACKS)]
    image_means = (
        attacked.groupby(["max_iou", "image"])[ROBUST_METRICS]
        .mean()
        .reset_index()
    )

    overall_rows = []
    for threshold, group in image_means.groupby("max_iou"):
        row = {
            "max_iou": threshold,
            "n_images": group["image"].nunique(),
        }
        for metric in ROBUST_METRICS:
            row[f"{metric}_mean"] = group[metric].mean()
            row[f"{metric}_sd"] = group[metric].std()
        overall_rows.append(row)

    overall = pd.DataFrame(overall_rows).sort_values("max_iou")
    overall.to_csv(
        OUT / "iou_sensitivity_overall_four_attack.csv",
        index=False,
    )

    # Robustness tests: four-attack average, paired by image.
    overall_tests = []
    for metric in ROBUST_METRICS:
        for candidate in CANDIDATES:
            overall_tests.append(
                paired_comparison(
                    image_means, metric, candidate,
                    scope="four_attack_overall",
                )
            )

    overall_tests = add_holm(
        pd.DataFrame(overall_tests), ["metric"]
    )
    overall_tests.to_csv(
        OUT / "iou_sensitivity_paired_overall_holm.csv",
        index=False,
    )

    # Per-attack tests, including no-attack as a separate condition.
    attack_tests = []
    for attack in ATTACKS:
        subset = df[df["attack"] == attack]
        for metric in ROBUST_METRICS:
            for candidate in CANDIDATES:
                attack_tests.append(
                    paired_comparison(
                        subset, metric, candidate,
                        scope="per_attack",
                        attack=attack,
                    )
                )

    attack_tests = add_holm(
        pd.DataFrame(attack_tests), ["metric"]
    )
    attack_tests.to_csv(
        OUT / "iou_sensitivity_paired_by_attack_holm.csv",
        index=False,
    )

    # Quality/area are image-level metrics, not five independent
    # copies of the same image. Collapse attack rows before testing.
    quality_image = (
        df.groupby(["max_iou", "image"])[QUALITY_METRICS]
        .mean()
        .reset_index()
    )

    quality_tests = []
    for metric in QUALITY_METRICS:
        for candidate in CANDIDATES:
            quality_tests.append(
                paired_comparison(
                    quality_image, metric, candidate,
                    scope="image_level_quality",
                )
            )

    quality_tests = add_holm(
        pd.DataFrame(quality_tests), ["metric"]
    )
    quality_tests.to_csv(
        OUT / "iou_sensitivity_paired_quality_holm.csv",
        index=False,
    )

    integrity = {
        "n_rows": int(len(df)),
        "n_images": int(df["image"].nunique()),
        "thresholds": THRESHOLDS,
        "attacks": ATTACKS,
        "expected_rows": expected_rows,
        "attack_seed_consistency": True,
        "default_reference_max_abs_differences": reference_diffs,
        "max_area_spread_across_attacks": float(area_spread),
        "max_psnr_spread_across_attacks": float(psnr_spread),
        "max_ssim_spread_across_attacks": float(ssim_spread),
        "bootstrap_replicates": N_BOOT,
        "bootstrap_ci": "paired image-level percentile 95% CI",
        "test": "two-sided Wilcoxon signed-rank",
        "holm_families": {
            "overall": "four threshold comparisons within each metric",
            "by_attack": (
                "20 comparisons per metric across five attacks "
                "and four candidate thresholds"
            ),
            "quality": "four threshold comparisons within each metric",
        },
        "difference_direction": (
            "candidate IoU minus baseline IoU=0.03; "
            "positive values favor the candidate"
        ),
    }

    with (OUT / "iou_sensitivity_integrity.json").open("w") as f:
        json.dump(integrity, f, indent=2)

    print("\n" + "=" * 72)
    print("3. FOUR-ATTACK DESCRIPTIVE MEANS")
    print("=" * 72)
    print(overall.to_string(index=False))

    print("\n" + "=" * 72)
    print("4. PAIRED OVERALL TESTS")
    print("=" * 72)
    print(overall_tests.to_string(index=False))

    print("\n" + "=" * 72)
    print("5. PAIRED QUALITY/AREA TESTS")
    print("=" * 72)
    print(quality_tests.to_string(index=False))

    print("\n" + "=" * 72)
    print("ANALYSIS COMPLETE")
    print("Output directory:", OUT)
    print("Per-attack tests:", OUT / "iou_sensitivity_paired_by_attack_holm.csv")
    print("Integrity record:", OUT / "iou_sensitivity_integrity.json")


if __name__ == "__main__":
    main()
