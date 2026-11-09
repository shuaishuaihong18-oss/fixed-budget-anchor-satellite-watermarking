from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

ROOT = Path("outputs/revision_202610")
INPUT = ROOT / "ablation_clean500/rho_vote_ablation.csv"
OUT = ROOT / "ablation_clean500"

ATTACKS = [
    "erase40",
    "randomcrop50",
    "cornercrop50",
    "splicing20",
]
ALL_ATTACKS = ["none"] + ATTACKS

RHOS = ["0.2", "0.4", "0.6", "0.8"]
MODES = ["orig", "uniform", "mask", "proposed"]
METRICS = ["vote_bitacc", "vote_recovery31"]
DESC_METRICS = [
    "actual_area", "psnr", "ssim",
    "vote_bitacc", "vote_recovery31",
]
N_BOOT = 5000
BASE_SEED = 20261028

if not INPUT.exists():
    raise SystemExit(f"Missing input: {INPUT}")

df = pd.read_csv(INPUT, dtype={"image": str, "message": str})
df["rho"] = pd.to_numeric(df["rho"], errors="raise")
df["rho_level"] = df["rho"].map(lambda x: f"{x:.1f}")
df["attack"] = df["attack"].str.lower()
df["vote_mode"] = df["vote_mode"].str.lower()

print("========== INPUT AUDIT ==========")
print("Rows:", len(df))
print("Unique images:", df["image"].nunique())

if len(df) != 17500 or df["image"].nunique() != 500:
    raise SystemExit("FAIL: unexpected dataset size")

keys = ["image", "rho_level", "vote_mode", "attack"]
if df.duplicated(keys).any():
    raise SystemExit("FAIL: duplicate image/configuration/attack rows")

if not df["attack"].isin(ALL_ATTACKS).all():
    raise SystemExit("FAIL: unexpected attack name")

if not df["vote_mode"].isin(MODES).all():
    raise SystemExit("FAIL: unexpected voting mode")

if df.groupby("image")["message"].nunique().max() != 1:
    raise SystemExit("FAIL: inconsistent messages across configurations")

if not df["message"].str.fullmatch(r"[01]{32}").all():
    raise SystemExit("FAIL: invalid 32-bit message")

# Every rho has one proposed-voting result for each image and attack.
rho_data = df[df["vote_mode"] == "proposed"].copy()

for rho in RHOS:
    count = (rho_data["rho_level"] == rho).sum()
    if count != 2500:
        raise SystemExit(f"FAIL: rho={rho}, expected 2500 rows, got {count}")

# At rho=0.4, four voting modes are present.
vote_data = df[df["rho_level"] == "0.4"].copy()

for mode in MODES:
    count = (vote_data["vote_mode"] == mode).sum()
    if count != 2500:
        raise SystemExit(
            f"FAIL: mode={mode}, expected 2500 rows, got {count}"
        )

print("Unique image/configuration/attack keys: verified")
print("Shared 32-bit messages: verified")
print("Rho configurations: verified")
print("Voting configurations: verified")

def holm_adjust(p_values):
    """Holm step-down adjusted p-values."""
    p = np.asarray(p_values, dtype=float)
    m = len(p)

    order = np.argsort(p)
    sorted_p = p[order]

    adjusted_sorted = np.maximum.accumulate(
        (m - np.arange(m)) * sorted_p
    )
    adjusted_sorted = np.minimum(adjusted_sorted, 1.0)

    adjusted = np.empty(m, dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted

def paired_test(diffs, seed):
    diffs = np.asarray(diffs, dtype=float)
    diffs = diffs[np.isfinite(diffs)]

    if len(diffs) != 500:
        raise ValueError(
            f"Expected 500 paired images; got {len(diffs)}"
        )

    mean_diff = float(diffs.mean())

    if np.allclose(diffs, 0.0):
        p_raw = 1.0
    else:
        p_raw = float(
            wilcoxon(
                diffs,
                zero_method="wilcox",
                alternative="two-sided",
            ).pvalue
        )

    rng = np.random.default_rng(seed)
    boot = np.empty(N_BOOT, dtype=float)

    # Chunk resampling to control memory use.
    for start in range(0, N_BOOT, 500):
        stop = min(start + 500, N_BOOT)
        samples = rng.choice(
            diffs,
            size=(stop - start, len(diffs)),
            replace=True,
        )
        boot[start:stop] = samples.mean(axis=1)

    ci_low, ci_high = np.quantile(boot, [0.025, 0.975])

    return {
        "n_images": len(diffs),
        "mean_diff": mean_diff,
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_raw": p_raw,
    }

def build_descriptive(data, analysis, level_col, levels):
    rows = []

    for scope in ALL_ATTACKS + ["overall_four_attacks"]:
        if scope == "overall_four_attacks":
            sub = data[data["attack"].isin(ATTACKS)].copy()

            # Aggregate attacks within each image before averaging images.
            per_image = (
                sub.groupby(["image", level_col], as_index=False)[
                    DESC_METRICS
                ].mean()
            )
        else:
            sub = data[data["attack"] == scope].copy()
            per_image = sub[
                ["image", level_col] + DESC_METRICS
            ].copy()

        for level in levels:
            part = per_image[per_image[level_col] == level]

            if part.empty:
                raise ValueError(
                    f"Missing data: {analysis}, {scope}, {level}"
                )

            row = {
                "analysis": analysis,
                "scope": scope,
                "configuration": level,
                "n_images": part["image"].nunique(),
            }

            for metric in DESC_METRICS:
                row[f"mean_{metric}"] = float(part[metric].mean())

            rows.append(row)

    return pd.DataFrame(rows)

def build_tests(data, analysis, level_col, comparisons):
    rows = []
    seed_counter = 0

    for metric in METRICS:
        for scope in ATTACKS + ["overall_four_attacks"]:

            if scope == "overall_four_attacks":
                sub = data[data["attack"].isin(ATTACKS)]
                per_image = (
                    sub.groupby(["image", level_col], as_index=False)[metric]
                    .mean()
                )
            else:
                per_image = data.loc[
                    data["attack"] == scope,
                    ["image", level_col, metric],
                ].copy()

            pivot = per_image.pivot(
                index="image",
                columns=level_col,
                values=metric,
            )

            for target, reference, label in comparisons:
                diffs = (
                    pivot[target] - pivot[reference]
                ).dropna().to_numpy(dtype=float)

                seed_counter += 1
                result = paired_test(
                    diffs,
                    BASE_SEED + seed_counter,
                )

                rows.append({
                    "analysis": analysis,
                    "scope": scope,
                    "metric": metric,
                    "comparison": label,
                    "direction": f"{target} minus {reference}",
                    **result,
                })

    result_df = pd.DataFrame(rows)

    # Separate Holm families for each metric within each ablation.
    result_df["p_holm"] = np.nan

    for metric in METRICS:
        mask = result_df["metric"] == metric
        result_df.loc[mask, "p_holm"] = holm_adjust(
            result_df.loc[mask, "p_raw"].to_numpy()
        )

    result_df["significant_holm_0.05"] = (
        result_df["p_holm"] < 0.05
    )

    return result_df

# A. Anchor ratio: proposed voting fixed; rho=0.4 is reference.
rho_comparisons = [
    (rho, "0.4", f"rho_{rho}_vs_0.4")
    for rho in ["0.2", "0.6", "0.8"]
]

rho_desc = build_descriptive(
    rho_data,
    "anchor_ratio",
    "rho_level",
    RHOS,
)

rho_tests = build_tests(
    rho_data,
    "anchor_ratio",
    "rho_level",
    rho_comparisons,
)

# B. Voting: rho=0.4 fixed; proposed is the reference.
vote_comparisons = [
    ("proposed", mode, f"proposed_vs_{mode}")
    for mode in ["orig", "uniform", "mask"]
]

vote_desc = build_descriptive(
    vote_data,
    "voting",
    "vote_mode",
    MODES,
)

vote_tests = build_tests(
    vote_data,
    "voting",
    "vote_mode",
    vote_comparisons,
)

outputs = {
    "rho_descriptive.csv": rho_desc,
    "rho_paired_tests_holm.csv": rho_tests,
    "vote_descriptive.csv": vote_desc,
    "vote_paired_tests_holm.csv": vote_tests,
}

for filename, table in outputs.items():
    path = OUT / filename
    table.to_csv(path, index=False)
    print("Saved:", path, "rows:", len(table))

print("\n========== ANCHOR RATIO: FOUR-ATTACK MEANS ==========")
print(
    rho_desc[rho_desc["scope"] == "overall_four_attacks"][
        [
            "configuration", "n_images",
            "mean_actual_area", "mean_vote_bitacc",
            "mean_vote_recovery31",
        ]
    ].to_string(index=False)
)

print("\n========== VOTING: FOUR-ATTACK MEANS ==========")
print(
    vote_desc[vote_desc["scope"] == "overall_four_attacks"][
        [
            "configuration", "n_images",
            "mean_actual_area", "mean_vote_bitacc",
            "mean_vote_recovery31",
        ]
    ].to_string(index=False)
)

print("\n========== ANCHOR RATIO: OVERALL TESTS ==========")
print(
    rho_tests[rho_tests["scope"] == "overall_four_attacks"][
        [
            "metric", "comparison", "mean_diff",
            "ci95_low", "ci95_high", "p_raw", "p_holm",
        ]
    ].to_string(index=False)
)

print("\n========== VOTING: OVERALL TESTS ==========")
print(
    vote_tests[vote_tests["scope"] == "overall_four_attacks"][
        [
            "metric", "comparison", "mean_diff",
            "ci95_low", "ci95_high", "p_raw", "p_holm",
        ]
    ].to_string(index=False)
)

print("\nSTATISTICAL ANALYSIS COMPLETE")
