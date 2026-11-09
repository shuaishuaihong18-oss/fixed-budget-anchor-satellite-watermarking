from pathlib import Path
import json

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon

D = Path(__file__).resolve().parent
SRC = D / "detection_param_sensitivity_results.csv"

BASE = "grid8_score008"
CONFIGS = [
    "grid4_score008",
    "grid8_score008",
    "grid16_score008",
    "grid8_score004",
    "grid8_score012",
]
CANDIDATES = [c for c in CONFIGS if c != BASE]
ATTACKS = [
    "none",
    "erase40",
    "randomcrop50",
    "cornercrop50",
    "splicing20",
]
ROBUST_ATTACKS = ATTACKS[1:]
METRICS = ["vote_bitacc", "vote_recovery31"]

RNG_SEED = 20261029
N_BOOT = 10000


def holm(pvalues):
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p, kind="mergesort")
    m = len(p)

    adj_sorted = np.minimum(
        np.maximum.accumulate(
            (m - np.arange(m)) * p[order]
        ),
        1.0,
    )
    adj = np.empty(m, dtype=float)
    adj[order] = adj_sorted
    return adj


def paired_test(a, b, seed):
    a = np.asarray(a, dtype=float)
    b = np.asarray(b, dtype=float)
    d = a - b

    if len(d) != 500 or not np.isfinite(d).all():
        raise ValueError("Invalid paired comparison")

    rng = np.random.default_rng(seed)
    idx = rng.integers(
        0, len(d), size=(N_BOOT, len(d))
    )
    boot_means = d[idx].mean(axis=1)
    ci_low, ci_high = np.quantile(
        boot_means, [0.025, 0.975]
    )

    if np.all(d == 0):
        p = 1.0
    else:
        p = float(
            wilcoxon(
                d,
                alternative="two-sided",
                zero_method="wilcox",
                method="auto",
            ).pvalue
        )

    return {
        "mean_diff": float(d.mean()),
        "ci95_low": float(ci_low),
        "ci95_high": float(ci_high),
        "p_raw": p,
        "n_images": len(d),
        "n_improved": int((d > 0).sum()),
        "n_worsened": int((d < 0).sum()),
        "n_equal": int((d == 0).sum()),
    }


def main():
    print("========== INPUT AUDIT ==========")
    df = pd.read_csv(
        SRC, dtype={"image": str, "message": str}
    )

    keys = ["image_index", "attack", "config"]

    assert len(df) == 12500
    assert not df.duplicated(keys).any()
    assert df["image_index"].nunique() == 500
    assert set(df["config"]) == set(CONFIGS)
    assert set(df["attack"]) == set(ATTACKS)

    counts = df.groupby(keys).size()
    assert len(counts) == 12500
    assert (counts == 1).all()

    assert df.groupby("image_index")["image"].nunique().eq(1).all()
    assert df.groupby("image_index")["message"].nunique().eq(1).all()
    assert df.groupby("image_index")["attack_seed"].nunique().eq(1).all()
    assert df["message"].str.fullmatch("[01]{32}").all()
    assert (df["attack_seed"] == df["image_index"] + 3).all()

    assert df["vote_bitacc"].between(0, 1).all()
    assert df["vote_recovery31"].isin([0, 1]).all()
    assert df["region_count"].ge(0).all()

    expected_settings = {
        "grid4_score008": (4, 0.08),
        "grid8_score008": (8, 0.08),
        "grid16_score008": (16, 0.08),
        "grid8_score004": (8, 0.04),
        "grid8_score012": (8, 0.12),
    }
    for config, (grid, score) in expected_settings.items():
        sub = df[df["config"] == config]
        assert sub["grid"].eq(grid).all()
        assert np.allclose(sub["min_cell_score"], score)
        assert np.allclose(sub["mask_threshold"], 0.5)
        assert sub["vote_mode"].eq("proposed").all()

    print("Rows: 12500 PASS")
    print("Images: 500 PASS")
    print("Image/message/seed mapping: PASS")
    print("Configuration and attack coverage: PASS")
    print("Metrics and settings: PASS")

    # Descriptive results by attack and configuration.
    descriptive = (
        df.groupby(["attack", "config"], as_index=False)
        .agg(
            n_images=("image", "nunique"),
            mean_vote_bitacc=("vote_bitacc", "mean"),
            mean_vote_recovery31=("vote_recovery31", "mean"),
            mean_region_count=("region_count", "mean"),
            zero_region_count=("region_count", lambda x: int((x == 0).sum())),
        )
    )
    descriptive.to_csv(
        D / "detection_descriptive_by_attack.csv",
        index=False,
    )

    # Average four attacks within each paired image.
    robust = df[df["attack"].isin(ROBUST_ATTACKS)].copy()

    image_avg = (
        robust.groupby(["image_index", "config"], as_index=False)
        [METRICS].mean()
    )
    assert len(image_avg) == 2500

    image_avg.to_csv(
        D / "detection_four_attack_per_image.csv",
        index=False,
    )

    overall = (
        image_avg.groupby("config", as_index=False)
        [METRICS].mean()
    )
    overall.to_csv(
        D / "detection_overall_four_attack.csv",
        index=False,
    )

    def make_comparisons(frame, group_key=None):
        rows = []
        families = (
            [("overall_four_attacks", frame)]
            if group_key is None
            else [
                (attack, frame[frame[group_key] == attack])
                for attack in ATTACKS
            ]
        )

        seed_index = 0

        for attack_name, part in families:
            for metric in METRICS:
                wide = part.pivot(
                    index="image_index",
                    columns="config",
                    values=metric,
                )
                assert wide.shape == (500, 5)
                assert not wide.isna().any().any()

                for candidate in CANDIDATES:
                    result = paired_test(
                        wide[candidate].to_numpy(),
                        wide[BASE].to_numpy(),
                        RNG_SEED + seed_index,
                    )
                    seed_index += 1

                    rows.append({
                        "attack": attack_name,
                        "metric": metric,
                        "baseline": BASE,
                        "candidate": candidate,
                        **result,
                    })

        return pd.DataFrame(rows)

    print("\n========== PAIRED STATISTICS ==========")

    overall_tests = make_comparisons(image_avg)
    assert len(overall_tests) == 8

    # One Holm family of 8 overall comparisons.
    overall_tests["p_holm_global8"] = holm(
        overall_tests["p_raw"].to_numpy()
    )

    overall_tests.to_csv(
        D / "detection_paired_overall_holm.csv",
        index=False,
    )

    by_attack_tests = make_comparisons(
        df, group_key="attack"
    )
    assert len(by_attack_tests) == 40

    # One Holm family of 40 per-attack comparisons,
    # including clean/no-attack results.
    by_attack_tests["p_holm_global40"] = holm(
        by_attack_tests["p_raw"].to_numpy()
    )

    by_attack_tests.to_csv(
        D / "detection_paired_by_attack_holm.csv",
        index=False,
    )

    print("\n========== FOUR-ATTACK MEANS ==========")
    print(overall.round(6).to_string(index=False))

    print("\n========== OVERALL PAIRED COMPARISONS ==========")
    print(
        overall_tests[
            [
                "metric",
                "candidate",
                "mean_diff",
                "ci95_low",
                "ci95_high",
                "p_raw",
                "p_holm_global8",
            ]
        ].round(7).to_string(index=False)
    )

    print("\n========== STATISTICAL SUMMARY ==========")
    sig_overall = int(
        (overall_tests["p_holm_global8"] < 0.05).sum()
    )
    sig_attack = int(
        (by_attack_tests["p_holm_global40"] < 0.05).sum()
    )

    print(
        f"Overall: {sig_overall}/8 significant "
        "after global Holm"
    )
    print(
        f"By attack: {sig_attack}/40 significant "
        "after global Holm"
    )

    if sig_attack:
        print("\nSignificant by-attack comparisons:")
        print(
            by_attack_tests.loc[
                by_attack_tests["p_holm_global40"] < 0.05,
                [
                    "attack",
                    "metric",
                    "candidate",
                    "mean_diff",
                    "p_holm_global40",
                ],
            ].round(7).to_string(index=False)
        )

    print("\n========== DETECTOR REGION COUNTS ==========")
    print(
        descriptive[
            [
                "attack",
                "config",
                "mean_region_count",
                "zero_region_count",
            ]
        ].round(3).to_string(index=False)
    )

    integrity = {
        "rows": len(df),
        "images": 500,
        "configurations": CONFIGS,
        "attacks": ATTACKS,
        "overall_comparisons": len(overall_tests),
        "by_attack_comparisons": len(by_attack_tests),
        "overall_significant_global_holm": sig_overall,
        "by_attack_significant_global_holm": sig_attack,
        "bootstrap_samples": N_BOOT,
        "bootstrap_ci_adjusted_for_multiplicity": False,
        "test": "two-sided paired Wilcoxon",
        "pairing_unit": "image",
    }

    with (D / "detection_analysis_integrity.json").open(
        "w"
    ) as f:
        json.dump(integrity, f, indent=2)

    print("\nANALYSIS COMPLETE")
    print("No GPU required; no original CSV modified.")


if __name__ == "__main__":
    main()
