import csv
import importlib.util
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from notebooks.inference_utils import load_model_from_checkpoint
from watermark_anything.data.transforms import default_transform
from watermark_anything.redundant_region import (
    _grid_candidate_masks,
    detect_confidence_voting,
    bit_accuracy,
    recovery_31_32,
)

SOURCE = (
    REPO / "outputs/revision_202610/"
    "iou_sensitivity_full_20261028/run_iou_sensitivity.py"
)

spec = importlib.util.spec_from_file_location(
    "fallback_source_reuse", SOURCE
)
sens = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = sens
spec.loader.exec_module(sens)

base = sens.base
D = Path(__file__).resolve().parent

OUT = D / "fallback_threshold_results.csv"
REFERENCE = D / "detection_param_sensitivity_results.csv"

THRESHOLDS = (0.3, 0.5, 0.7)
GRID = 8
MIN_SCORE = 0.08
IOU = 0.03
SEED = sens.SEED
ATTACKS = list(sens.ATTACKS)

FIELDS = [
    "image_index", "image", "attack", "attack_seed",
    "mask_threshold", "grid_candidate_count",
    "grid_fallback_triggered", "threshold_support_pixels",
    "region_count", "vote_bitacc", "vote_recovery31",
]


class CachedDetector:
    """Use identical WAM predictions for all threshold conditions."""
    def __init__(self, preds):
        self.preds = preds

    def detect(self, image):
        return {"preds": self.preds}


def as_float(x):
    return float(
        x.detach().cpu().item()
        if torch.is_tensor(x) else x
    )


def main():
    if OUT.exists():
        raise FileExistsError(
            f"Result already exists: {OUT}. Do not overwrite."
        )

    reference = pd.read_csv(REFERENCE)
    reference = reference[
        reference["config"] == "grid8_score008"
    ].copy()

    assert len(reference) == 2500
    assert not reference.duplicated(["image", "attack"]).any()

    ref_lookup = {
        (r.image, r.attack): r
        for r in reference.itertuples(index=False)
    }

    messages, _ = base.load_main_reference()
    paths = list(
        sens.iter_images(
            REPO / "data/eval/coco500_clean", 500
        )
    )

    assert len(paths) == 500
    assert set(messages) == {p.name for p in paths}

    geometry = sens.load_geometry(paths)

    torch.manual_seed(SEED)
    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("Device:", device)
    print("Loading frozen WAM...")

    wam = load_model_from_checkpoint(
        str(REPO / "checkpoints/params.json"),
        str(REPO / "checkpoints/wam_mit.pth"),
    ).to(device).eval()

    n_written = 0
    max_baseline_diff = 0.0
    baseline_region_mismatch = 0

    print("Attacks:", ATTACKS)
    print("Fallback thresholds:", THRESHOLDS)
    print("Expected rows:", 500 * len(ATTACKS) * len(THRESHOLDS))

    with OUT.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        with torch.inference_mode():
            for idx, path in enumerate(paths):
                with Image.open(path) as im:
                    img = default_transform(
                        im.convert("RGB")
                    ).unsqueeze(0).to(device)

                H, W = img.shape[-2:]
                g = geometry[(idx, IOU)]

                assert (H, W) == (
                    int(g["height"]), int(g["width"])
                )

                import json
                boxes = json.loads(g["boxes"])

                mask = torch.zeros(
                    (1, 1, H, W),
                    device=device,
                    dtype=img.dtype,
                )

                for raw_box in boxes:
                    box = base.parse_box(raw_box)
                    mask[
                        :, :, box.y1:box.y2, box.x1:box.x2
                    ] = 1.0

                msg = torch.tensor(
                    [[int(x) for x in messages[path.name]]],
                    dtype=torch.float32,
                    device=device,
                )

                full_w = wam.embed(img, msg)["imgs_w"]
                watermarked = (
                    full_w * mask + img * (1.0 - mask)
                )

                attack_seed = SEED + idx

                for attack in ATTACKS:
                    attacked = sens.apply_attack(
                        watermarked,
                        img,
                        attack,
                        seed=attack_seed,
                    )

                    # Only one expensive detector forward per attack.
                    preds = wam.detect(attacked)["preds"]
                    cached = CachedDetector(preds)

                    mask_prob = torch.sigmoid(
                        preds[:, 0:1]
                    )

                    grid_candidates = _grid_candidate_masks(
                        mask_prob,
                        grid=GRID,
                        min_score=MIN_SCORE,
                    )

                    n_candidates = len(grid_candidates)
                    fallback_triggered = int(
                        n_candidates == 0
                    )

                    for threshold in THRESHOLDS:
                        det = detect_confidence_voting(
                            cached,
                            attacked,
                            grid=GRID,
                            min_cell_score=MIN_SCORE,
                            mask_threshold=threshold,
                            vote_mode="proposed",
                        )

                        pred_msg = det["pred_msg"].to(device)

                        ba = as_float(
                            bit_accuracy(pred_msg, msg)
                        )
                        rec = as_float(
                            recovery_31_32(pred_msg, msg)
                        )

                        regions = int(
                            as_float(det["region_count"])
                        )

                        support_pixels = int(
                            (mask_prob > threshold)
                            .sum().item()
                        )

                        writer.writerow({
                            "image_index": idx,
                            "image": path.name,
                            "attack": attack,
                            "attack_seed": attack_seed,
                            "mask_threshold": threshold,
                            "grid_candidate_count": n_candidates,
                            "grid_fallback_triggered": fallback_triggered,
                            "threshold_support_pixels": support_pixels,
                            "region_count": regions,
                            "vote_bitacc": ba,
                            "vote_recovery31": rec,
                        })
                        n_written += 1

                        if threshold == 0.5:
                            ref = ref_lookup[
                                (path.name, attack)
                            ]
                            diff = max(
                                abs(
                                    ba -
                                    float(ref.vote_bitacc)
                                ),
                                abs(
                                    rec -
                                    float(ref.vote_recovery31)
                                ),
                            )
                            max_baseline_diff = max(
                                max_baseline_diff, diff
                            )
                            if regions != int(ref.region_count):
                                baseline_region_mismatch += 1

                f.flush()

                if (idx + 1) % 25 == 0:
                    print(
                        f"Processed {idx+1}/500; "
                        f"rows={n_written}",
                        flush=True,
                    )

    assert n_written == 7500
    assert max_baseline_diff <= 1e-6
    assert baseline_region_mismatch == 0

    print("\n========== DEFAULT REPRODUCTION ==========")
    print("Rows:", n_written)
    print("Max baseline difference:", max_baseline_diff)
    print("Region-count mismatches:", baseline_region_mismatch)
    print("PASS: Original default results reproduced.")

    df = pd.read_csv(OUT)
    assert not df.duplicated(
        ["image", "attack", "mask_threshold"]
    ).any()

    summary = (
        df.groupby(
            ["attack", "mask_threshold"],
            as_index=False
        )
        .agg(
            n_images=("image", "nunique"),
            grid_fallback_cases=(
                "grid_fallback_triggered", "sum"
            ),
            zero_region_cases=(
                "region_count", lambda x: int((x == 0).sum())
            ),
            mean_bitacc=("vote_bitacc", "mean"),
            mean_recovery31=("vote_recovery31", "mean"),
        )
    )

    summary.to_csv(
        D / "fallback_threshold_summary.csv",
        index=False,
    )

    print("\n========== FALLBACK SUMMARY ==========")
    print(summary.round(6).to_string(index=False))

    # Verify whether changing only fallback threshold affects
    # any actual message prediction outcome.
    wide_ba = df.pivot(
        index=["image", "attack"],
        columns="mask_threshold",
        values="vote_bitacc"
    )
    wide_rec = df.pivot(
        index=["image", "attack"],
        columns="mask_threshold",
        values="vote_recovery31"
    )

    print("\n========== THRESHOLD EFFECT ==========")
    for t in (0.3, 0.7):
        changed_ba = int(
            (np.abs(wide_ba[t] - wide_ba[0.5]) > 1e-9).sum()
        )
        changed_rec = int(
            (np.abs(wide_rec[t] - wide_rec[0.5]) > 1e-9).sum()
        )

        print(
            f"Threshold {t} vs 0.5: "
            f"changed BitAcc cases={changed_ba}, "
            f"changed Recovery31 cases={changed_rec}"
        )

    print("\nEXPERIMENT COMPLETE")
    print("Raw:", OUT)
    print("Summary:", D / "fallback_threshold_summary.csv")


if __name__ == "__main__":
    main()
