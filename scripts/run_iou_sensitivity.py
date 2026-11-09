import argparse
import csv
import importlib.util
import json
import random
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

import torch
from PIL import Image

from notebooks.inference_utils import load_model_from_checkpoint
from watermark_anything.data.transforms import default_transform
from watermark_anything.redundant_region import (
    detect_confidence_voting,
    bit_accuracy,
    recovery_31_32,
)
from scripts.run_fixed_budget_exp import (
    apply_attack,
    original_wam_decode,
    iter_images,
)

# Reuse the validated functions and constants from the existing experiment.
BASE_PATH = (
    REPO / "outputs/revision_202610/"
    "anchor_position_matched_20261028/"
    "run_anchor_position_matched.py"
)

spec = importlib.util.spec_from_file_location(
    "anchor_position_base", BASE_PATH
)
base = importlib.util.module_from_spec(spec)
sys.modules[spec.name] = base
spec.loader.exec_module(base)

LIMIT = 500
THRESHOLDS = (0.00, 0.01, 0.03, 0.05, 0.10)
ATTACKS = list(base.ATTACKS)
SEED = base.SEED

GEOMETRY_CSV = (
    REPO / "outputs/revision_202610/"
    "iou_sensitivity_preflight_20261028/"
    "iou_geometry_sensitivity_preflight.csv"
)

OUT_DIR = (
    REPO / "outputs/revision_202610/"
    "iou_sensitivity_full_20261028"
)

FIELDS = [
    "image", "message", "method", "anchor_position", "attack",
    "attack_seed", "geometry_image_index", "geometry_seed",
    "max_iou", "max_observed_pair_iou", "geometry_region_count",
    "geometry_actual_area", "geometry_union_area_pixels",
    "configured_budget", "anchor_ratio", "actual_area",
    "anchor_area", "satellite_union_area",
    "anchor_satellite_overlap_area",
    "anchor_center_reference_overlap_area",
    "satellite_center_reference_overlap_area",
    "satellite_pairwise_overlap_area",
    "anchor_satellite_overlap_pixels",
    "satellite_pairwise_overlap_pixels",
    "anchor_box", "center_reference_box", "satellite_boxes",
    "psnr", "ssim", "regions_detected",
    "orig_bitacc", "vote_bitacc",
    "orig_recovery31", "vote_recovery31",
]


def load_geometry(all_paths):
    if not GEOMETRY_CSV.exists():
        raise FileNotFoundError(GEOMETRY_CSV)

    with GEOMETRY_CSV.open(newline="") as f:
        rows = list(csv.DictReader(f))

    expected = LIMIT * len(THRESHOLDS)
    if len(rows) != expected:
        raise ValueError(
            f"Expected {expected} geometry rows, got {len(rows)}"
        )

    geometry = {}

    for r in rows:
        idx = int(r["image_index"])
        threshold = round(float(r["max_iou"]), 2)
        key = (idx, threshold)

        if key in geometry:
            raise ValueError(f"Duplicate geometry key: {key}")

        if not 0 <= idx < LIMIT:
            raise ValueError(f"Invalid image index: {idx}")

        if threshold not in THRESHOLDS:
            raise ValueError(f"Unexpected IoU threshold: {threshold}")

        if r["image"] != all_paths[idx].name:
            raise ValueError(
                f"Image mapping mismatch at index {idx}: "
                f"{r['image']} != {all_paths[idx].name}"
            )

        if int(r["seed"]) != SEED + idx:
            raise ValueError(f"Geometry seed mismatch: {key}")

        H, W = int(r["height"]), int(r["width"])
        boxes = json.loads(r["boxes"])

        if len(boxes) != base.K:
            raise ValueError(
                f"Expected {base.K} boxes at {key}, got {len(boxes)}"
            )

        for b in boxes:
            if not (
                0 <= int(b["x1"]) < int(b["x2"]) <= W
                and 0 <= int(b["y1"]) < int(b["y2"]) <= H
            ):
                raise ValueError(f"Invalid box coordinates: {key}, {b}")

        if int(r["region_count"]) != base.K:
            raise ValueError(f"Incorrect region count: {key}")

        if (
            float(r["max_observed_pair_iou"])
            > threshold + 1e-6
        ):
            raise ValueError(f"IoU threshold exceeded: {key}")

        geometry[key] = r

    expected_keys = {
        (idx, t)
        for idx in range(LIMIT)
        for t in THRESHOLDS
    }

    if set(geometry) != expected_keys:
        raise ValueError("Geometry index/threshold combinations incomplete")

    print("Geometry rows:", len(geometry))
    print("Image mapping: PASS")
    print("Region counts and bounds: PASS")
    print("Configured IoU constraints: PASS")
    return geometry


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate inputs without loading WAM or running attacks",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=LIMIT,
        help="Number of images; use 10 for the initial smoke test",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=str(OUT_DIR / "iou_sensitivity_results.csv"),
    )
    args = parser.parse_args()

    if not 1 <= args.limit <= LIMIT:
        raise ValueError("--limit must be between 1 and 500")

    messages, main_rows = base.load_main_reference()

    all_paths = list(
        iter_images(REPO / "data/eval/coco500_clean", LIMIT)
    )

    if len(all_paths) != LIMIT:
        raise ValueError(f"Expected {LIMIT} image paths")

    if len({p.name for p in all_paths}) != LIMIT:
        raise ValueError("Duplicate image paths")

    if {p.name for p in all_paths} != set(messages):
        raise ValueError("Image paths do not match main experiment")

    geometry = load_geometry(all_paths)
    paths = all_paths[:args.limit]

    print("\n========== PRE-RUN AUDIT ==========")
    print("Images to process:", len(paths))
    print("IoU thresholds:", THRESHOLDS)
    print("Attacks:", ATTACKS)
    print("Expected result rows:",
          len(paths) * len(THRESHOLDS) * len(ATTACKS))
    print("The 0.03 condition will be checked against Hybrid.")

    if args.check_only:
        print("\nCHECK-ONLY PASS. No model loaded.")
        return

    out_csv = Path(args.output).resolve()
    if out_csv.exists():
        raise FileExistsError(
            f"Output already exists: {out_csv}. "
            "Choose a new output path or archive the old file."
        )

    out_csv.parent.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )
    print("Device:", device)
    print("Loading frozen WAM model...")

    wam = load_model_from_checkpoint(
        str(REPO / "checkpoints/params.json"),
        str(REPO / "checkpoints/wam_mit.pth"),
    ).to(device).eval()

    expected_rows = (
        len(paths) * len(THRESHOLDS) * len(ATTACKS)
    )
    written = 0

    # Exclusive creation prevents accidental overwriting of prior results.
    with out_csv.open("x", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS)
        writer.writeheader()

        with torch.no_grad():
            for idx, path in enumerate(paths):
                g_by_threshold = {
                    t: geometry[(idx, t)]
                    for t in THRESHOLDS
                }

                with Image.open(path) as im:
                    img = default_transform(
                        im.convert("RGB")
                    ).unsqueeze(0).to(device)

                H, W = img.shape[-2:]

                for t, g in g_by_threshold.items():
                    if (H, W) != (
                        int(g["height"]), int(g["width"])
                    ):
                        raise ValueError(
                            f"Dimension mismatch: {path.name}, IoU={t}"
                        )

                msg_string = messages[path.name]
                msg = torch.tensor(
                    [[int(bit) for bit in msg_string]],
                    dtype=torch.float32,
                    device=device,
                )

                # One identical full watermarked image for every threshold.
                full_w = wam.embed(img, msg)["imgs_w"]
                center_box = base.center_reference_box_hw(H, W)

                for threshold in THRESHOLDS:
                    g = g_by_threshold[threshold]
                    raw_boxes = json.loads(g["boxes"])
                    boxes = [base.parse_box(b) for b in raw_boxes]

                    method = f"iou_max_{threshold:.2f}"

                    condition = base.make_condition(
                        method,
                        f"max_iou_{threshold:.2f}",
                        boxes,
                        img,
                        full_w,
                        center_box,
                    )

                    geometry_area = float(g["actual_area"])
                    actual_area = condition["actual_area"]

                    if abs(actual_area - geometry_area) > 1e-6:
                        raise ValueError(
                            f"Area mismatch: {path.name}, IoU={threshold}, "
                            f"computed={actual_area}, "
                            f"geometry={geometry_area}"
                        )

                    for attack in ATTACKS:
                        # Identical base seed for the same image and attack
                        # across all five IoU conditions.
                        attack_seed = SEED + idx

                        attacked = apply_attack(
                            condition["watermarked"],
                            img,
                            attack,
                            seed=attack_seed,
                        )

                        pred_orig = original_wam_decode(
                            wam, attacked, msg
                        )

                        det = detect_confidence_voting(
                            wam,
                            attacked,
                            vote_mode="proposed",
                        )
                        pred_vote = det["pred_msg"].to(device)

                        row = {
                            "image": path.name,
                            "message": msg_string,
                            "method": method,
                            "anchor_position": condition["anchor_position"],
                            "attack": attack,
                            "attack_seed": attack_seed,
                            "geometry_image_index": idx,
                            "geometry_seed": int(g["seed"]),
                            "max_iou": threshold,
                            "max_observed_pair_iou": float(
                                g["max_observed_pair_iou"]
                            ),
                            "geometry_region_count": int(
                                g["region_count"]
                            ),
                            "geometry_actual_area": geometry_area,
                            "geometry_union_area_pixels": round(
                                geometry_area * H * W
                            ),
                            "configured_budget": base.BUDGET,
                            "anchor_ratio": base.ANCHOR_RATIO,
                            "actual_area": actual_area,
                            "anchor_area": condition["anchor_area"],
                            "satellite_union_area": condition[
                                "satellite_union_area"
                            ],
                            "anchor_satellite_overlap_area": condition[
                                "anchor_satellite_overlap_area"
                            ],
                            "anchor_center_reference_overlap_area": condition[
                                "anchor_center_reference_overlap_area"
                            ],
                            "satellite_center_reference_overlap_area": condition[
                                "satellite_center_reference_overlap_area"
                            ],
                            "satellite_pairwise_overlap_area": condition[
                                "satellite_pairwise_overlap_area"
                            ],
                            "anchor_satellite_overlap_pixels": int(
                                g["anchor_satellite_overlap_pixels"]
                            ),
                            "satellite_pairwise_overlap_pixels": int(
                                g["satellite_pairwise_overlap_pixels"]
                            ),
                            "anchor_box": condition["anchor_box"],
                            "center_reference_box": condition[
                                "center_reference_box"
                            ],
                            "satellite_boxes": condition["satellite_boxes"],
                            "psnr": condition["psnr"],
                            "ssim": condition["ssim"],
                            "regions_detected": int(
                                det["region_count"].item()
                            ),
                            "orig_bitacc": bit_accuracy(pred_orig, msg),
                            "vote_bitacc": bit_accuracy(pred_vote, msg),
                            "orig_recovery31": recovery_31_32(
                                pred_orig, msg
                            ),
                            "vote_recovery31": recovery_31_32(
                                pred_vote, msg
                            ),
                        }

                        writer.writerow(row)
                        written += 1

                f.flush()

                if (idx + 1) % 25 == 0 or idx + 1 == len(paths):
                    print(
                        f"Processed {idx + 1}/{len(paths)} images; "
                        f"rows={written}"
                    )

    # Audit the completed file, including the default IoU=0.03 condition.
    with out_csv.open(newline="") as f:
        output_rows = list(csv.DictReader(f))

    if len(output_rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, got {len(output_rows)}"
        )

    seen = set()
    for r in output_rows:
        key = (r["image"], r["method"], r["attack"])
        if key in seen:
            raise ValueError(f"Duplicate result key: {key}")
        seen.add(key)

    control_rows = {
        (r["image"], r["attack"]): r
        for r in output_rows
        if r["method"] == "iou_max_0.03"
    }

    metric_cols = [
        "actual_area",
        "orig_bitacc",
        "vote_bitacc",
        "orig_recovery31",
        "vote_recovery31",
    ]
    max_diffs = {c: 0.0 for c in metric_cols}

    for key, new_row in control_rows.items():
        if key not in main_rows:
            raise ValueError(f"Missing Hybrid reference: {key}")

        ref = main_rows[key]

        if new_row["message"] != ref["message"]:
            raise ValueError(f"Message mismatch: {key}")

        for col in metric_cols:
            diff = abs(float(new_row[col]) - float(ref[col]))
            max_diffs[col] = max(max_diffs[col], diff)

    print("\n========== IOU SENSITIVITY COMPLETE ==========")
    print("Images:", len(paths))
    print("Rows:", len(output_rows))
    print("Expected rows:", expected_rows)
    print("Output:", out_csv)

    print("\n========== IOU=0.03 HYBRID AUDIT ==========")
    for col, diff in max_diffs.items():
        print(f"{col}: max_abs_diff={diff:.10f}")

    if max(max_diffs.values()) > 1e-5:
        raise RuntimeError(
            "IoU=0.03 does not reproduce Hybrid. "
            "Do not interpret the sensitivity results until investigated."
        )

    print("PASS: IoU=0.03 reproduces the Hybrid reference.")

    print("\n========== FOUR-ATTACK DESCRIPTIVE MEANS ==========")
    for threshold in THRESHOLDS:
        method = f"iou_max_{threshold:.2f}"
        subset = [
            r for r in output_rows
            if r["method"] == method and r["attack"] != "none"
        ]

        def mean(column):
            return sum(float(r[column]) for r in subset) / len(subset)

        print(
            f"IoU={threshold:.2f} | "
            f"area={mean('actual_area'):.6f} | "
            f"orig_BitAcc={mean('orig_bitacc'):.6f} | "
            f"vote_BitAcc={mean('vote_bitacc'):.6f} | "
            f"orig_Recovery31={mean('orig_recovery31'):.6f} | "
            f"vote_Recovery31={mean('vote_recovery31'):.6f}"
        )

    print("\nPASS: Result keys are unique and all rows are present.")
    print("Existing main experiment files were not modified.")


if __name__ == "__main__":
    main()
