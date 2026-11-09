import argparse
import csv
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
    RegionBox,
    boxes_to_masks,
    select_regions,
    detect_confidence_voting,
    bit_accuracy,
    recovery_31_32,
    _area_to_side,
)
from scripts.run_fixed_budget_exp import (
    apply_attack,
    original_wam_decode,
    iter_images,
    psnr01,
    ssim01,
)

SEED = 3
LIMIT = 500
BUDGET = 0.20
K = 4
ANCHOR_RATIO = 0.40

ATTACKS = [
    "none",
    "erase40",
    "randomcrop50",
    "cornercrop50",
    "splicing20",
]

ROOT = REPO / "outputs/revision_202610"
MAIN_CSV = ROOT / "raw/coco500_clean_allattacks/results.csv"
GEOMETRY_CSV = (
    ROOT / "anchor_position_matched_20261028/matched_geometry_native_500.csv"
)
OUT_DIR = ROOT / "anchor_position_matched_20261028"
OUT_CSV = OUT_DIR / "anchor_position_matched_results.csv"


def box_record(box):
    return {
        "x1": int(box.x1),
        "y1": int(box.y1),
        "x2": int(box.x2),
        "y2": int(box.y2),
    }


def parse_box(value, score=0.0):
    if isinstance(value, str):
        value = json.loads(value)
    return RegionBox(
        int(value["x1"]),
        int(value["y1"]),
        int(value["x2"]),
        int(value["y2"]),
        score,
    )


def intersection_area(a, b):
    w = max(0, min(a.x2, b.x2) - max(a.x1, b.x1))
    h = max(0, min(a.y2, b.y2) - max(a.y1, b.y1))
    return w * h


def center_reference_box_hw(H, W):
    side = _area_to_side(H, W, BUDGET * ANCHOR_RATIO)
    x1 = max(0, (W - side) // 2)
    y1 = max(0, (H - side) // 2)
    return RegionBox(x1, y1, x1 + side, y1 + side, 1.0)


def load_main_reference():
    if not MAIN_CSV.exists():
        raise FileNotFoundError(MAIN_CSV)

    messages = {}
    rows = {}

    with MAIN_CSV.open(newline="") as f:
        for r in csv.DictReader(f):
            if r["method"] != "hybrid":
                continue

            key = (r["image"], r["attack"])
            if key in rows:
                raise ValueError(f"Duplicate main result: {key}")

            msg = r["message"]
            if len(msg) != 32 or set(msg) - {"0", "1"}:
                raise ValueError(f"Invalid 32-bit message: {key}")

            if r["image"] in messages and messages[r["image"]] != msg:
                raise ValueError(f"Inconsistent message: {r['image']}")

            messages[r["image"]] = msg
            rows[key] = r

    expected = LIMIT * len(ATTACKS)
    if len(rows) != expected:
        raise ValueError(
            f"Expected {expected} Hybrid reference rows, got {len(rows)}"
        )

    if len(messages) != LIMIT:
        raise ValueError(
            f"Expected {LIMIT} unique images, got {len(messages)}"
        )

    for image in messages:
        attacks = {
            attack for (name, attack) in rows if name == image
        }
        if attacks != set(ATTACKS):
            raise ValueError(f"Incomplete attack results: {image}")

    return messages, rows


def load_and_validate_geometry(paths):
    if not GEOMETRY_CSV.exists():
        raise FileNotFoundError(GEOMETRY_CSV)

    with GEOMETRY_CSV.open(newline="") as f:
        raw_rows = list(csv.DictReader(f))

    if len(raw_rows) != LIMIT:
        raise ValueError(
            f"Expected {LIMIT} geometry rows, got {len(raw_rows)}"
        )

    geometry = {}

    for raw in raw_rows:
        idx = int(raw["image_index"])

        if idx in geometry or not 0 <= idx < LIMIT:
            raise ValueError(f"Invalid or duplicate image_index: {idx}")

        if int(raw["seed"]) != SEED + idx:
            raise ValueError(f"Unexpected geometry seed at index {idx}")

        H, W = int(raw["height"]), int(raw["width"])
        if H <= 0 or W <= 0:
            raise ValueError(f"Invalid dimensions at index {idx}")

        center = parse_box(raw["center_anchor"], 1.0)
        offcenter = parse_box(raw["offcenter_anchor"], 1.0)
        satellites = [
            parse_box(b)
            for b in json.loads(raw["satellite_boxes"])
        ]

        if len(satellites) != 3:
            raise ValueError(f"Expected 3 satellites at index {idx}")

        expected_center = center_reference_box_hw(H, W)
        if box_record(center) != box_record(expected_center):
            raise ValueError(
                f"Center anchor mismatch at geometry index {idx}"
            )

        anchor_side = int(raw["anchor_side"])
        satellite_side = int(raw["satellite_side"])

        for anchor in (center, offcenter):
            if (
                anchor.x2 - anchor.x1 != anchor_side
                or anchor.y2 - anchor.y1 != anchor_side
            ):
                raise ValueError(f"Invalid anchor dimensions at index {idx}")

            if (
                anchor.x1 < 0 or anchor.y1 < 0
                or anchor.x2 > W or anchor.y2 > H
            ):
                raise ValueError(f"Anchor out of bounds at index {idx}")

        # The two anchor positions must be spatially distinct.
        if intersection_area(center, offcenter) != 0:
            raise ValueError(
                f"Center/off-center anchors overlap at index {idx}"
            )

        for sat in satellites:
            if (
                sat.x2 - sat.x1 != satellite_side
                or sat.y2 - sat.y1 != satellite_side
            ):
                raise ValueError(
                    f"Invalid satellite dimensions at index {idx}"
                )

            if (
                sat.x1 < 0 or sat.y1 < 0
                or sat.x2 > W or sat.y2 > H
            ):
                raise ValueError(
                    f"Satellite out of bounds at index {idx}"
                )

            for anchor in (center, offcenter):
                if intersection_area(sat, anchor) != 0:
                    raise ValueError(
                        f"Satellite overlaps an anchor at index {idx}"
                    )

        for i, sat in enumerate(satellites):
            for other in satellites[i + 1:]:
                if intersection_area(sat, other) != 0:
                    raise ValueError(
                        f"Satellite overlap at index {idx}"
                    )

        boxes = [center, *satellites]
        union_pixels = sum(b.area for b in boxes)

        if union_pixels != int(raw["union_area_pixels"]):
            raise ValueError(
                f"Union area mismatch at geometry index {idx}"
            )

        area = union_pixels / (H * W)
        if abs(area - float(raw["actual_area"])) > 1e-8:
            raise ValueError(
                f"Actual area mismatch at geometry index {idx}"
            )

        geometry[idx] = {
            "seed": int(raw["seed"]),
            "height": H,
            "width": W,
            "center_anchor": center,
            "offcenter_anchor": offcenter,
            "satellite_boxes": satellites,
            "union_area_pixels": union_pixels,
            "actual_area": area,
        }

    if set(geometry) != set(range(LIMIT)):
        raise ValueError("Geometry indices are incomplete")

    if len(paths) != LIMIT:
        raise ValueError(f"Expected {LIMIT} image paths, got {len(paths)}")

    print("Geometry rows:", len(geometry))
    print("Anchor/satellite geometry: PASS")
    print("Unique image paths:", len({p.name for p in paths}))
    print(
        "Image index mapping uses the iter_images order. "
        "The geometry file does not contain image filenames."
    )

    print("\nFirst five index-to-filename mappings:")
    for idx, path in enumerate(paths[:5]):
        print(
            f"index={idx}, seed={geometry[idx]['seed']}, "
            f"image={path.name}"
        )

    return geometry


def make_condition(
    method,
    anchor_position,
    boxes,
    img,
    full_w,
    center_box,
):
    H, W = img.shape[-2:]

    if len(boxes) != K:
        raise ValueError(f"{method}: expected {K} boxes")

    masks = boxes_to_masks(
        boxes, H, W, img.device, img.dtype
    )

    anchor_mask = masks[0:1]
    satellite_union = (
        masks[1:].sum(dim=0, keepdim=True).clamp(0, 1)
    )
    union = masks.sum(dim=0, keepdim=True).clamp(0, 1)

    center_mask = boxes_to_masks(
        [center_box], H, W, img.device, img.dtype
    )

    anchor_satellite_overlap = float(
        (anchor_mask * satellite_union).mean().item()
    )
    anchor_center_overlap = float(
        (anchor_mask * center_mask).mean().item()
    )
    satellite_center_overlap = float(
        (satellite_union * center_mask).mean().item()
    )

    satellite_pairwise_pixels = sum(
        intersection_area(boxes[i], boxes[j])
        for i in range(1, len(boxes))
        for j in range(i + 1, len(boxes))
    )

    watermarked = full_w * union + img * (1.0 - union)

    return {
        "method": method,
        "anchor_position": anchor_position,
        "watermarked": watermarked,
        "actual_area": float(union.mean().item()),
        "anchor_area": float(anchor_mask.mean().item()),
        "satellite_union_area": float(satellite_union.mean().item()),
        "anchor_satellite_overlap_area": anchor_satellite_overlap,
        "anchor_center_reference_overlap_area": anchor_center_overlap,
        "satellite_center_reference_overlap_area": satellite_center_overlap,
        "satellite_pairwise_overlap_area": (
            satellite_pairwise_pixels / (H * W)
        ),
        "anchor_box": json.dumps(
            box_record(boxes[0]), separators=(",", ":")
        ),
        "center_reference_box": json.dumps(
            box_record(center_box), separators=(",", ":")
        ),
        "satellite_boxes": json.dumps(
            [box_record(b) for b in boxes[1:]],
            separators=(",", ":"),
        ),
        "psnr": psnr01(watermarked, img),
        "ssim": ssim01(watermarked, img),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--check-only",
        action="store_true",
        help="Validate inputs and geometry without loading WAM or running attacks",
    )
    args = parser.parse_args()

    messages, main_rows = load_main_reference()
    paths = iter_images(REPO / "data/eval/coco500_clean", LIMIT)
    paths = list(paths)

    if len({p.name for p in paths}) != LIMIT:
        raise ValueError("Image paths are incomplete or duplicated")

    if set(p.name for p in paths) != set(messages):
        raise ValueError("Image list does not match the main experiment")

    geometry = load_and_validate_geometry(paths)

    print("\n========== PRE-RUN AUDIT ==========")
    print("Main Hybrid reference rows:", len(main_rows))
    print("Images:", len(paths))
    print("Attacks per image:", len(ATTACKS))
    print("Matched conditions: center vs off-center")
    print("Geometry preflight: PASS")

    if args.check_only:
        print("\nCHECK-ONLY COMPLETE. No model loaded; no GPU experiment run.")
        return

    if OUT_CSV.exists():
        raise FileExistsError(
            f"Output already exists: {OUT_CSV}. "
            "Inspect or archive it before rerunning."
        )

    OUT_DIR.mkdir(parents=True, exist_ok=True)

    random.seed(SEED)
    torch.manual_seed(SEED)

    device = torch.device(
        "cuda" if torch.cuda.is_available() else "cpu"
    )

    print("\nLoading frozen WAM model...")
    wam = load_model_from_checkpoint(
        str(REPO / "checkpoints/params.json"),
        str(REPO / "checkpoints/wam_mit.pth"),
    ).to(device).eval()

    fields = [
        "image", "message", "method", "anchor_position", "attack",
        "geometry_image_index", "geometry_seed", "geometry_union_area_pixels",
        "configured_budget", "anchor_ratio", "actual_area",
        "anchor_area", "satellite_union_area",
        "anchor_satellite_overlap_area",
        "anchor_center_reference_overlap_area",
        "satellite_center_reference_overlap_area",
        "satellite_pairwise_overlap_area",
        "anchor_box", "center_reference_box", "satellite_boxes",
        "psnr", "ssim", "regions_detected",
        "orig_bitacc", "vote_bitacc",
        "orig_recovery31", "vote_recovery31",
    ]

    expected_rows = LIMIT * 3 * len(ATTACKS)

    with OUT_CSV.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()

        with torch.no_grad():
            for idx, path in enumerate(paths):
                g = geometry[idx]

                with Image.open(path) as im:
                    img = default_transform(
                        im.convert("RGB")
                    ).unsqueeze(0).to(device)

                H, W = img.shape[-2:]
                if (H, W) != (g["height"], g["width"]):
                    raise ValueError(
                        f"Image/geometry dimension mismatch: {path.name}"
                    )

                msg_string = messages[path.name]
                msg = torch.tensor(
                    [[int(bit) for bit in msg_string]],
                    dtype=torch.float32,
                    device=device,
                )

                # Embed once; all three conditions use the same message
                # and the same frozen WAM embedder.
                full_w = wam.embed(img, msg)["imgs_w"]
                center_box = center_reference_box_hw(H, W)

                # Condition 1: original Hybrid control.
                hybrid_masks, hybrid_boxes = select_regions(
                    img,
                    strategy="hybrid",
                    total_area=BUDGET,
                    num_regions=K,
                    seed=SEED + idx,
                    anchor_ratio=ANCHOR_RATIO,
                )

                if len(hybrid_boxes) != K:
                    raise ValueError(
                        f"Unexpected Hybrid region count: {path.name}"
                    )

                if box_record(hybrid_boxes[0]) != box_record(center_box):
                    raise ValueError(
                        f"Hybrid center anchor mismatch: {path.name}"
                    )

                hybrid_condition = make_condition(
                    "hybrid_control",
                    "center",
                    hybrid_boxes,
                    img,
                    full_w,
                    center_box,
                )

                # Conditions 2 and 3 share exactly the same satellite boxes.
                matched_conditions = []

                for method, anchor_position, anchor in [
                    (
                        "center_anchor_matched",
                        "center",
                        g["center_anchor"],
                    ),
                    (
                        "offcenter_anchor_matched",
                        "offcenter",
                        g["offcenter_anchor"],
                    ),
                ]:
                    boxes = [anchor, *g["satellite_boxes"]]

                    condition = make_condition(
                        method,
                        anchor_position,
                        boxes,
                        img,
                        full_w,
                        center_box,
                    )

                    if abs(
                        condition["actual_area"] - g["actual_area"]
                    ) > 1e-6:
                        raise ValueError(
                            f"Matched area mismatch: {path.name}, {method}"
                        )

                    matched_conditions.append(condition)

                conditions = [
                    hybrid_condition,
                    *matched_conditions,
                ]

                # Both matched conditions must have identical satellite
                # coordinates and exactly the same geometric area.
                if (
                    matched_conditions[0]["satellite_boxes"]
                    != matched_conditions[1]["satellite_boxes"]
                ):
                    raise ValueError(
                        f"Satellite mismatch: {path.name}"
                    )

                if abs(
                    matched_conditions[0]["actual_area"]
                    - matched_conditions[1]["actual_area"]
                ) > 1e-7:
                    raise ValueError(
                        f"Matched condition area mismatch: {path.name}"
                    )

                for condition in conditions:
                    watermarked = condition["watermarked"]

                    for attack in ATTACKS:
                        attacked = apply_attack(
                            watermarked,
                            img,
                            attack,
                            seed=SEED + idx,
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
                            "method": condition["method"],
                            "anchor_position": condition["anchor_position"],
                            "attack": attack,
                            "geometry_image_index": idx,
                            "geometry_seed": g["seed"],
                            "geometry_union_area_pixels": (
                                g["union_area_pixels"]
                            ),
                            "configured_budget": BUDGET,
                            "anchor_ratio": ANCHOR_RATIO,
                            "actual_area": condition["actual_area"],
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

                f.flush()

                if (idx + 1) % 25 == 0:
                    print(f"Processed {idx + 1}/{LIMIT} images")

    with OUT_CSV.open(newline="") as f:
        output_rows = list(csv.DictReader(f))

    if len(output_rows) != expected_rows:
        raise ValueError(
            f"Expected {expected_rows} rows, got {len(output_rows)}"
        )

    methods = {
        "hybrid_control",
        "center_anchor_matched",
        "offcenter_anchor_matched",
    }

    seen = set()
    for r in output_rows:
        key = (r["image"], r["method"], r["attack"])
        if key in seen:
            raise ValueError(f"Duplicate output key: {key}")
        seen.add(key)

        if r["method"] not in methods:
            raise ValueError(f"Unexpected method: {r['method']}")

    for method in methods:
        subset = [r for r in output_rows if r["method"] == method]

        if len(subset) != LIMIT * len(ATTACKS):
            raise ValueError(f"Incomplete method results: {method}")

        if len({r["image"] for r in subset}) != LIMIT:
            raise ValueError(f"Image count mismatch: {method}")

    # Verify that rerunning Hybrid reproduces the existing main results.
    control_rows = {
        (r["image"], r["attack"]): r
        for r in output_rows
        if r["method"] == "hybrid_control"
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
            raise ValueError(f"Missing main reference row: {key}")

        ref = main_rows[key]

        if new_row["message"] != ref["message"]:
            raise ValueError(f"Message mismatch: {key}")

        for col in metric_cols:
            diff = abs(float(new_row[col]) - float(ref[col]))
            max_diffs[col] = max(max_diffs[col], diff)

    print("\n========== POSITION-MATCHED ABLATION COMPLETE ==========")
    print("Images:", LIMIT)
    print("Methods:", sorted(methods))
    print("Rows:", len(output_rows))
    print("Expected rows:", expected_rows)
    print("Output:", OUT_CSV)

    print("\n========== HYBRID CONTROL AUDIT ==========")
    for col, diff in max_diffs.items():
        print(f"{col}: max_abs_diff={diff:.10f}")

    if max(max_diffs.values()) > 1e-5:
        raise RuntimeError(
            "Hybrid control does not match the main results. "
            "Investigate reproducibility before interpreting the ablation."
        )

    print("PASS: Hybrid control matches the existing main results.")
    print("PASS: All methods have complete paired image/attack records.")
    print("PASS: Matched conditions share satellite geometry and area.")
    print("Existing main experiment files were not modified.")


if __name__ == "__main__":
    main()
